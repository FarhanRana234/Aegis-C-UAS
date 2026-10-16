"""Aegis C-UAS v2: dual-sensor (thermal IR + optical RGB) single-target tracker.

Changes from v1:
  - DBSCAN swarm clustering removed. One target is locked at a time.
  - Target priority score with switching hysteresis (no box jumping).
  - Kalman filter on the locked target: smooth position, predicted coasting, optional lead.
  - Tracker logic now runs once per NEW frame (v1 ran it every loop iteration).
  - Optional aim output to an ESP32 over serial (aim only; no launcher control here).
  - Sources can be video files, camera indexes (0, 1) or stream URLs (e.g. ESP32-CAM).
  - Headless mode, per-stream inference timing, everything tunable from config.json.
  - Operator display: ranked target list with scores, swarm alert banner (simple
    distance grouping, no scikit-learn), click a box to force the lock.
    Keys: q quit | a back to auto-select | r release lock.

Usage: python main.py [config.json]      Press q in the window (or Ctrl+C) to quit.
"""
import json
import math
import struct
import sys
import time

import cv2
import numpy as np
import torch
import torch_directml
from ultralytics import YOLO

with open(sys.argv[1] if len(sys.argv) > 1 else "config.json") as f:
    cfg = json.load(f)

W, H = cfg["frame_width"], cfg["frame_height"]
DIAG = math.hypot(W, H)
BANNER_H = 45

if cfg.get("device") in ["cuda", "dml", "gpu"]:
    device = torch_directml.device()
else:
    device = "cpu"

def largest_group_size(dets, eps):
    """Size of the biggest cluster of detections whose centres chain within eps pixels.
    Display only: this never influences which target is locked."""
    n = len(dets)
    if n < 2:
        return n
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if math.hypot(dets[i]["cx"] - dets[j]["cx"], dets[i]["cy"] - dets[j]["cy"]) <= eps:
                parent[find(i)] = find(j)
    sizes = {}
    for i in range(n):
        r = find(i)
        sizes[r] = sizes.get(r, 0) + 1
    return max(sizes.values())


# --------------------------------------------------------------------------
# Kalman filter (constant velocity, time step = one processed frame)
# --------------------------------------------------------------------------
class KalmanCV:
    def __init__(self, accel_var, meas_sigma):
        self.F = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        G = np.array([[0.5, 0], [0, 0.5], [1, 0], [0, 1]], float)
        self.Q = accel_var * (G @ G.T)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.R = np.eye(2) * meas_sigma ** 2
        self.x = np.zeros(4)
        self.P = np.eye(4)

    def reset(self, cx, cy):
        self.x = np.array([cx, cy, 0.0, 0.0])
        self.P = np.diag([25.0, 25.0, 400.0, 400.0])

    def predict(self):
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q

    def update(self, cx, cy):
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.array([cx, cy]) - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P


# --------------------------------------------------------------------------
# Per-stream tracker: filters, target priority, lock, coast
# --------------------------------------------------------------------------
class StreamTracker:
    def __init__(self, group_eps):
        self.eps = group_eps
        self.manual = False          # operator forced the lock
        self.dets, self.ranked, self.group = [], [], 0
        self.history, self.spawn = {}, {}
        self.kf = KalmanCV(cfg["kalman_accel_var"], cfg["kalman_meas_sigma"])
        self.locked, self.locked_id = False, None
        self.missing, self.challenge = 0, 0
        self.last_wh = (0.0, 0.0)
        self.status = "SEARCHING"
        self.box, self.center = None, None

    # ---- filtering (size, aspect ratio, static-clutter gate) ----
    def filter_detections(self, boxes):
        dets, seen = [], set()
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
                w, h = x2 - x1, y2 - y1
                
                # --- DEBUG LOGGING ---
                if w < cfg["min_box_width"] or h < cfg["min_box_height"]:
                    print(f"[DROP] Box too small: {w:.1f}x{h:.1f} (min {cfg['min_box_width']})")
                    continue
                ar = w / (h + 1e-5)
                if ar < cfg["min_aspect_ratio"] or ar > cfg["max_aspect_ratio"]:
                    print(f"[DROP] Aspect ratio out of bounds: {ar:.2f}")
                    continue
                
                tid = int(ids[i]) if ids is not None else -(i + 1)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                seen.add(tid)
                hist = self.history.setdefault(tid, [])
                self.spawn.setdefault(tid, (cx, cy))
                hist.append((cx, cy))
                if len(hist) > 30:
                    hist.pop(0)
                
                if (cfg["static_gate_enabled"] and tid != self.locked_id
                        and len(hist) >= cfg["static_gate_frames"]):
                    sx, sy = self.spawn[tid]
                    if math.hypot(cx - sx, cy - sy) < cfg["static_gate_px"]:
                        print(f"[DROP] Static clutter gate triggered for id {tid}")
                        continue
                
                dets.append(dict(tid=tid, cx=cx, cy=cy, w=w, h=h, conf=float(confs[i]),
                                 age=len(hist), box=(x1, y1, x2, y2)))
        for k in [k for k in self.history if k not in seen]:
            self.history.pop(k, None)
            self.spawn.pop(k, None)
        return dets

    # ---- target priority: bigger, nearer the centre, more confident, older ----
    def score(self, d):
        w = cfg["priority_weights"]
        size = min(math.sqrt(d["w"] * d["h"]) / (cfg["priority_size_ref"] * DIAG), 1.0)
        centre = 1 - math.hypot(d["cx"] - W / 2, d["cy"] - H / 2) / (DIAG / 2)
        age = min(d["age"] / 30.0, 1.0)
        return w["size"] * size + w["center"] * centre + w["conf"] * d["conf"] + w["age"] * age

    def _lock(self, d):
        self.kf.reset(d["cx"], d["cy"])
        self.locked, self.locked_id = True, d["tid"]
        self.last_wh = (d["w"], d["h"])
        self.missing, self.challenge = 0, 0
        self.status = self._locked_status()

    def _locked_status(self):
        return "LOCKED (MANUAL)" if self.manual else "LOCKED"

    def manual_lock(self, det):
        self._lock(det)
        self.manual = True
        self.status = self._locked_status()
        self._publish()

    def _release(self):
        self.locked, self.locked_id = False, None
        self.manual = False
        self.missing, self.challenge = 0, 0
        self.status = "SEARCHING"

    def step(self, dets):
        if self.locked:
            self.kf.predict()
            px, py = self.kf.x[0], self.kf.x[1]
            match = next((d for d in dets if d["tid"] == self.locked_id), None)
            if match is None and dets:  # ByteTrack may have re-issued the ID
                near = min(dets, key=lambda d: math.hypot(d["cx"] - px, d["cy"] - py))
                if math.hypot(near["cx"] - px, near["cy"] - py) <= cfg["gate_px"]:
                    match = near
            if match is not None:
                self.kf.update(match["cx"], match["cy"])
                self.locked_id = match["tid"]
                self.last_wh = (match["w"], match["h"])
                self.missing, self.status = 0, self._locked_status()
                others = [d for d in dets if d is not match]
                best = max(others, key=self.score) if others else None
                if (not self.manual and best
                        and self.score(best) > self.score(match) * (1 + cfg["switch_margin"])):
                    self.challenge += 1
                else:
                    self.challenge = 0
                if not self.manual and best and self.challenge >= cfg["switch_hold_frames"]:
                    self._lock(best)
            else:
                self.missing += 1
                self.kf.x[2:] *= 0.9  # damp velocity so a long coast does not run away
                if self.missing > cfg["max_coast_frames"]:
                    self._release()
                else:
                    self.status = f"COASTING ({self.missing})"
        elif dets:
            self._lock(max(dets, key=self.score))
        self.dets = dets
        self.ranked = sorted(((self.score(d), d) for d in dets), key=lambda t: t[0], reverse=True)
        self.group = largest_group_size(dets, self.eps)
        self._publish()

    def _publish(self):
        if not self.locked:
            self.box = self.center = None
            return
        lead = cfg["lead_frames"]
        cx = self.kf.x[0] + self.kf.x[2] * lead
        cy = self.kf.x[1] + self.kf.x[3] * lead
        w, h = self.last_wh
        self.box = (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)
        self.center = (cx, cy)


# --------------------------------------------------------------------------
# ESP32 serial link (aim only). Launcher stays under ESP32 firmware + hardware arm switch.
# Packet: 0xA5 | cmd(1) seq(2) pan_mrad(2) tilt_mrad(2) flags(1) | xor checksum
# flags bit0 = track valid. ESP32 should zero motion if no packet for ~150 ms.
# --------------------------------------------------------------------------
class Esp32Link:
    def __init__(self, c):
        self.ser, self.seq = None, 0
        if c.get("enabled"):
            import serial
            self.ser = serial.Serial(c["port"], c["baud"], timeout=0)

    def send(self, pan_rad, tilt_rad, valid):
        if self.ser is None:
            return
        body = struct.pack("<BHhhB", 0x01, self.seq & 0xFFFF,
                           int(pan_rad * 1000), int(tilt_rad * 1000), 1 if valid else 0)
        crc = 0
        for b in body:
            crc ^= b
        self.ser.write(bytes([0xA5]) + body + bytes([crc]))
        self.seq += 1

    def close(self):
        if self.ser:
            self.ser.close()


def px_to_angles(cx, cy):
    fx = (W / 2) / math.tan(math.radians(cfg["camera_hfov_deg"]) / 2)
    pan = math.atan((cx - W / 2) / fx)
    tilt = math.atan((cy - H / 2) / fx)
    return pan, (-tilt if cfg["invert_tilt"] else tilt)


# --------------------------------------------------------------------------
# I/O helpers
# --------------------------------------------------------------------------
def open_source(s):
    cap = cv2.VideoCapture(int(s) if str(s).isdigit() else s)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def read_frame(cap):
    ok, fr = cap.read()
    if not ok and cfg["loop_videos"]:
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ok, fr = cap.read()
    return cv2.resize(fr, (W, H)) if ok else None


def infer(model, frame, conf):
    return model.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml",
                       imgsz=cfg["inference_imgsz"], verbose=False,
                       device=device)[0].boxes

def render(frame, tr, title):
    cv2.line(frame, (W // 2 - 12, H // 2), (W // 2 + 12, H // 2), (90, 90, 90), 1)
    cv2.line(frame, (W // 2, H // 2 - 12), (W // 2, H // 2 + 12), (90, 90, 90), 1)
    rank_of = {}
    for rank, (_, d) in enumerate(tr.ranked, 1):
        rank_of[d["tid"]] = rank
        if tr.locked and d["tid"] == tr.locked_id:
            continue  # drawn below as the locked target
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 1)
        cv2.putText(frame, f"T{rank}", (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if tr.box is not None:
        x1, y1, x2, y2 = map(int, tr.box)
        color = (0, 0, 255) if tr.status.startswith("LOCKED") else (0, 165, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        c = (int(tr.center[0]), int(tr.center[1]))
        cv2.circle(frame, c, 5, (0, 255, 0), -1)
        cv2.putText(frame, f"T{rank_of.get(tr.locked_id, '-')} TARGET {c}", (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(frame, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(frame, f"STATUS: {tr.status}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2)
    return frame


def panel(ir_tr, rgb_tr):
    rows = cfg["panel_rows"]
    ph = 28 + rows * 18 + 8
    img = np.full((ph, W * 2, 3), 18, np.uint8)
    for k, (label, tr) in enumerate((("IR", ir_tr), ("RGB", rgb_tr))):
        x0 = k * W + 15
        cv2.putText(img, f"{label} TARGETS: {len(tr.dets)}   LARGEST GROUP: {tr.group}",
                    (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        for i, (sc, d) in enumerate(tr.ranked[:rows], 1):
            mark = ">" if (tr.locked and d["tid"] == tr.locked_id) else " "
            line = f"{mark} T{i}  id {d['tid']:<4} score {sc:.2f}  conf {d['conf']:.2f}  ({d['cx']:.0f},{d['cy']:.0f})"
            col = (0, 0, 255) if mark == ">" else (210, 210, 210)
            cv2.putText(img, line, (x0, 28 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.line(img, (W, 0), (W, ph), (50, 50, 50), 2)
    return img


def compose(disp_ir, disp_rgb, now, ir_tr, rgb_tr):
    feeds = np.hstack((disp_ir, disp_rgb))
    cv2.line(feeds, (W, 0), (W, H), (50, 50, 50), 3)
    banner = np.full((BANNER_H, W * 2, 3), 25, np.uint8)
    cv2.putText(banner, "AEGIS C-UAS COMMAND & CONTROL SYSTEM", (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    n = max(ir_tr.group, rgb_tr.group)
    if n >= cfg["swarm_alert_size"]:
        cv2.putText(banner, f"SWARM ALERT x{n}", (450, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now))
    cv2.putText(banner, f"SYS TIME: {ts} | MODE: ACTIVE TRACKING", (W + 40, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
    return np.vstack((banner, feeds, panel(ir_tr, rgb_tr)))


def select_at(x, y, ir_tr, rgb_tr):
    """Click in the combined window -> force-lock the target box under the cursor."""
    y -= BANNER_H
    if not (0 <= y < H) or not (0 <= x < 2 * W):
        return
    tr, x = (ir_tr, x) if x < W else (rgb_tr, x - W)
    hits = [d for _, d in tr.ranked
            if d["box"][0] - 6 <= x <= d["box"][2] + 6 and d["box"][1] - 6 <= y <= d["box"][3] + 6]
    if hits:
        tr.manual_lock(min(hits, key=lambda d: d["w"] * d["h"]))


# --------------------------------------------------------------------------
def main():
    print("Starting Aegis C-UAS v2...")
    ir_model, rgb_model = YOLO(cfg["thermal_model_path"]), YOLO(cfg["rgb_model_path"])
    cap_ir, cap_rgb = open_source(cfg["ir_source"]), open_source(cfg["rgb_source"])
    iv_ir = 1.0 / (cap_ir.get(cv2.CAP_PROP_FPS) or 30)
    iv_rgb = 1.0 / (cap_rgb.get(cv2.CAP_PROP_FPS) or 30)
    ir_tr = StreamTracker(cfg["group_eps"]["ir"])
    rgb_tr = StreamTracker(cfg["group_eps"]["rgb"])
    link = Esp32Link(cfg["esp32"])
    show = cfg["show_window"]
    win = "Aegis C-UAS Command & Control Interface"
    if show:
        cv2.namedWindow(win)
        cv2.setMouseCallback(
            win, lambda ev, x, y, fl, p: ev == cv2.EVENT_LBUTTONDOWN and select_at(x, y, ir_tr, rgb_tr))
    disp_ir = np.zeros((H, W, 3), np.uint8)
    disp_rgb = disp_ir.copy()
    next_ir = next_rgb = time.time()
    ms = {"IR": 0.0, "RGB": 0.0}
    t_log = time.time()

    try:
        while True:
            now, updated = time.time(), False

            if now >= next_ir:
                fr = read_frame(cap_ir)
                if fr is None:
                    break
                next_ir = now + iv_ir
                t0 = time.time()
                boxes = infer(ir_model, fr, cfg["thermal_confidence"])
                ms["IR"] = (time.time() - t0) * 1000
                ir_tr.step(ir_tr.filter_detections(boxes))
                disp_ir = render(fr, ir_tr, "SENSOR 01: THERMAL IR")
                updated = True

            if now >= next_rgb:
                fr = read_frame(cap_rgb)
                if fr is None:
                    break
                next_rgb = now + iv_rgb
                t0 = time.time()
                boxes = infer(rgb_model, fr, cfg["rgb_confidence"])
                ms["RGB"] = (time.time() - t0) * 1000
                rgb_tr.step(rgb_tr.filter_detections(boxes))
                disp_rgb = render(fr, rgb_tr, "SENSOR 02: OPTICAL RGB")
                updated = True

            if updated:
                order = (rgb_tr, ir_tr) if cfg["aim_sensor"] == "rgb" else (ir_tr, rgb_tr)
                src = (next((t for t in order if t.locked and t.manual), None)
                       or next((t for t in order if t.locked), None))
                if src:
                    pan, tilt = px_to_angles(*src.center)
                    link.send(pan, tilt, True)
                else:
                    link.send(0.0, 0.0, False)
                if show:
                    cv2.imshow(win, compose(disp_ir, disp_rgb, now, ir_tr, rgb_tr))

            if show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("a"):      # back to automatic target selection
                    ir_tr.manual = rgb_tr.manual = False
                if key == ord("r"):      # release both locks
                    ir_tr._release()
                    rgb_tr._release()
            time.sleep(0.001)

            if now - t_log > 2.0:
                print(f"inference ms  IR {ms['IR']:.0f} | RGB {ms['RGB']:.0f}   "
                      f"IR {ir_tr.status} | RGB {rgb_tr.status}")
                t_log = now
    except KeyboardInterrupt:
        pass
    finally:
        link.send(0.0, 0.0, False)
        link.close()
        cap_ir.release()
        cap_rgb.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
