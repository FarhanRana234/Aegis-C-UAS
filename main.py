"""C-UAV v2: dual-sensor (thermal IR + optical RGB) single-target tracker, ONNX / CPU build.

Interactive UI (click or hotkey):
  [ AIM: RGB/IR ]  (s)      switch the sensor used for aiming
  [ AUTO: ON/OFF ] (space)  automatic interception (needs ARM, a stable lock and the target on the crosshair)
  [ RELEASE ]      (r)      clear all target locks
  [ LOGS: ON/OFF ] (l)      show / hide the on-screen terminal
  [ ARM: ON/OFF ]  (x)      safety arm. Starts OFF every run. Fire commands are blocked while disarmed
  [ FIRE! ]        (f)      manual interception command (needs ARM)
  Click a bounding box to force-lock it.  (a) returns to automatic target selection.  (q) quits.

Models: ONNX exports run on CPU through ONNX Runtime (see export_models.py). .pt files still work.
Usage:  python main.py [config.json]      Ctrl+C also quits.
"""
import json
import math
import os
import struct
import sys
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

with open(sys.argv[1] if len(sys.argv) > 1 else "config.json") as f:
    cfg = json.load(f)

W, H = cfg["frame_width"], cfg["frame_height"]
DIAG = math.hypot(W, H)
BANNER_H = 45
DEVICE = cfg.get("device", "cpu")  # ONNX models run on CPU via ONNX Runtime


# --------------------------------------------------------------------------
# Dual-destination logging (console + on-screen terminal)
# --------------------------------------------------------------------------
LOG_BUFFER = []
MAX_LOG_LINES = 7


def log_message(msg, console=True):
    """Appends to the on-screen terminal buffer and (by default) prints to the console."""
    if console:
        print(msg)
    LOG_BUFFER.append(f"{time.strftime('[%H:%M:%S]')} {msg}")
    if len(LOG_BUFFER) > MAX_LOG_LINES:
        LOG_BUFFER.pop(0)


# --------------------------------------------------------------------------
# Timing statistics (rolling window; the first few samples are warm-up and skipped)
# --------------------------------------------------------------------------
class Stats:
    def __init__(self, window=100, skip=5):
        self.window, self.skip = window, skip
        self.d, self.n = {}, {}

    def add(self, key, ms):
        self.n[key] = self.n.get(key, 0) + 1
        if self.n[key] > self.skip:
            self.d.setdefault(key, deque(maxlen=self.window)).append(ms)

    def avg(self, key):
        v = self.d.get(key)
        return sum(v) / len(v) if v else 0.0

    def mx(self, key):
        v = self.d.get(key)
        return max(v) if v else 0.0


# --------------------------------------------------------------------------
# Grouping (display only: never influences which target is locked)
# --------------------------------------------------------------------------
def largest_group_size(dets, eps):
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
        self.manual = False                      # operator forced the lock
        self.dets, self.ranked, self.group = [], [], 0
        self.history, self.spawn = {}, {}
        self.kf = KalmanCV(cfg["kalman_accel_var"], cfg["kalman_meas_sigma"])
        self.locked, self.locked_id = False, None
        self.locked_conf, self.locked_age = 0.0, 0
        self.missing, self.challenge = 0, 0
        self.last_wh = (0.0, 0.0)
        self.status = "SEARCHING"
        self.box, self.center = None, None

    def filter_detections(self, boxes):
        dets, seen = [], set()
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
                w, h = x2 - x1, y2 - y1
                if w < cfg["min_box_width"] or h < cfg["min_box_height"]:
                    continue
                ar = w / (h + 1e-5)
                if ar < cfg["min_aspect_ratio"] or ar > cfg["max_aspect_ratio"]:
                    continue
                tid = int(ids[i]) if ids is not None else -(i + 1)
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                seen.add(tid)
                hist = self.history.setdefault(tid, [])
                self.spawn.setdefault(tid, (cx, cy))
                hist.append((cx, cy))
                if len(hist) > 30:
                    hist.pop(0)
                # Static-clutter gate (clouds, hot rocks). The current lock is exempt:
                # once a turret centres a target it barely moves in the image.
                if (cfg["static_gate_enabled"] and tid != self.locked_id
                        and len(hist) >= cfg["static_gate_frames"]):
                    sx, sy = self.spawn[tid]
                    if math.hypot(cx - sx, cy - sy) < cfg["static_gate_px"]:
                        continue
                dets.append(dict(tid=tid, cx=cx, cy=cy, w=w, h=h, conf=float(confs[i]),
                                 age=len(hist), box=(x1, y1, x2, y2)))
        for k in [k for k in self.history if k not in seen]:
            self.history.pop(k, None)
            self.spawn.pop(k, None)
        return dets

    def score(self, d):
        w = cfg["priority_weights"]
        size = min(math.sqrt(d["w"] * d["h"]) / (cfg["priority_size_ref"] * DIAG), 1.0)
        centre = 1 - math.hypot(d["cx"] - W / 2, d["cy"] - H / 2) / (DIAG / 2)
        age = min(d["age"] / 30.0, 1.0)
        return w["size"] * size + w["center"] * centre + w["conf"] * d["conf"] + w["age"] * age

    def _locked_status(self):
        return "LOCKED (MANUAL)" if self.manual else "LOCKED"

    def _lock(self, d):
        self.kf.reset(d["cx"], d["cy"])
        self.locked, self.locked_id = True, d["tid"]
        self.locked_conf = d["conf"]
        self.locked_age = d["age"]
        self.last_wh = (d["w"], d["h"])
        self.missing, self.challenge = 0, 0
        self.status = self._locked_status()

    def manual_lock(self, det):
        self._lock(det)
        self.manual = True
        self.status = self._locked_status()
        self._publish()

    def _release(self):
        self.locked, self.locked_id = False, None
        self.locked_conf, self.locked_age = 0.0, 0
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
                self.locked_conf = match["conf"]
                self.locked_age = match["age"]
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
# ESP32 serial link (aim + fire flag)
# Packet: 0xA5 | cmd(1) seq(2) pan_mrad(2) tilt_mrad(2) flags(1) | xor checksum
# flags: bit0 = track valid, bit1 = fire request. The ESP32 must still enforce its own
# arm switch, cooldown, travel limits and watchdog (zero motion if no packet for ~150 ms).
# --------------------------------------------------------------------------
class Esp32Link:
    def __init__(self, c):
        self.ser, self.seq = None, 0
        if c.get("enabled"):
            import serial
            self.ser = serial.Serial(c["port"], c["baud"], timeout=0)

    def send(self, pan_rad, tilt_rad, valid, fire=False):
        if self.ser is None:
            return
        flags = (1 if valid else 0) | (2 if fire else 0)
        body = struct.pack("<BHhhB", 0x01, self.seq & 0xFFFF,
                           int(pan_rad * 1000), int(tilt_rad * 1000), flags)
        crc = 0
        for b in body:
            crc ^= b
        self.ser.write(bytes([0xA5]) + body + bytes([crc]))
        self.seq += 1

    def close(self):
        if self.ser:
            self.ser.close()


def px_to_angles(cx, cy, sensor_type="rgb"):
    offsets = cfg.get("camera_offsets", {"rgb": {"x": 0, "y": 0}, "ir": {"x": 0, "y": 0}})
    adjusted_cx = cx + offsets.get(sensor_type, {}).get("x", 0)
    adjusted_cy = cy + offsets.get(sensor_type, {}).get("y", 0)
    fx = (W / 2) / math.tan(math.radians(cfg["camera_hfov_deg"]) / 2)
    pan = math.atan((adjusted_cx - W / 2) / fx)
    tilt = math.atan((adjusted_cy - H / 2) / fx)
    return pan, (-tilt if cfg["invert_tilt"] else tilt)


def aim_error_ok(tr, sensor_type):
    """True when the locked target is within auto_shoot_max_error_deg of the crosshair."""
    pan, tilt = px_to_angles(*tr.center, sensor_type=sensor_type)
    return math.degrees(math.hypot(pan, tilt)) <= cfg.get("auto_shoot_max_error_deg", 2.0)


# --------------------------------------------------------------------------
# Fire rules (single place: manual and auto)
# --------------------------------------------------------------------------
def decide_fire(state, src, sensor_name, now, last_fire, dry=""):
    """Returns (should_fire, last_fire_time)."""
    cooldown_ok = (now - last_fire) >= cfg.get("fire_cooldown", 2.0)
    armed_ok = state["armed"] or not cfg.get("require_arm", True)

    if state["manual_fire_trigger"]:
        state["manual_fire_trigger"] = False
        if not armed_ok:
            log_message("[SAFE] Fire blocked: system is DISARMED (press x or click ARM).")
            return False, last_fire
        if src and cooldown_ok:
            log_message("[ACTION] Manual Interception Fired!" + dry)
            return True, now
        log_message("[WARNING] Cannot fire: No active target lock or cooldown active.")
        return False, last_fire

    if (state["auto_shoot"] and armed_ok and cooldown_ok and src and src.locked
            and src.missing == 0):          # a real detection this frame, not a coasting prediction
        conf_ok = src.locked_conf >= cfg.get("auto_shoot_confidence", 0.65)
        age_ok = src.locked_age >= cfg.get("auto_shoot_min_age", 15)
        if conf_ok and age_ok and aim_error_ok(src, sensor_name):
            log_message(f"[AUTO-SHOOT] Target locked (Conf: {src.locked_conf:.2f}, "
                        f"Age: {src.locked_age}f). Interception Fired!{dry}")
            return True, now
    return False, last_fire


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


def load_model(path, label):
    """Loads an ONNX export (or .pt) and returns (model, imgsz). ONNX inputs are fixed-size,
    so the size is read from the model itself and overrides inference_imgsz."""
    if not os.path.exists(path):
        sys.exit(f"[MODEL] {label} model not found: {path}\n"
                 f"        Export it first:  python export_models.py")
    imgsz = cfg["inference_imgsz"]
    if path.lower().endswith(".onnx"):
        try:
            import onnxruntime as ort
            shape = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs()[0].shape
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int):
                imgsz = h if h == w else [h, w]
                if imgsz != cfg["inference_imgsz"]:
                    log_message(f"[MODEL] {label}: ONNX exported at {imgsz}px, using that "
                                f"(config inference_imgsz = {cfg['inference_imgsz']}).")
        except Exception as e:
            log_message(f"[MODEL] {label}: could not read ONNX input size ({e}); using {imgsz}.")
    log_message(f"[MODEL] {label}: {os.path.basename(path)} @ {imgsz}px on CPU")
    return YOLO(path, task="detect"), imgsz


def infer(model, imgsz, frame, conf):
    return model.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml",
                       imgsz=imgsz, verbose=False, device=DEVICE)[0].boxes


# --------------------------------------------------------------------------
# Rendering and UI
# --------------------------------------------------------------------------
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
            line = (f"{mark} T{i}  id {d['tid']:<4} score {sc:.2f}  conf {d['conf']:.2f}  "
                    f"({d['cx']:.0f},{d['cy']:.0f})")
            col = (0, 0, 255) if mark == ">" else (210, 210, 210)
            cv2.putText(img, line, (x0, 28 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.line(img, (W, 0), (W, ph), (50, 50, 50), 2)
    return img


def draw_button(banner, text, x1, y1, x2, y2, active=False, danger=False):
    if danger:
        bg_color = (0, 0, 180) if not active else (0, 0, 255)
        border_color = (0, 0, 255)
        text_color = (255, 255, 255)
    else:
        bg_color = (60, 60, 60) if not active else (0, 100, 150)
        border_color = (0, 200, 255) if not active else (0, 255, 255)
        text_color = (255, 255, 255) if not active else (0, 255, 255)
    cv2.rectangle(banner, (x1, y1), (x2, y2), bg_color, -1)
    cv2.rectangle(banner, (x1, y1), (x2, y2), border_color, 1)
    cv2.putText(banner, text, (x1 + 6, y1 + 21), cv2.FONT_HERSHEY_SIMPLEX, 0.38, text_color, 1)


def compose(disp_ir, disp_rgb, now, ir_tr, rgb_tr, state):
    feeds = np.hstack((disp_ir, disp_rgb))
    cv2.line(feeds, (W, 0), (W, H), (50, 50, 50), 3)
    banner = np.full((BANNER_H, W * 2, 3), 25, np.uint8)

    cv2.putText(banner, "C-UAV COMMAND & CONTROL SYSTEM", (20, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    n = max(ir_tr.group, rgb_tr.group)
    if n >= cfg["swarm_alert_size"]:
        cv2.putText(banner, f"SWARM ALERT x{n}", (400, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # Buttons (banner is 2*W = 1280 px wide with the default 640 frame width)
    draw_button(banner, f"AIM: {state['aim_sensor'].upper()}", 600, 8, 715, 36, active=True)
    draw_button(banner, f"AUTO: {'ON' if state['auto_shoot'] else 'OFF'}", 725, 8, 850, 36,
                active=state["auto_shoot"])
    draw_button(banner, "RELEASE", 860, 8, 960, 36)
    draw_button(banner, f"LOGS: {'ON' if state['show_logs'] else 'OFF'}", 970, 8, 1060, 36,
                active=state["show_logs"])
    draw_button(banner, f"ARM: {'ON' if state['armed'] else 'OFF'}", 1070, 8, 1160, 36,
                active=state["armed"], danger=state["armed"])
    draw_button(banner, "FIRE!", 1170, 8, 1260, 36, danger=True)

    parts = [banner, feeds, panel(ir_tr, rgb_tr)]
    if state["show_logs"]:
        term_h = 130
        term = np.full((term_h, W * 2, 3), 12, np.uint8)
        cv2.rectangle(term, (0, 0), (W * 2, term_h), (30, 30, 30), 1)
        cv2.putText(term, ">> LIVE SYSTEM TERMINAL LOGS", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
        cv2.line(term, (10, 28), (W * 2 - 10, 28), (50, 50, 50), 1)
        for idx, line in enumerate(LOG_BUFFER[-5:]):
            cv2.putText(term, line, (15, 48 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)
        parts.append(term)
    return np.vstack(parts)


def handle_click(x, y, ir_tr, rgb_tr, state):
    """Mouse clicks on banner buttons or target bounding boxes."""
    if 0 <= y < BANNER_H:
        if 600 <= x <= 715:
            state["aim_sensor"] = "ir" if state["aim_sensor"] == "rgb" else "rgb"
            log_message(f"[UI] Aim Sensor switched to: {state['aim_sensor'].upper()}")
        elif 725 <= x <= 850:
            state["auto_shoot"] = not state["auto_shoot"]
            log_message(f"[UI] Auto-Shoot toggled to: {'ON' if state['auto_shoot'] else 'OFF'}")
        elif 860 <= x <= 960:
            ir_tr._release()
            rgb_tr._release()
            log_message("[UI] All target locks released.")
        elif 970 <= x <= 1060:
            state["show_logs"] = not state["show_logs"]
            log_message(f"[UI] Terminal logs toggled to: {'ON' if state['show_logs'] else 'OFF'}")
        elif 1070 <= x <= 1160:
            state["armed"] = not state["armed"]
            log_message(f"[UI] System {'ARMED' if state['armed'] else 'DISARMED'}.")
        elif 1170 <= x <= 1260:
            state["manual_fire_trigger"] = True
            log_message("[UI] MANUAL FIRE REQUESTED VIA BUTTON.")
        else:
            return False
        return True

    y -= BANNER_H
    if not (0 <= y < H) or not (0 <= x < 2 * W):
        return False
    tr, ax = (ir_tr, x) if x < W else (rgb_tr, x - W)
    hits = [d for _, d in tr.ranked
            if d["box"][0] - 6 <= ax <= d["box"][2] + 6 and d["box"][1] - 6 <= y <= d["box"][3] + 6]
    if hits:
        tr.manual_lock(min(hits, key=lambda d: d["w"] * d["h"]))
        log_message(f"[UI] Force-locked target ID {tr.locked_id} via mouse click.")
        return True
    return False


# --------------------------------------------------------------------------
def main():
    log_message("Starting C-UAV v2 (ONNX / CPU)...")
    state = {
        "aim_sensor": cfg.get("aim_sensor", "rgb").lower(),
        "show_logs": True,
        "auto_shoot": cfg.get("auto_shoot_enabled", False),
        "manual_fire_trigger": False,
        "armed": False,                 # always starts disarmed
    }

    ir_model, ir_imgsz = load_model(cfg["thermal_model_path"], "IR")
    rgb_model, rgb_imgsz = load_model(cfg["rgb_model_path"], "RGB")
    cap_ir, cap_rgb = open_source(cfg["ir_source"]), open_source(cfg["rgb_source"])
    iv_ir = 1.0 / (cap_ir.get(cv2.CAP_PROP_FPS) or 30)
    iv_rgb = 1.0 / (cap_rgb.get(cv2.CAP_PROP_FPS) or 30)
    ir_tr = StreamTracker(cfg["group_eps"]["ir"])
    rgb_tr = StreamTracker(cfg["group_eps"]["rgb"])
    link = Esp32Link(cfg["esp32"])
    dry = "" if link.ser else " [DRY RUN: ESP32 disabled]"
    show = cfg["show_window"]
    win = "C-UAV Command & Control Interface"
    scale = cfg.get("window_scale", 1.0)   # <1.0 shrinks the whole window to fit small screens

    if show:
        cv2.namedWindow(win)
        cv2.setMouseCallback(
            win, lambda ev, x, y, fl, p: ev == cv2.EVENT_LBUTTONDOWN
            and handle_click(int(x / scale), int(y / scale), ir_tr, rgb_tr, state))

    disp_ir = np.zeros((H, W, 3), np.uint8)
    disp_rgb = disp_ir.copy()
    next_ir = next_rgb = time.time()
    stats = Stats()
    frames = {"IR": 0, "RGB": 0}          # frames processed since the last report
    t_cap = {"IR": time.time(), "RGB": time.time()}   # when each stream's latest frame was read
    t_log = time.time()
    csv_path = cfg.get("timing_csv", "")
    if csv_path and not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("time,ir_fps,ir_infer_avg_ms,ir_infer_max_ms,rgb_fps,rgb_infer_avg_ms,"
                    "rgb_infer_max_ms,cmd_avg_ms,cmd_max_ms\n")
    last_fire_time = 0.0

    try:
        while True:
            now, updated = time.time(), False

            if now >= next_ir:
                fr = read_frame(cap_ir)
                if fr is None:
                    break
                next_ir = now + iv_ir
                t_cap["IR"] = time.time()
                boxes = infer(ir_model, ir_imgsz, fr, cfg["thermal_confidence"])
                stats.add("IR infer", (time.time() - t_cap["IR"]) * 1000)
                t1 = time.time()
                ir_tr.step(ir_tr.filter_detections(boxes))
                stats.add("IR track", (time.time() - t1) * 1000)
                frames["IR"] += 1
                disp_ir = render(fr, ir_tr, "SENSOR 01: THERMAL IR")
                updated = True

            if now >= next_rgb:
                fr = read_frame(cap_rgb)
                if fr is None:
                    break
                next_rgb = now + iv_rgb
                t_cap["RGB"] = time.time()
                boxes = infer(rgb_model, rgb_imgsz, fr, cfg["rgb_confidence"])
                stats.add("RGB infer", (time.time() - t_cap["RGB"]) * 1000)
                t1 = time.time()
                rgb_tr.step(rgb_tr.filter_detections(boxes))
                stats.add("RGB track", (time.time() - t1) * 1000)
                frames["RGB"] += 1
                disp_rgb = render(fr, rgb_tr, "SENSOR 02: OPTICAL RGB")
                updated = True

            if updated:
                active_tr = rgb_tr if state["aim_sensor"] == "rgb" else ir_tr
                other_tr = ir_tr if state["aim_sensor"] == "rgb" else rgb_tr
                src = None
                if active_tr.locked and active_tr.manual:
                    src = active_tr
                elif other_tr.locked and other_tr.manual:
                    src = other_tr
                elif active_tr.locked:
                    src = active_tr
                elif other_tr.locked:
                    src = other_tr

                sensor_name = "rgb" if src is rgb_tr else "ir"
                should_fire, last_fire_time = decide_fire(
                    state, src, sensor_name, now, last_fire_time, dry)

                if src:
                    pan, tilt = px_to_angles(*src.center, sensor_type=sensor_name)
                    link.send(pan, tilt, True, fire=should_fire)
                else:
                    link.send(0.0, 0.0, False, fire=False)
                if src:   # capture -> command sent, for the sensor that is aiming
                    stats.add("cmd", (time.time() - t_cap[sensor_name.upper()]) * 1000)

                if show:
                    img = compose(disp_ir, disp_rgb, now, ir_tr, rgb_tr, state)
                    if scale != 1.0:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow(win, img)

            if show:
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break
                if key == ord("s"):
                    state["aim_sensor"] = "ir" if state["aim_sensor"] == "rgb" else "rgb"
                    log_message(f"[MODE] Aim Sensor switched via key to: {state['aim_sensor'].upper()}")
                if key == ord("l"):
                    state["show_logs"] = not state["show_logs"]
                    log_message(f"[MODE] Terminal logs toggled via key to: {'ON' if state['show_logs'] else 'OFF'}")
                if key == ord("f"):
                    state["manual_fire_trigger"] = True
                    log_message("[KEY] Manual fire requested via hotkey 'f'.")
                if key == ord(" "):
                    state["auto_shoot"] = not state["auto_shoot"]
                    log_message(f"[MODE] Auto-shoot toggled via spacebar to: {'ON' if state['auto_shoot'] else 'OFF'}")
                if key == ord("x"):
                    state["armed"] = not state["armed"]
                    log_message(f"[KEY] System {'ARMED' if state['armed'] else 'DISARMED'}.")
                if key == ord("a"):
                    ir_tr.manual = rgb_tr.manual = False
                    log_message("[MODE] Automatic target selection restored.")
                if key == ord("r"):
                    ir_tr._release()
                    rgb_tr._release()
                    log_message("[KEY] All target locks released.")
            time.sleep(0.001)

            if now - t_log > 2.0:   # timing + status go to the console only
                dt = now - t_log
                print(f"[TIMING] IR  {frames['IR'] / dt:4.1f} fps | infer {stats.avg('IR infer'):3.0f} avg / "
                      f"{stats.mx('IR infer'):3.0f} max ms | tracking {stats.avg('IR track'):.1f} ms")
                print(f"[TIMING] RGB {frames['RGB'] / dt:4.1f} fps | infer {stats.avg('RGB infer'):3.0f} avg / "
                      f"{stats.mx('RGB infer'):3.0f} max ms | tracking {stats.avg('RGB track'):.1f} ms")
                print(f"[TIMING] capture->command ({state['aim_sensor'].upper()} aim): "
                      f"{stats.avg('cmd'):3.0f} avg / {stats.mx('cmd'):3.0f} max ms   (goal < 50)")
                log_message(f"[TIMING] IR {frames['IR'] / dt:.1f} fps {stats.avg('IR infer'):.0f} ms | "
                            f"RGB {frames['RGB'] / dt:.1f} fps {stats.avg('RGB infer'):.0f} ms | "
                            f"cmd {stats.avg('cmd'):.0f} ms", console=False)   # live line in the UI terminal
                print(f"[STATUS] IR {ir_tr.status} | RGB {rgb_tr.status} | AIM: {state['aim_sensor'].upper()} "
                      f"| AUTO: {state['auto_shoot']} | ARMED: {state['armed']}")
                if csv_path:
                    with open(csv_path, "a") as f:
                        f.write(f"{time.strftime('%H:%M:%S')},{frames['IR'] / dt:.1f},"
                                f"{stats.avg('IR infer'):.1f},{stats.mx('IR infer'):.1f},"
                                f"{frames['RGB'] / dt:.1f},{stats.avg('RGB infer'):.1f},"
                                f"{stats.mx('RGB infer'):.1f},{stats.avg('cmd'):.1f},{stats.mx('cmd'):.1f}\n")
                frames["IR"] = frames["RGB"] = 0
                t_log = now
    except KeyboardInterrupt:
        pass
    finally:
        link.send(0.0, 0.0, False, fire=False)
        link.close()
        cap_ir.release()
        cap_rgb.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()