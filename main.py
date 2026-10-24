"""C-UAV v2: dual-sensor single-target tracker, ONNX / CPU, threaded, per-sensor configuration.

Every sensor ('ir' and 'rgb' are just the two slots) has its own block in config.json: source and
camera format, processing size, field of view, model, input size and confidence. The hardware behind a
slot is free (thermal, NoIR, normal camera, video file); see README.md for how to add a new source type.

Interactive UI (click or hotkey):
  [ AIM: RGB/IR ]  (s)      switch the sensor used for aiming
  [ AUTO: ON/OFF ] (space)  automatic interception (needs ARM, a stable lock and the target on the crosshair)
  [ RELEASE ]      (r)      clear all target locks
  [ LOGS: ON/OFF ] (l)      show / hide the on-screen terminal
  [ ARM: ON/OFF ]  (x)      safety arm. Starts OFF every run. Fire commands are blocked while disarmed
  [ FIRE! ]        (f)      manual interception command (needs ARM)
  [ RELOAD ]       (e)      send a reload command to the ESP32 now
  [ A-RELOAD: ON/OFF ] (z)  automatically send a reload command after every fire (like AUTO)
  Click a bounding box to force-lock it.  (a) returns to automatic target selection.  (q) quits.

Threading: one worker thread per sensor. The aiming sensor runs on every frame; the other sensor runs
every secondary_every_n frames (or every frame while the aiming sensor has no lock / is frozen).
Live cameras are read by a background thread that keeps only the newest frame (no stale backlog).
Commands to the ESP32 go out right after each new aiming result and at send_hz as a keep-alive.
A sensor whose results are older than stale_result_ms is ignored (no stale aim, no auto-fire).

Usage:  python main.py [config.json]      Ctrl+C also quits.
"""
import json
import math
import os
import struct
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
from ultralytics import YOLO

with open(sys.argv[1] if len(sys.argv) > 1 else "config.json") as f:
    cfg = json.load(f)

SENS = cfg["sensors"]
KEYS = ("ir", "rgb")                                  # display order, left to right
DW, DH = cfg["display"]["width"], cfg["display"]["height"]   # size of one feed cell in the window
BANNER_H = 45
DEVICE = cfg.get("device", "cpu")                     # ONNX models run on CPU via ONNX Runtime


class SourceError(Exception):
    """A configured sensor could not be started (camera missing, model missing...)."""


def sensor_enabled(key):
    """A sensor is OFF when its source is empty / null, or when sensors.<key>.enabled is false."""
    sc = SENS[key]
    src = sc.get("source")
    return bool(sc.get("enabled", True)) and src is not None and str(src).strip() != ""


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


LIVE_LINE = [""]        # one pinned status line at the bottom of the terminal (replaced, never appended)


def set_live_line(text):
    LIVE_LINE[0] = f"{time.strftime('[%H:%M:%S]')} {text}"


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
# Per-stream tracker: filters, target priority, lock, coast.
# All pixel values are in this sensor's PROCESSING frame (sensors.<key>.width x height).
# --------------------------------------------------------------------------
class StreamTracker:
    def __init__(self, key):
        self.key = key
        self.w, self.h = SENS[key]["width"], SENS[key]["height"]
        self.diag = math.hypot(self.w, self.h)
        self.eps = self.p("group_eps")
        self.manual = False                      # operator forced the lock
        self.dets, self.ranked, self.group = [], [], 0
        self.history, self.spawn = {}, {}
        self.kf = KalmanCV(self.p("kalman_accel_var"), self.p("kalman_meas_sigma"))
        self.locked, self.locked_id = False, None
        self.locked_conf, self.locked_age = 0.0, 0
        self.missing, self.challenge = 0, 0
        self.last_wh = (0.0, 0.0)
        self.status = "SEARCHING"
        self.box, self.center = None, None

    def p(self, name):
        """Tracker parameter: the sensor's own value if set, else the global default."""
        sc = SENS[self.key]
        return sc[name] if name in sc else cfg[name]

    def filter_detections(self, boxes):
        dets, seen = [], set()
        if boxes is not None and len(boxes) > 0:
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            ids = boxes.id.cpu().numpy() if boxes.id is not None else None
            for i in range(len(xyxy)):
                x1, y1, x2, y2 = (float(v) for v in xyxy[i][:4])
                w, h = x2 - x1, y2 - y1
                if w < self.p("min_box_width") or h < self.p("min_box_height"):
                    continue
                ar = w / (h + 1e-5)
                if ar < self.p("min_aspect_ratio") or ar > self.p("max_aspect_ratio"):
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
                if (self.p("static_gate_enabled") and tid != self.locked_id
                        and len(hist) >= self.p("static_gate_frames")):
                    sx, sy = self.spawn[tid]
                    if math.hypot(cx - sx, cy - sy) < self.p("static_gate_px"):
                        continue
                dets.append(dict(tid=tid, cx=cx, cy=cy, w=w, h=h, conf=float(confs[i]),
                                 age=len(hist), box=(x1, y1, x2, y2)))
        for k in [k for k in self.history if k not in seen]:
            self.history.pop(k, None)
            self.spawn.pop(k, None)
        return dets

    def score(self, d):
        w = cfg["priority_weights"]
        size = min(math.sqrt(d["w"] * d["h"]) / (cfg["priority_size_ref"] * self.diag), 1.0)
        centre = 1 - math.hypot(d["cx"] - self.w / 2, d["cy"] - self.h / 2) / (self.diag / 2)
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
                if math.hypot(near["cx"] - px, near["cy"] - py) <= self.p("gate_px"):
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
                if self.missing > self.p("max_coast_frames"):
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
# ESP32 serial link.
# PC -> ESP32 command: 0xA5 | cmd(1)=0x01 seq(2) pan_mrad(2) tilt_mrad(2) flags(1) | xor checksum of the 8 body bytes
#   flags: bit0 = track valid, bit1 = fire request, bit2 = reload request.
# ESP32 -> PC status:  0x5A | type(1)=0x02 | flags(1) | xor(type, flags)      (send at ~10 Hz and on change)
#   flags: bit0 = dart present in the barrel, bit1 = dart sensor OK, bit2 = reload running (busy).
# The ESP32 must still enforce its own arm switch, cooldown, travel limits and watchdog
# (zero motion if no packet for ~150 ms). Do not rely on the PC for safety.
# --------------------------------------------------------------------------
STATUS_PAYLOAD_LEN = {0x02: 1}        # status packet types the PC understands -> payload length


class Esp32Link:
    def __init__(self, c):
        self.ser, self.seq = None, 0
        self.buf = bytearray()
        self.status, self.status_t = None, 0.0        # newest status from the ESP32 and when it arrived
        if c.get("enabled"):
            try:
                import serial
                # serial_for_url accepts "COM3", "/dev/ttyUSB0" and URLs such as socket://127.0.0.1:5555
                self.ser = serial.serial_for_url(c["port"], baudrate=c["baud"], timeout=0)
            except Exception as e:
                log_message(f"[ESP32] Could not open {c['port']}: {e}. Running without the ESP32 (dry run).")

    def send(self, pan_rad, tilt_rad, valid, fire=False, reload=False):
        if self.ser is None:
            return
        flags = (1 if valid else 0) | (2 if fire else 0) | (4 if reload else 0)
        body = struct.pack("<BHhhB", 0x01, self.seq & 0xFFFF,
                           int(pan_rad * 1000), int(tilt_rad * 1000), flags)
        crc = 0
        for b in body:
            crc ^= b
        try:
            self.ser.write(bytes([0xA5]) + body + bytes([crc]))
        except Exception as e:
            log_message(f"[ESP32] Write failed: {e}")
            self.ser = None
            return
        self.seq += 1

    def poll(self):
        """Read whatever the ESP32 has sent and keep the newest status. Never blocks."""
        if self.ser is None:
            return
        try:
            n = self.ser.in_waiting
            if n:
                self.buf += self.ser.read(n)
        except Exception as e:
            log_message(f"[ESP32] Read failed: {e}")
            self.ser = None
            return
        b = self.buf
        while b:
            if b[0] != 0x5A:                              # resync on the start byte
                del b[0]
                continue
            if len(b) < 2:
                return
            plen = STATUS_PAYLOAD_LEN.get(b[1])
            if plen is None:
                del b[0]
                continue
            if len(b) < 3 + plen:
                return
            x = b[1]
            for v in b[2:2 + plen]:
                x ^= v
            if x != b[2 + plen]:                          # bad checksum: drop the start byte, resync
                del b[0]
                continue
            fl = b[2]
            del b[:3 + plen]
            self.status = {"dart": bool(fl & 1), "sensor_ok": bool(fl & 2), "busy": bool(fl & 4)}
            self.status_t = time.time()

    def close(self):
        if self.ser:
            self.ser.close()


def px_to_angles(cx, cy, key, box_w=None):
    """Target pixel -> pan/tilt error in radians.
    Accepts an optional box_w (bounding box width in pixels) to dynamically
    estimate target distance and compute precise physical offsets."""
    sc = SENS[key]
    w, h = sc["width"], sc["height"]
    off = sc.get("offset", {})
    half = math.radians(sc["hfov_deg"]) / 2
    span = h if sc.get("fov_across", "width") == "height" else w
    
    # Calculate camera focal length in pixels based on field of view
    f_pixels = (span / 2) / math.tan(half)
    
    # 1. Check if we are using dynamic physical offsets
    if ("x_inches" in off or "y_inches" in off) and "known_target_width_inches" in off:
        x_inch = off.get("x_inches", 0.0)
        y_inch = off.get("y_inches", 0.0)
        target_real_w = off.get("known_target_width_inches", 12.0)
        
        # 2. Estimate distance dynamically using the bounding box pixel width
        # If no valid bounding box width is available yet, fall back to a reasonable baseline (e.g., 15 meters / 590 inches)
        if box_w and box_w > 0:
            estimated_dist_inches = (target_real_w * f_pixels) / box_w
        else:
            estimated_dist_inches = 590.0 
            
        # 3. Calculate full visibility window size in inches at this specific distance
        visible_width_inches = 2.0 * estimated_dist_inches * math.tan(half)
        visible_height_inches = visible_width_inches * (h / w)
        
        # 4. Convert your physical camera offset inches to pixels at this distance
        pixel_offset_x = (x_inch / visible_width_inches) * w
        pixel_offset_y = (y_inch / visible_height_inches) * h
        
        dx = cx + pixel_offset_x - w / 2
        dy = cy + pixel_offset_y - h / 2
    else:
        # Fallback to standard manual pixel offsets if inch configurations aren't provided
        dx = cx + off.get("x", 0) - w / 2
        dy = cy + off.get("y", 0) - h / 2

    if sc.get("lens", "rectilinear") == "fisheye":
        r = math.hypot(dx, dy)
        if r == 0:
            vx, vy, vz = 0.0, 0.0, 1.0
        else:
            theta = r / f_pixels
            vx, vy, vz = math.sin(theta) * dx / r, math.sin(theta) * dy / r, math.cos(theta)
    else:
        vx, vy, vz = dx, dy, f_pixels
        
    pan = math.atan2(vx, vz)
    tilt = math.atan2(vy, math.hypot(vx, vz))
    return pan, (-tilt if cfg["invert_tilt"] else tilt)


def aim_error_deg(tr):
    # Pass the last tracked bounding box width to calculate dynamic distance
    box_w = tr.last_wh[0] if hasattr(tr, 'last_wh') and tr.last_wh else None
    pan, tilt = px_to_angles(tr.center[0], tr.center[1], tr.key, box_w=box_w)
    return math.degrees(math.hypot(pan, tilt))


def aim_error_ok(tr):
    """True when the locked target is within auto_shoot_max_error_deg of the crosshair."""
    return aim_error_deg(tr) <= cfg.get("auto_shoot_max_error_deg", 2.0)


# --------------------------------------------------------------------------
# Fire rules (single place: manual and auto)
# --------------------------------------------------------------------------
AUTO_QUIET = {"no target", "cooldown", "reloading"}      # expected states: not announced in the log


def auto_block_reason(state, src, now, last_fire):
    """None when auto-fire may shoot now, otherwise (category, message)."""
    if src is None or not src.locked:
        return "no target", "no target locked"
    if cfg.get("require_arm", True) and not state["armed"]:
        return "disarmed", "DISARMED (press x or click ARM)"
    am = ammo_state(state, now)
    if am == "reloading":
        return "reloading", f"reloading ({state['reload_until'] - now:.0f}s left)"
    if am == "unknown":
        return "unknown", "dart state unknown (no sensor data from the ESP32)"
    if am == "empty":
        if state.get("auto_reload") and state.get("reload_due") is not None:
            return "reloading", "autom
