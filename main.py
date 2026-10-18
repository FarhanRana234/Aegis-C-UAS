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


def px_to_angles(cx, cy, key):
    """Target pixel (in sensor `key`'s processing frame) -> pan/tilt error in radians.
    lens "rectilinear" (default): normal pinhole lens.  lens "fisheye": equidistant model (radius = f * angle).
    hfov_deg is the field of view of the processed image across its width (or across its height when
    fov_across is "height", e.g. a circular fisheye image whose circle spans the frame height)."""
    sc = SENS[key]
    w, h = sc["width"], sc["height"]
    off = sc.get("offset", {})
    dx, dy = cx + off.get("x", 0) - w / 2, cy + off.get("y", 0) - h / 2
    half = math.radians(sc["hfov_deg"]) / 2
    span = h if sc.get("fov_across", "width") == "height" else w   # axis that hfov_deg spans
    if sc.get("lens", "rectilinear") == "fisheye":
        f = (span / 2) / half
        r = math.hypot(dx, dy)
        if r == 0:
            vx, vy, vz = 0.0, 0.0, 1.0
        else:
            theta = r / f
            vx, vy, vz = math.sin(theta) * dx / r, math.sin(theta) * dy / r, math.cos(theta)
    else:
        vx, vy, vz = dx, dy, (span / 2) / math.tan(half)
    pan = math.atan2(vx, vz)                       # rotate about the vertical axis first,
    tilt = math.atan2(vy, math.hypot(vx, vz))      # then tilt: the order an az-el gimbal uses
    return pan, (-tilt if cfg["invert_tilt"] else tilt)


def aim_error_deg(tr):
    pan, tilt = px_to_angles(*tr.center, tr.key)
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
            return "reloading", "automatic reload queued"          # expected right after a shot
        return "empty", "no dart loaded (press RELOAD or turn on A-RELOAD)"
    if src.missing > 0:
        return "coasting", "target is coasting (no fresh detection)"
    thr = cfg.get("auto_shoot_confidence", 0.65)
    if src.locked_conf < thr:
        return "confidence", f"confidence {src.locked_conf:.2f} < {thr:.2f}"
    need = cfg.get("auto_shoot_min_age", 15)
    if src.locked_age < need:
        return "age", f"lock age {src.locked_age} < {need} frames"
    err, lim = aim_error_deg(src), cfg.get("auto_shoot_max_error_deg", 2.0)
    if err > lim:
        return "off crosshair", f"target {err:.1f} deg off the crosshair (limit {lim:.1f})"
    if now - last_fire < cfg.get("fire_cooldown", 2.0):
        return "cooldown", "cooldown"
    return None


def _ammo_from_timer(state, now):
    """No sensor data: assume the reload finished when reload_timeout_s has passed."""
    state["ammo_src"] = "timer"
    if state["loaded"]:
        return "loaded"
    if state["reloading"]:
        if now >= state["reload_until"]:
            state["loaded"], state["reloading"] = True, False
            log_message("[RELOAD] Reload timeout finished: dart assumed loaded, ready to fire.")
            return "loaded"
        return "reloading"
    return "empty"


def _ammo_from_sensor(state, st, now):
    """The ESP32 reports the dart sensor, so the real state is known."""
    state["ammo_src"] = "sensor"
    present, busy = st["dart"], st["busy"]
    if state["reloading"]:
        if busy or now - state["reload_sent_at"] < cfg.get("reload_grace_s", 0.5):
            return "reloading"                        # ESP32 is running the reload
        if present:
            state["loaded"], state["reloading"] = True, False
            log_message("[RELOAD] Sensor confirms: dart loaded, ready to fire.")
            return "loaded"
        if now >= state["reload_until"]:
            state["loaded"], state["reloading"] = False, False
            log_message("[RELOAD] FAILED: timeout reached and the sensor still shows no dart (jam or out of darts?).")
            return "empty"
        return "reloading"
    if now - state.get("fired_at", -1e9) < cfg.get("sensor_settle_s", 0.6):
        return "empty"                                # the dart is still leaving the barrel
    state["loaded"] = present and not busy
    return "loaded" if state["loaded"] else ("reloading" if busy else "empty")


def ammo_state(state, now):
    """One dart fits in the barrel. Returns 'loaded', 'reloading', 'empty' or 'unknown'.
    ammo_source: 'auto' (ESP32 sensor when its status is arriving, else a timer), 'sensor' (sensor only; no data
    means 'unknown' and fire is blocked) or 'timer' (never use the sensor)."""
    mode = cfg.get("ammo_source", "auto")
    st = state.get("esp_status")
    if mode != "timer" and st is not None and st["sensor_ok"]:
        return _ammo_from_sensor(state, st, now)
    if mode == "sensor":
        state["ammo_src"] = "sensor"
        return "unknown"
    return _ammo_from_timer(state, now)


def note_fired(state, now):
    """After a fire command: the dart is gone, count the shot, queue the automatic reload if it is on."""
    state["loaded"], state["reloading"] = False, False
    state["fired_at"] = now
    state["shots"] = state.get("shots", 0) + 1
    if state.get("auto_reload"):
        state["reload_due"] = now + cfg.get("auto_reload_delay_s", 0.5)


def _start_reload(state, now):
    state["loaded"], state["reloading"] = False, True
    state["reload_sent_at"] = now
    state["reload_until"] = now + cfg.get("reload_timeout_s", 6.0)
    state["shots"] = 0
    state["reload_due"] = None


def decide_reload(state, now, last_reload, tag=""):
    """Manual reload request, or the delayed automatic reload queued after a shot.
    Returns (reload_now, last_reload_time). With auto reload OFF nothing is ever sent on its own."""
    if state.get("manual_reload_trigger"):
        state["manual_reload_trigger"] = False
        if now - last_reload < cfg.get("reload_min_interval_s", 1.0):
            log_message("[RELOAD] Ignored: a reload command was just sent.")
            return False, last_reload
        _start_reload(state, now)
        log_message(f"[RELOAD] Manual reload command sent (timeout {cfg.get('reload_timeout_s', 6.0):g}s)." + tag)
        return True, now
    due = state.get("reload_due")
    if due is not None and now >= due:
        state["reload_due"] = None
        if state.get("auto_reload"):                       # switched off meanwhile -> nothing is sent
            _start_reload(state, now)
            log_message(f"[RELOAD] Auto reload command sent after fire (timeout {cfg.get('reload_timeout_s', 6.0):g}s)." + tag)
            return True, now
    return False, last_reload


def decide_fire(state, src, now, last_fire, tag=""):
    """src: the driving StreamTracker or None. tag: ' [DRY RUN: ...]' or ' [sent to ESP32]'.
    Returns (should_fire, last_fire_time)."""
    cooldown_ok = (now - last_fire) >= cfg.get("fire_cooldown", 2.0)
    armed_ok = state["armed"] or not cfg.get("require_arm", True)

    if state["manual_fire_trigger"]:
        state["manual_fire_trigger"] = False
        if not armed_ok:
            log_message("[SAFE] Fire blocked: system is DISARMED (press x or click ARM).")
            return False, last_fire
        am = ammo_state(state, now)
        if am == "reloading":
            log_message(f"[SAFE] Fire blocked: reloading ({state['reload_until'] - now:.0f}s left).")
            return False, last_fire
        if am == "unknown":
            log_message("[SAFE] Fire blocked: dart state unknown (no sensor data from the ESP32).")
            return False, last_fire
        if am == "empty":
            log_message("[SAFE] Fire blocked: no dart loaded (press RELOAD or turn on A-RELOAD).")
            return False, last_fire
        if src and cooldown_ok:
            log_message("[ACTION] Manual Interception Fired!" + tag)
            note_fired(state, now)
            return True, now
        log_message("[WARNING] Cannot fire: No active target lock or cooldown active.")
        return False, last_fire

    if state["auto_shoot"]:
        why = auto_block_reason(state, src, now, last_fire)
        if why is None:
            log_message(f"[AUTO-SHOOT] Target locked (Conf: {src.locked_conf:.2f}, Age: {src.locked_age}f, "
                        f"{aim_error_deg(src):.1f} deg off). Interception Fired!{tag}")
            state["auto_cat"], state["auto_msg"] = "fired", "fired"
            note_fired(state, now)
            return True, now
        cat, msg = why
        if cat != state.get("auto_cat"):
            state["auto_cat"] = cat
            if cat not in AUTO_QUIET and now - state.get("auto_log_t", 0.0) > 1.5:
                log_message(f"[AUTO] Holding fire: {msg}")      # says WHY auto-fire is not shooting
                state["auto_log_t"] = now
        state["auto_msg"] = msg
    else:
        state["auto_cat"], state["auto_msg"] = None, ""
    return False, last_fire


# --------------------------------------------------------------------------
# Frame sources. A source is any object with  read() -> (frame, capture_time) | None  and  release().
# Add your own (a thermal array over SPI, a vendor SDK...) and register it in SOURCE_TYPES.
# --------------------------------------------------------------------------
def to_proc(fr, w, h, crop=None):
    """Crop (optional, fractions of the raw frame) then resize to the sensor's processing size.
    Cropping the centre of a wide-angle camera keeps full pixel detail on the target (digital zoom)."""
    if crop:
        H0, W0 = fr.shape[:2]
        x0, y0 = max(0, int(round(crop["x"] * W0))), max(0, int(round(crop["y"] * H0)))
        x1 = min(W0, x0 + max(1, int(round(crop["w"] * W0))))
        y1 = min(H0, y0 + max(1, int(round(crop["h"] * H0))))
        fr = fr[y0:y1, x0:x1]
    if fr.shape[1] == w and fr.shape[0] == h:
        return fr
    shrinking = fr.shape[1] * fr.shape[0] > w * h
    return cv2.resize(fr, (w, h), interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR)


class LiveCapture(threading.Thread):
    """Reads a camera as fast as it delivers and keeps only the newest frame (no stale backlog)."""

    def __init__(self, cap):
        super().__init__(name="camera-reader", daemon=True)
        self.cap = cap
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.alive, self.stopping = True, False

    def run(self):
        fails = 0
        while not self.stopping:
            ok, fr = self.cap.read()
            if not ok:
                fails += 1
                if fails > 30:
                    break
                time.sleep(0.01)
                continue
            fails = 0
            with self.cond:
                self.frame, self.t, self.seq = fr, time.time(), self.seq + 1
                self.cond.notify_all()
        with self.cond:
            self.alive = False
            self.cond.notify_all()

    def get(self, last_seq, timeout=2.0):
        """Newest (frame, time, seq) newer than last_seq; None on timeout or camera loss."""
        end = time.time() + timeout
        with self.cond:
            while self.seq == last_seq and self.alive:
                left = end - time.time()
                if left <= 0:
                    return None
                self.cond.wait(left)
            if self.seq == last_seq:
                return None
            return self.frame, self.t, self.seq


BACKENDS = {"ANY": cv2.CAP_ANY, "V4L2": cv2.CAP_V4L2, "DSHOW": cv2.CAP_DSHOW,
            "MSMF": cv2.CAP_MSMF, "GSTREAMER": cv2.CAP_GSTREAMER, "FFMPEG": cv2.CAP_FFMPEG}


def _fourcc_str(v):
    v = int(v)
    return "".join(chr((v >> (8 * i)) & 0xFF) for i in range(4)) if v > 0 else "?"


class OpenCVSource:
    """Video file, camera index, stream URL or GStreamer pipeline (anything cv2.VideoCapture opens)."""

    def __init__(self, key, sc):
        self.key, self.w, self.h = key, sc["width"], sc["height"]
        self.crop, self._crop_checked = sc.get("crop"), False
        src = sc["source"]
        self.is_file = os.path.isfile(str(src))
        pipeline = isinstance(src, str) and "!" in src
        is_url = isinstance(src, str) and "://" in src      # e.g. ESP32-CAM http://ip:81/stream
        backend = BACKENDS[sc.get("backend", "GSTREAMER" if pipeline else "ANY").upper()]
        self.cap = cv2.VideoCapture(int(src) if str(src).isdigit() else src, backend)
        if not self.cap.isOpened():
            raise SourceError(f"could not open source: {src} (backend {sc.get('backend', 'auto')})")
        self.iv = 1.0 / (self.cap.get(cv2.CAP_PROP_FPS) or 30)
        self.realtime = cfg.get("realtime_files", True)
        self.idx, self.t0 = 0, time.time()
        self.live, self.last_seq = None, 0
        if not self.is_file:
            cam = {} if is_url else sc.get("camera", {})   # format/size/fps cannot be set on a network stream
            if cam.get("fourcc"):                      # set the pixel format first (MJPG for 720p60)
                self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*cam["fourcc"]))
            if cam.get("width"):
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, cam["width"])
            if cam.get("height"):
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cam["height"])
            if cam.get("fps"):
                self.cap.set(cv2.CAP_PROP_FPS, cam["fps"])
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            gw, gh = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            gf, gc = self.cap.get(cv2.CAP_PROP_FPS), _fourcc_str(self.cap.get(cv2.CAP_PROP_FOURCC))
            log_message(f"[CAM] {key.upper()}: {gw}x{gh} @ {gf:.0f} fps, format {gc} -> processed at {self.w}x{self.h}")
            for name, want, got in (("width", cam.get("width"), gw), ("height", cam.get("height"), gh),
                                    ("fps", cam.get("fps"), round(gf)), ("format", cam.get("fourcc"), gc)):
                if want and want != got:
                    log_message(f"[CAM] {key.upper()}: asked for {name} {want} but the camera gives {got}.")
            if sc.get("camera", {}).get("latest_only", True):
                self.live = LiveCapture(self.cap)
                self.live.start()

    def proc(self, fr):
        if self.crop and not self._crop_checked:       # warn once if the crop would stretch the image
            self._crop_checked = True
            H0, W0 = fr.shape[:2]
            ca = (self.crop["w"] * W0) / (self.crop["h"] * H0)
            if abs(ca / (self.w / self.h) - 1) > 0.03:
                log_message(f"[CAM] {self.key.upper()}: crop aspect {ca:.2f} differs from processing aspect "
                            f"{self.w / self.h:.2f}; the image will be stretched.")
            else:
                log_message(f"[CAM] {self.key.upper()}: centre crop {self.crop} of {W0}x{H0}")
        return to_proc(fr, self.w, self.h, self.crop)

    def read(self):
        if self.live is not None:                      # live camera: always the newest frame
            item = self.live.get(self.last_seq)
            if item is None:
                return None
            fr, t, self.last_seq = item
            return self.proc(fr), t
        if self.is_file:
            due = int((time.time() - self.t0) / self.iv)
            if self.realtime:
                while self.idx < due:                  # behind the video clock: drop frames
                    if not self.cap.grab():
                        return self._eof()
                    self.idx += 1
            wait = self.t0 + self.idx * self.iv - time.time()
            if wait > 0:                               # ahead of the clock: wait for the next frame
                time.sleep(wait)
        ok, fr = self.cap.read()
        if not ok:
            return self._eof()
        self.idx += 1
        return self.proc(fr), time.time()

    def _eof(self):
        if cfg["loop_videos"] and self.is_file:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.idx, self.t0 = 0, time.time()
            ok, fr = self.cap.read()
            if ok:
                self.idx = 1
                return self.proc(fr), time.time()
        return None

    def release(self):
        if self.live is not None:
            self.live.stopping = True
        self.cap.release()



SOURCE_TYPES = {"opencv": OpenCVSource}



def make_source(key):
    kind = SENS[key].get("type", "opencv")
    if kind not in SOURCE_TYPES:
        raise SourceError(f"unknown source type '{kind}'. Available: {', '.join(SOURCE_TYPES)}")
    return SOURCE_TYPES[kind](key, SENS[key])


# --------------------------------------------------------------------------
# ONNX Runtime thread limit, model loading, inference
# --------------------------------------------------------------------------
_ort_local = threading.local()


def install_ort_thread_limit():
    """Two sensors share the CPU, so give each ONNX session a fixed thread count (onnx_threads).
    Applied to sessions created on a worker thread; 0 leaves ONNX Runtime's default."""
    try:
        import onnxruntime as ort
    except ImportError:
        return False
    orig = ort.InferenceSession

    def limited(*args, **kw):
        n = getattr(_ort_local, "threads", 0)
        if n > 0 and len(args) < 2:
            so = kw.get("sess_options") or ort.SessionOptions()
            so.intra_op_num_threads = n
            kw["sess_options"] = so
        return orig(*args, **kw)

    ort.InferenceSession = limited
    return True


def _norm_size(v):
    return [v, v] if isinstance(v, int) else [int(v[0]), int(v[1])]


def load_model(key):
    """Loads the sensor's model. ONNX inputs are fixed-size, so the size is read from the model itself
    and overrides sensors.<key>.imgsz. Returns (model, imgsz)."""
    sc, label = SENS[key], key.upper()
    path = sc["model_path"]
    if not os.path.exists(path):
        raise SourceError(f"model not found: {path} (export it: python export_models.py {key})")
    imgsz = sc.get("imgsz", 480)
    if path.lower().endswith(".onnx"):
        try:
            import onnxruntime as ort
            shape = ort.InferenceSession(path, providers=["CPUExecutionProvider"]).get_inputs()[0].shape
            h, w = shape[2], shape[3]
            if isinstance(h, int) and isinstance(w, int):
                if _norm_size(imgsz) != [h, w]:
                    log_message(f"[MODEL] {label}: ONNX exported at {h}x{w}, using that "
                                f"(config imgsz = {imgsz}).")
                imgsz = h if h == w else [h, w]
        except Exception as e:
            log_message(f"[MODEL] {label}: could not read ONNX input size ({e}); using {imgsz}.")
    log_message(f"[MODEL] {label}: {os.path.basename(path)} @ {imgsz} on CPU, frame {sc['width']}x{sc['height']}")
    return YOLO(path, task="detect"), imgsz


def infer(model, imgsz, frame, conf):
    return model.track(frame, conf=conf, persist=True, tracker="bytetrack.yaml",
                       imgsz=imgsz, verbose=False, device=DEVICE)[0].boxes


# --------------------------------------------------------------------------
# Sensor worker: reads frames and runs the model on its own thread
# --------------------------------------------------------------------------
class SensorWorker(threading.Thread):
    def __init__(self, key, source, model, imgsz, state, trackers):
        super().__init__(name=f"{key}-worker", daemon=True)
        self.key, self.source, self.model, self.imgsz = key, source, model, imgsz
        self.conf = SENS[key]["confidence"]
        self.state, self.trackers = state, trackers
        self.lock = threading.Lock()
        self.frame, self.result = None, None
        self.seq = self.seen = 0
        self.read_count = 0
        self.stop_flag = threading.Event()
        self.eof, self.error = False, None
        self.last_pub = time.time()      # when this worker last published a result
        self.peers = {}                  # all workers, set by main()

    def _should_infer(self):
        n = cfg.get("secondary_every_n", 1)
        aim = self.state["aim_sensor"]
        if n <= 1 or aim == self.key or aim not in self.peers:
            return True                                  # the aiming sensor runs on every frame
        if cfg.get("secondary_full_rate_when_unlocked", True):
            aim_stale = time.time() - self.peers[aim].last_pub > cfg.get("stale_result_ms", 500) / 1000.0
            if not self.trackers[aim].locked or aim_stale:
                return True                              # aiming sensor lost / frozen: run at full rate
        return self.read_count % n == 0

    def run(self):
        _ort_local.threads = cfg.get("onnx_threads", 0)
        try:
            while not self.stop_flag.is_set():
                item = self.source.read()
                if item is None:
                    break
                fr, t_cap = item
                self.read_count += 1
                with self.lock:
                    self.frame = fr
                if not self._should_infer():
                    continue
                boxes = infer(self.model, self.imgsz, fr, self.conf)
                with self.lock:
                    self.seq += 1
                    self.last_pub = time.time()
                    self.result = dict(boxes=boxes, t_cap=t_cap,
                                       ms=(time.time() - t_cap) * 1000, seq=self.seq)
        except Exception as e:  # surfaced by the main loop
            self.error = e
        finally:
            self.eof = True

    def take(self):
        """Newest inference result not yet consumed, or None."""
        with self.lock:
            if self.result is not None and self.result["seq"] != self.seen:
                self.seen = self.result["seq"]
                return self.result
        return None

    def get_frame(self):
        with self.lock:
            return None if self.frame is None else self.frame.copy()


def pick_source(state, trackers, fresh):
    """Which sensor drives the aim: manual locks first, then the aiming sensor, then the other.
    Sensors whose results are stale are skipped."""
    active = state["aim_sensor"]
    other = "ir" if active == "rgb" else "rgb"
    for need_manual in (True, False):
        for key in (active, other):
            tr = trackers[key]
            if tr.locked and (tr.manual or not need_manual) and fresh(key):
                return key
    return None


# --------------------------------------------------------------------------
# Rendering and UI. Each feed is letterboxed into a DW x DH cell, so sensors with different
# processing sizes or shapes sit side by side without being stretched.
# --------------------------------------------------------------------------
def fit_geom(w, h):
    s = min(DW / w, DH / h)
    nw, nh = max(1, int(round(w * s))), max(1, int(round(h * s)))
    return (DW - nw) // 2, (DH - nh) // 2, nw / w, nw, nh


CELL_GEOM = {k: fit_geom(SENS[k]["width"], SENS[k]["height"])[:3] for k in KEYS}   # (ox, oy, scale)


def render(frame, tr, title):
    """Letterbox the processed frame into its cell and draw overlays. Returns the cell image."""
    h, w = frame.shape[:2]
    ox, oy, s, nw, nh = fit_geom(w, h)
    CELL_GEOM[tr.key] = (ox, oy, s)
    cell = np.zeros((DH, DW, 3), np.uint8)
    interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
    cell[oy:oy + nh, ox:ox + nw] = cv2.resize(frame, (nw, nh), interpolation=interp)

    def T(x, y):
        return int(ox + x * s), int(oy + y * s)

    cx0, cy0 = T(w / 2, h / 2)
    cv2.line(cell, (cx0 - 12, cy0), (cx0 + 12, cy0), (90, 90, 90), 1)
    cv2.line(cell, (cx0, cy0 - 12), (cx0, cy0 + 12), (90, 90, 90), 1)
    rank_of = {}
    for rank, (_, d) in enumerate(tr.ranked, 1):
        rank_of[d["tid"]] = rank
        if tr.locked and d["tid"] == tr.locked_id:
            continue  # drawn below as the locked target
        p1, p2 = T(d["box"][0], d["box"][1]), T(d["box"][2], d["box"][3])
        cv2.rectangle(cell, p1, p2, (0, 255, 255), 1)
        cv2.putText(cell, f"T{rank}", (p1[0], max(p1[1] - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    if tr.box is not None:
        p1, p2 = T(tr.box[0], tr.box[1]), T(tr.box[2], tr.box[3])
        color = (0, 0, 255) if tr.status.startswith("LOCKED") else (0, 165, 255)
        cv2.rectangle(cell, p1, p2, color, 2)
        c = T(*tr.center)
        cv2.circle(cell, c, 5, (0, 255, 0), -1)
        cv2.putText(cell, f"T{rank_of.get(tr.locked_id, '-')} TARGET ({int(tr.center[0])},{int(tr.center[1])})",
                    (p1[0], max(p1[1] - 10, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    cv2.putText(cell, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    cv2.putText(cell, f"STATUS: {tr.status}", (15, 60), cv2.FONT_HERSHEY_SIMPLEX,
                0.55, (0, 255, 0), 2)
    return cell


def placeholder_cell(title, st):
    """What a sensor cell shows when the sensor is off, missing or lost."""
    cell = np.full((DH, DW, 3), 22, np.uint8)
    label, color = {"off": ("SENSOR OFF", (150, 150, 150)),
                    "missing": ("SENSOR MISSING", (0, 165, 255)),
                    "lost": ("SENSOR LOST", (0, 0, 255))}[st["state"]]
    cv2.putText(cell, title, (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 3)
    cv2.putText(cell, label, ((DW - tw) // 2, DH // 2), cv2.FONT_HERSHEY_SIMPLEX, 1.1, color, 3)
    msg = st["msg"][:64]
    (mw, mh), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.putText(cell, msg, (max(10, (DW - mw) // 2), DH // 2 + 36), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
    return cell


def panel(trackers, sstate):
    rows = cfg["panel_rows"]
    ph = 28 + rows * 18 + 8
    img = np.full((ph, DW * len(KEYS), 3), 18, np.uint8)
    for k, key in enumerate(KEYS):
        tr, x0 = trackers[key], k * DW + 15
        if sstate[key]["state"] != "ok":
            label = {"off": "SENSOR OFF", "missing": "SENSOR MISSING", "lost": "SENSOR LOST"}[sstate[key]["state"]]
            cv2.putText(img, f"{key.upper()}: {label}", (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
            continue
        cv2.putText(img, f"{key.upper()} TARGETS: {len(tr.dets)}   LARGEST GROUP: {tr.group}",
                    (x0, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 255), 1)
        for i, (sc, d) in enumerate(tr.ranked[:rows], 1):
            mark = ">" if (tr.locked and d["tid"] == tr.locked_id) else " "
            line = (f"{mark} T{i}  id {d['tid']:<4} score {sc:.2f}  conf {d['conf']:.2f}  "
                    f"({d['cx']:.0f},{d['cy']:.0f})")
            col = (0, 0, 255) if mark == ">" else (210, 210, 210)
            cv2.putText(img, line, (x0, 28 + i * 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1)
    cv2.line(img, (DW, 0), (DW, ph), (50, 50, 50), 2)
    return img


BUTTONS = [("aim", 80), ("auto", 85), ("areload", 112), ("reload", 75), ("release", 80),
           ("logs", 80), ("arm", 75), ("fire", 65)]


def active_buttons(state):
    """The AIM button only exists while both sensors are live; with one sensor the aim is fixed."""
    return [b for b in BUTTONS if b[0] != "aim" or len(state.get("live", KEYS)) > 1]


def button_rects(bw, buttons):
    """Buttons are laid out from the right edge of the banner (banner width = DW * number of feeds)."""
    rects, x = {}, bw - 10
    for name, wd in reversed(buttons):
        rects[name] = (x - wd, 8, x, 36)
        x -= wd + 8
    return rects


def draw_button(banner, text, rect, active=False, danger=False):
    x1, y1, x2, y2 = rect
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


def compose(cells, now, trackers, state, sstate):
    bw = DW * len(KEYS)
    feeds = np.hstack([cells[k] for k in KEYS])
    cv2.line(feeds, (DW, 0), (DW, DH), (50, 50, 50), 3)
    banner = np.full((BANNER_H, bw, 3), 25, np.uint8)

    cv2.putText(banner, "C-UAV COMMAND & CONTROL SYSTEM", (15, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    am = ammo_state(state, now)
    am_txt = {"loaded": "LOADED", "empty": "EMPTY - RELOAD", "unknown": "UNKNOWN",
              "reloading": f"RELOADING {max(0, state['reload_until'] - now):.0f}s"}[am]
    am_col = {"loaded": (0, 255, 0), "empty": (0, 0, 255), "unknown": (0, 0, 255),
              "reloading": (0, 165, 255)}[am]
    cv2.putText(banner, f"DART: {am_txt} [{state.get('ammo_src', 'timer').upper()}]   SHOTS: {state['shots']}", (15, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, am_col, 1)
    n = max([trackers[k].group for k in KEYS if sstate[k]["state"] == "ok"] or [0])
    if n >= cfg["swarm_alert_size"]:
        cv2.putText(banner, f"SWARM ALERT x{n}", (365, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

    r = button_rects(bw, active_buttons(state))
    if "aim" in r:
        draw_button(banner, f"AIM: {state['aim_sensor'].upper()}", r["aim"], active=True)
    draw_button(banner, f"AUTO: {'ON' if state['auto_shoot'] else 'OFF'}", r["auto"], active=state["auto_shoot"])
    draw_button(banner, f"A-RELOAD: {'ON' if state['auto_reload'] else 'OFF'}", r["areload"],
                active=state["auto_reload"])
    draw_button(banner, "RELOAD", r["reload"], active=ammo_state(state, now) == "reloading")
    draw_button(banner, "RELEASE", r["release"])
    draw_button(banner, f"LOGS: {'ON' if state['show_logs'] else 'OFF'}", r["logs"], active=state["show_logs"])
    draw_button(banner, f"ARM: {'ON' if state['armed'] else 'OFF'}", r["arm"],
                active=state["armed"], danger=state["armed"])
    draw_button(banner, "FIRE!", r["fire"], danger=True)

    parts = [banner, feeds, panel(trackers, sstate)]
    if state["show_logs"]:
        term_h = 140
        term = np.full((term_h, bw, 3), 12, np.uint8)
        cv2.rectangle(term, (0, 0), (bw, term_h), (30, 30, 30), 1)
        cv2.putText(term, ">> LIVE SYSTEM TERMINAL LOGS", (15, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 100), 1)
        cv2.line(term, (10, 28), (bw - 10, 28), (50, 50, 50), 1)
        for idx, line in enumerate(LOG_BUFFER[-5:]):
            cv2.putText(term, line, (15, 48 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                        (200, 200, 200), 1)
        cv2.putText(term, LIVE_LINE[0], (15, 48 + 5 * 16 + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.4,
                    (0, 200, 255), 1)
        parts.append(term)
    return np.vstack(parts)


def handle_click(x, y, trackers, state):
    """Mouse clicks (in composed-image pixels) on banner buttons or target bounding boxes."""
    if 0 <= y < BANNER_H:
        r = button_rects(DW * len(KEYS), active_buttons(state))
        hit = next((n for n, (x1, y1, x2, y2) in r.items() if x1 <= x <= x2 and y1 <= y <= y2), None)
        if hit == "aim":
            state["aim_sensor"] = "ir" if state["aim_sensor"] == "rgb" else "rgb"
            log_message(f"[UI] Aim Sensor switched to: {state['aim_sensor'].upper()}")
        elif hit == "auto":
            state["auto_shoot"] = not state["auto_shoot"]
            log_message(f"[UI] Auto-Shoot toggled to: {'ON' if state['auto_shoot'] else 'OFF'}")
        elif hit == "areload":
            state["auto_reload"] = not state["auto_reload"]
            if not state["auto_reload"]:
                state["reload_due"] = None                  # a queued automatic reload is cancelled
            log_message(f"[UI] Auto-Reload toggled to: {'ON' if state['auto_reload'] else 'OFF'}")
        elif hit == "reload":
            state["manual_reload_trigger"] = True
            log_message("[UI] RELOAD REQUESTED VIA BUTTON.")
        elif hit == "release":
            for tr in trackers.values():
                tr._release()
            log_message("[UI] All target locks released.")
        elif hit == "logs":
            state["show_logs"] = not state["show_logs"]
            log_message(f"[UI] Terminal logs toggled to: {'ON' if state['show_logs'] else 'OFF'}")
        elif hit == "arm":
            state["armed"] = not state["armed"]
            log_message(f"[UI] System {'ARMED' if state['armed'] else 'DISARMED'}.")
        elif hit == "fire":
            state["manual_fire_trigger"] = True
            log_message("[UI] MANUAL FIRE REQUESTED VIA BUTTON.")
        else:
            return False
        return True

    y -= BANNER_H
    if not (0 <= y < DH) or not (0 <= x < DW * len(KEYS)):
        return False
    k = int(x // DW)
    key, cx = KEYS[k], x - k * DW
    ox, oy, sc_ = CELL_GEOM[key]
    px, py = (cx - ox) / sc_, (y - oy) / sc_            # back to processing-frame pixels
    tol = 6 / sc_
    tr = trackers[key]
    hits = [d for _, d in tr.ranked
            if d["box"][0] - tol <= px <= d["box"][2] + tol and d["box"][1] - tol <= py <= d["box"][3] + tol]
    if hits:
        tr.manual_lock(min(hits, key=lambda d: d["w"] * d["h"]))
        log_message(f"[UI] Force-locked {key.upper()} target ID {tr.locked_id} via mouse click.")
        return True
    return False


# --------------------------------------------------------------------------
def main():
    log_message("Starting C-UAV v2 (ONNX / CPU, threaded, per-sensor config)...")
    state = {
        "aim_sensor": cfg.get("aim_sensor", "rgb").lower(),
        "show_logs": True,
        "auto_shoot": cfg.get("auto_shoot_enabled", False),
        "manual_fire_trigger": False,
        "armed": False,                 # always starts disarmed
        "live": (),                     # sensors currently delivering data
        "auto_cat": None, "auto_msg": "", "auto_log_t": 0.0,
        "auto_reload": cfg.get("auto_reload_enabled", False),
        "manual_reload_trigger": False,
        "reload_due": None,             # when the queued automatic reload should be sent
        "loaded": cfg.get("start_loaded", False),   # one dart fits in the barrel
        "reloading": False,
        "reload_until": 0.0,            # when the running reload times out
        "reload_sent_at": 0.0, "fired_at": -1e9,
        "esp_status": None,             # newest fresh status from the ESP32 (dart sensor), else None
        "ammo_src": "timer",
        "shots": 0,                     # fire commands since the last reload
    }
    if state["aim_sensor"] not in KEYS:
        sys.exit(f"aim_sensor must be one of {KEYS}")

    limited = install_ort_thread_limit()
    log_message(f"[PERF] cpu cores: {os.cpu_count()} | onnx_threads per sensor: "
                f"{cfg.get('onnx_threads', 0) or 'default'}{'' if limited else ' (onnxruntime not found)'} | "
                f"secondary_every_n: {cfg.get('secondary_every_n', 1)}")

    trackers = {k: StreamTracker(k) for k in KEYS}
    sstate, models, sources = {}, {}, {}
    for key in KEYS:                       # a sensor with no source is simply OFF: no model, no thread
        if not sensor_enabled(key):
            why = "disabled in config" if SENS[key].get("enabled", True) is False else "no source set in config"
            sstate[key] = {"state": "off", "msg": why}
            log_message(f"[SENSOR] {key.upper()}: OFF ({why})")
            continue
        try:
            models[key] = load_model(key)
            sources[key] = make_source(key)
            sstate[key] = {"state": "ok", "msg": ""}
        except Exception as e:             # configured but cannot start: show it, keep the other sensor running
            sstate[key] = {"state": "missing", "msg": str(e)}
            log_message(f"[SENSOR] {key.upper()}: MISSING - {e}")
    ok_keys = [k for k in KEYS if sstate[k]["state"] == "ok"]
    if not ok_keys:
        sys.exit("No sensor is active. Set a source for at least one sensor in config.json.")
    if state["aim_sensor"] not in ok_keys:
        state["aim_sensor"] = ok_keys[0]
        log_message(f"[MODE] Aim sensor set to {ok_keys[0].upper()} (the only active sensor).")
    state["live"] = tuple(ok_keys)

    workers = {k: SensorWorker(k, sources[k], models[k][0], models[k][1], state, trackers) for k in ok_keys}
    for w in workers.values():
        w.peers = workers
    titles = {k: SENS[k].get("title", k.upper()) for k in KEYS}

    link = Esp32Link(cfg["esp32"])
    tag = " [sent to ESP32]" if link.ser else " [DRY RUN: ESP32 disabled]"
    status_timeout = cfg["esp32"].get("status_timeout_s", 1.0)
    if link.ser is not None:                           # ask the ESP32 for the initial dart state
        t_end = time.time() + cfg["esp32"].get("status_wait_s", 2.0)
        while time.time() < t_end and link.status is None:
            link.poll()
            time.sleep(0.02)
        if link.status is not None and link.status["sensor_ok"]:
            state["esp_status"] = link.status
            log_message(f"[ESP32] Initial dart state from the sensor: {ammo_state(state, time.time()).upper()}")
        elif link.status is not None:
            log_message("[ESP32] The dart sensor reports a fault; using the reload timer.")
        else:
            log_message(f"[ESP32] No status within {cfg['esp32'].get('status_wait_s', 2.0):g}s: "
                        f"dart state uses the reload timer (start_loaded={state['loaded']}).")
    esp_fresh = link.status is not None
    show = cfg["show_window"]
    win = "C-UAV Command & Control Interface"
    scale = cfg.get("window_scale", 1.0)   # <1.0 shrinks the whole window to fit small screens
    if show:
        cv2.namedWindow(win)
        cv2.setMouseCallback(
            win, lambda ev, x, y, fl, p: ev == cv2.EVENT_LBUTTONDOWN
            and handle_click(int(x / scale), int(y / scale), trackers, state))

    stats = Stats()
    frames = {k: 0 for k in KEYS}                      # inference results since the last report
    last_result = {k: 0.0 for k in KEYS}               # arrival time of each sensor's newest result
    pending_cap = {k: None for k in KEYS}              # capture time of a result not yet sent as a command
    disp = {k: np.zeros((DH, DW, 3), np.uint8) for k in KEYS}
    csv_path = cfg.get("timing_csv", "")
    if csv_path and not os.path.exists(csv_path):
        with open(csv_path, "w") as f:
            f.write("time,ir_fps,ir_infer_avg_ms,ir_infer_max_ms,rgb_fps,rgb_infer_avg_ms,"
                    "rgb_infer_max_ms,cmd_avg_ms,cmd_max_ms\n")
    stale_s = cfg.get("stale_result_ms", 500) / 1000.0
    send_dt = 1.0 / cfg.get("send_hz", 30)
    draw_dt = 1.0 / cfg.get("ui_fps", 30)
    t_log = last_send = last_draw = time.time()
    last_fire_time = last_reload_time = 0.0
    quit_now = False

    for w in workers.values():
        w.start()
    try:
        while not quit_now:
            now = time.time()
            link.poll()                                  # dart-sensor status from the ESP32
            fresh = link.status is not None and (now - link.status_t) <= status_timeout
            if link.ser is not None and fresh != esp_fresh:
                log_message("[ESP32] Status link restored." if fresh else
                            "[ESP32] Status link lost: dart state falls back to the reload timer.")
                esp_fresh = fresh
            state["esp_status"] = link.status if fresh else None

            # 0) sensors that stopped delivering (video ended, camera unplugged) are marked LOST
            for key, w in workers.items():
                if w.eof and sstate[key]["state"] == "ok":
                    why = repr(w.error) if w.error else (
                        "video ended" if getattr(sources[key], "is_file", False) else "no frames from the camera")
                    sstate[key] = {"state": "lost", "msg": why}
                    trackers[key]._release()
                    trackers[key].dets, trackers[key].ranked, trackers[key].group = [], [], 0
                    trackers[key].status = "SENSOR LOST"
                    log_message(f"[SENSOR] {key.upper()}: LOST - {why}")
            live = tuple(k for k in ok_keys if sstate[k]["state"] == "ok")
            state["live"] = live
            if not live:
                break                                   # nothing left to track with
            if len(live) == 1 and state["aim_sensor"] != live[0]:
                state["aim_sensor"] = live[0]
                log_message(f"[MODE] Aim sensor set to {live[0].upper()} (the only active sensor).")

            # 1) consume new inference results (tracking is cheap and stays on this thread)
            got = False
            for key in live:
                res = workers[key].take()
                if res is None:
                    continue
                tr = trackers[key]
                stats.add(f"{key} infer", res["ms"])
                t1 = time.time()
                tr.step(tr.filter_detections(res["boxes"]))
                stats.add(f"{key} track", (time.time() - t1) * 1000)
                frames[key] += 1
                last_result[key] = time.time()
                pending_cap[key] = res["t_cap"]
                got = True

            # 2) command to the ESP32: right after a new result, and at send_hz as keep-alive
            if got or now - last_send >= send_dt:
                t_now = time.time()
                key = pick_source(state, trackers, lambda k: k in live and t_now - last_result[k] <= stale_s)
                src = trackers[key] if key else None
                should_fire, last_fire_time = decide_fire(state, src, t_now, last_fire_time, tag)
                do_reload, last_reload_time = decide_reload(state, t_now, last_reload_time, tag)
                if src:
                    pan, tilt = px_to_angles(*src.center, key)
                    link.send(pan, tilt, True, fire=should_fire, reload=do_reload)
                    if pending_cap[key] is not None:     # capture -> command, for the driving sensor
                        stats.add("cmd", (time.time() - pending_cap[key]) * 1000)
                        pending_cap[key] = None
                else:
                    link.send(0.0, 0.0, False, fire=False, reload=do_reload)
                last_send = t_now

            # 3) UI at ui_fps, independent of inference speed
            if now - last_draw >= draw_dt:
                last_draw = now
                for key in KEYS:
                    if key in live:
                        fr = workers[key].get_frame()
                        if fr is not None:
                            disp[key] = render(fr, trackers[key], titles[key])
                    else:
                        disp[key] = placeholder_cell(titles[key], sstate[key])
                if show:
                    img = compose(disp, now, trackers, state, sstate)
                    if scale != 1.0:
                        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
                    cv2.imshow(win, img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord("q"):
                        quit_now = True
                    if key == ord("s"):
                        if len(live) > 1:
                            state["aim_sensor"] = "ir" if state["aim_sensor"] == "rgb" else "rgb"
                            log_message(f"[MODE] Aim Sensor switched via key to: {state['aim_sensor'].upper()}")
                        else:
                            log_message(f"[MODE] Only {live[0].upper()} is active; aim is fixed to it.")
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
                    if key == ord("e"):
                        state["manual_reload_trigger"] = True
                        log_message("[KEY] Reload requested via hotkey 'e'.")
                    if key == ord("z"):
                        state["auto_reload"] = not state["auto_reload"]
                        if not state["auto_reload"]:
                            state["reload_due"] = None
                        log_message(f"[MODE] Auto-reload toggled via key to: {'ON' if state['auto_reload'] else 'OFF'}")
                    if key == ord("a"):
                        for tr in trackers.values():
                            tr.manual = False
                        log_message("[MODE] Automatic target selection restored.")
                    if key == ord("r"):
                        for tr in trackers.values():
                            tr._release()
                        log_message("[KEY] All target locks released.")

            # 4) report
            if now - t_log > 2.0:   # full detail on the console; one pinned live line in the UI terminal
                dt = now - t_log
                bits = []
                for k in KEYS:
                    if k in live:
                        print(f"[TIMING] {k.upper():3s} {frames[k] / dt:4.1f} fps | infer {stats.avg(k + ' infer'):3.0f} avg / "
                              f"{stats.mx(k + ' infer'):3.0f} max ms | tracking {stats.avg(k + ' track'):.1f} ms")
                        bits.append(f"{k.upper()} {frames[k] / dt:.1f} fps {stats.avg(k + ' infer'):.0f} ms")
                    else:
                        print(f"[TIMING] {k.upper():3s} {sstate[k]['state'].upper()}")
                        bits.append(f"{k.upper()} {sstate[k]['state'].upper()}")
                print(f"[TIMING] capture->command ({state['aim_sensor'].upper()} aim): "
                      f"{stats.avg('cmd'):3.0f} avg / {stats.mx('cmd'):3.0f} max ms   (goal < 50)")
                auto_txt = ("ON" + (f" ({state['auto_msg']})" if state["auto_msg"] else "")) if state["auto_shoot"] else "OFF"
                am = ammo_state(state, now)
                rl_txt = (f"DART {am.upper()} [{state['ammo_src'].upper()}]"
                          + (f" {max(0, state['reload_until'] - now):.0f}s" if am == "reloading" else "")
                          + f" | shots {state['shots']} | A-RELOAD {'ON' if state['auto_reload'] else 'OFF'}")
                set_live_line("TIMING " + " | ".join(bits) + f" | cmd {stats.avg('cmd'):.0f} ms | AUTO {auto_txt} | {rl_txt}")
                print(f"[STATUS] IR {trackers['ir'].status} | RGB {trackers['rgb'].status} | "
                      f"AIM: {state['aim_sensor'].upper()} | AUTO: {state['auto_shoot']} | ARMED: {state['armed']}")
                if csv_path and stats.d:
                    with open(csv_path, "a") as f:
                        f.write(f"{time.strftime('%H:%M:%S')},{frames['ir'] / dt:.1f},"
                                f"{stats.avg('ir infer'):.1f},{stats.mx('ir infer'):.1f},"
                                f"{frames['rgb'] / dt:.1f},{stats.avg('rgb infer'):.1f},"
                                f"{stats.mx('rgb infer'):.1f},{stats.avg('cmd'):.1f},{stats.mx('cmd'):.1f}\n")
                for k in KEYS:
                    frames[k] = 0
                t_log = now

            time.sleep(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        for w in workers.values():
            w.stop_flag.set()
        for w in workers.values():
            w.join(timeout=2.0)
        link.send(0.0, 0.0, False, fire=False)
        link.close()
        for s_ in sources.values():
            s_.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
