# C-UAV v2 (ONNX, CPU, threaded)

Two sensor slots (`ir`, `rgb`) feed one single-target tracker. Each slot can be any camera or video;
the slot names are only labels. Aim commands go to an ESP32 over serial.

## Quick start
```
pip install -r requirements-export.txt     # on the PC that has the .pt files
python export_models.py                    # writes the .onnx files named in config.json
pip install -r requirements.txt            # on the Pi / mini PC
python main.py                             # or: python main.py config.cameras.example.json
```
Hotkeys: `q` quit, `s` switch aim sensor, `x` arm/disarm, `f` fire, `space` auto-fire, `r` release locks,
`a` auto target selection, `l` log panel, `e` reload, `z` auto-reload on/off. Everything is also clickable. Click a box to force-lock it.
The system always starts DISARMED; fire commands are blocked until ARM is on.

## Files
| File | Purpose |
|---|---|
| `main.py` | the application |
| `config.json` | your test setup (video files) |
| `config.cameras.example.json` | live cameras: 720p60 RGB + a second camera |
| `export_models.py` | `.pt` to `.onnx`, per sensor, square or rectangular |
| `requirements.txt` / `requirements-export.txt` | runtime / export dependencies |
| `timing_log.csv` | written while running (fps, inference ms, capture-to-command ms) |

## Per-sensor settings (`sensors.ir`, `sensors.rgb`)
| Key | Meaning |
|---|---|
| `source` empty | **Sensor OFF**: no model is loaded and no inference runs. The cell shows `SENSOR OFF`. Same with `"enabled": false` |
| `type` | `opencv` (files, camera index, URL, GStreamer pipeline) or `picamera2` (Pi CSI, untested) |
| `source` | file path, camera index (0, 1), stream URL, or a GStreamer pipeline string |
| `backend` | optional: `V4L2` (Linux), `DSHOW` / `MSMF` (Windows), `GSTREAMER`, `FFMPEG`, `ANY` |
| `camera` | live cameras only: `fourcc` (use `MJPG` for 720p60 over USB), `width`, `height`, `fps`, `latest_only` |
| `width`, `height` | processing size. Frames are resized to this; all pixel settings below refer to it |
| `hfov_deg` | horizontal field of view of the processed image (used for pixel-to-angle) |
| `fov_across` | `width` (default) or `height`: which axis of the processed image `hfov_deg` spans. Use `height` for a circular fisheye image whose circle fills the frame height |
| `crop` | optional `{x, y, w, h}` as fractions (0-1) of the raw camera frame, applied before resizing. A centred crop is a digital zoom that keeps native pixel detail. Keep its shape equal to `width`:`height` (the script warns if not) |
| `lens` | `rectilinear` (normal lens, default) or `fisheye` (equidistant model, for wide lenses like the OV2710 fisheye) |
| `offset` | `{x, y}` pixels added before the angle conversion (boresight correction) |
| `model_path`, `imgsz`, `confidence` | model file, input size (`480` or `[height, width]`), detection threshold |
| `group_eps` | swarm-alert grouping distance in processing pixels |

Any tracker setting (`min_box_width`, `static_gate_px`, `gate_px`, `max_coast_frames`, `kalman_*`...)
can be placed inside a sensor block to override the global value for that sensor only.

### Choosing sizes
- Keep the processing size the same shape as the camera (16:9 camera -> 640x360), so nothing is stretched.
- ONNX inputs are fixed, so export at the size you will run. Multiples of 32 only.
  16:9 at 640x360 -> `imgsz: [384, 640]`; 4:3 at 640x480 -> `480` or `[480, 640]`.
- More camera pixels do not help unless the model input also grows (which costs speed).
- Pixel-based values (box sizes, gates) scale with the processing size. Retune them if you change it.

## Sensors that are off, missing or lost
- **OFF**: `source` is empty (or `enabled: false`). Nothing is loaded or run; the cell reads `SENSOR OFF`.
- **MISSING**: a source is set but the camera/file could not be opened, or the model file is missing. The cell
  reads `SENSOR MISSING` with the reason; the other sensor keeps running.
- **LOST**: a sensor that was working stopped (video ended, camera unplugged). Its lock is dropped and the cell
  reads `SENSOR LOST`.
- With only one live sensor the **AIM button disappears** and the aim follows the live sensor automatically
  (also if `aim_sensor` in the config names the sensor that is off). The program exits when no sensor is left.
- To enable the IR sensor later: set `sensors.ir.source` (for the ESP32-CAM: `http://<its-ip>:81/stream`),
  put the new ONNX model at `sensors.ir.model_path`, and restart. Nothing else changes.

## Auto-fire: why it is not shooting
Auto-fire only shoots when ALL of these hold: AUTO is on, system is ARMED, the lock is on a real detection
(not coasting), confidence >= `auto_shoot_confidence`, lock age >= `auto_shoot_min_age`, the target is within
`auto_shoot_max_error_deg` of the crosshair, and the cooldown has passed. The log now says which one is
blocking, e.g. `[AUTO] Holding fire: target 4.2 deg off the crosshair (limit 2.0)`. Every shot is logged with
`[DRY RUN: ESP32 disabled]` or `[sent to ESP32]`.
When testing with video files nothing steers the camera, so the target is rarely on the crosshair. To test the
firing path on videos, set `auto_shoot_max_error_deg` to 90 (set it back to about 2 for the turret).
The pinned line at the bottom of the terminal panel shows the current AUTO status and timing.

## Camera notes
- `latest_only` (default on) reads the camera in a background thread and always hands out the newest frame,
  which keeps latency low with 60-120 fps cameras. The log prints what the camera actually delivered;
  `[CAM] ... asked for fps 60 but the camera gives 30` means the format or USB bandwidth is limiting it.
- USB cameras on Linux are often slow in their default format. Set `fourcc: "MJPG"`.
- **NoIR is not thermal.** A Pi NoIR camera is an ordinary camera without the IR filter. It is not a thermal
  imager, and a model trained on thermal images will not transfer to it. Train a model on images from the
  sensor you will actually fly with.
- **ELP-USBFHD01M-BL180 (OV2710, 180 degree fisheye, USB 2.0):** the interface is USB 2.0, so use `fourcc: "MJPG"`.
  Uncompressed YUY2 at high resolutions only manages a few frames per second over USB 2.0. Mode lists on this
  module family usually include 1080p at 30 fps, 720p at 60 fps and 480p at a higher rate, but check yours
  (`v4l2-ctl --list-formats-ext` on Linux); the startup log shows what the camera really delivers.
- **A 180 degree lens makes drones tiny.** Estimate (equidistant lens near the image centre, 0.3 m drone):
  full 180 degree view at 640 px wide is about 3.6 px per degree, so a drone is about 6 px at 10 m and 3 px at
  20 m. A centred crop of half the width (90 degrees across 640 px, as in `config.cameras.example.json`) gives
  about 7 px per degree: roughly 12 px at 10 m and 6 px at 20 m. Narrower crops reach further but watch a smaller
  area. Test your real range.
- **Check the image when the camera arrives:** if it is a circle with black sides, set `fov_across` to `height`
  and recompute `hfov_deg` for the processed image. The fisheye model assumes an equidistant lens, which is an
  approximation of the real lens.
- Models trained on normal photos struggle with fisheye distortion, especially near the edges. Train on images
  from this camera (centre crop if you use one).
- **ESP32-CAM as the IR sensor:** use its stream URL as `source` (the script then reads it like a camera, newest
  frame only). Expect WiFi delay of a few hundred ms, which is fine for a future IR upgrade but not for the
  50 ms goal. It is only the camera; motors still need a normal ESP32 board. With the IR filter removed it sees
  850 nm light, so a visible-light-blocking filter helps in daytime. Never look into IR LED boards at close range;
  you cannot see the light and your eye will not blink.
- Pi CSI cameras: either `type: "picamera2"` (untested here), or `type: "opencv"` with `backend: "GSTREAMER"`
  and a libcamera pipeline in `source`.

## Adding a new sensor type
A source is any class with `read() -> (frame_bgr, capture_time) | None` and `release()`. Write it, add it to
`SOURCE_TYPES` in `main.py`, then use its name in `type`. Resizing to the processing size is done with
`to_proc(frame, w, h)`.

## Speed
- Aim sensor runs on every frame; the other runs every `secondary_every_n` frames (every frame while the
  aim sensor has no lock or has frozen).
- `onnx_threads`: CPU threads per model. Try 0 (default), 2 and 4 and compare the `[TIMING]` lines.
- Check the `[TIMING]` console lines or `timing_log.csv`. The capture-to-command figure starts when a frame
  arrives and ends when the packet is written to the serial port. It does not include camera exposure or
  the ESP32 and motor response.

## Reload and the dart state
One dart fits in the barrel, so every shot needs a reload. The script tracks this as the **dart state**
(shown top-left in the banner and in the terminal's pinned line):
- `LOADED`: fire is allowed (together with ARM and the other rules).
- `EMPTY`: a fire command was sent and no reload has been sent since. **Fire is blocked** (`no dart loaded`).
- `RELOADING 4s`: a reload command was sent; fire stays blocked until `reload_timeout_s` has passed. The PC gets
  no feedback from the ESP32 yet, so set this to the real reload time. After it, the state returns to `LOADED`.
- At start the state is `EMPTY` unless `start_loaded` is true. Press RELOAD once (or set `start_loaded`).

Buttons:
- **RELOAD / `e`**: sends one reload request now (no ARM needed). A second press within
  `reload_min_interval_s` is ignored.
- **A-RELOAD / `z`** (starts OFF, or set `auto_reload_enabled`): after every fire command (manual or auto) one
  reload request is sent `auto_reload_delay_s` later, so the shot has left before the turret moves. With
  A-RELOAD off, no reload is ever sent on its own and the state stays `EMPTY` until you press RELOAD. Turning it
  off cancels a reload that is still queued.
- Reload commands are logged as `[RELOAD] ... [sent to ESP32]` or `[DRY RUN: ESP32 disabled]`.

## ESP32 packet (10 bytes)
`0xA5 | 0x01 | seq(u16) | pan_mrad(i16) | tilt_mrad(i16) | flags(u8) | xor checksum`
flags: bit0 track valid, bit1 fire request, bit2 reload request. The firmware must enforce its own arm switch, fire cooldown,
travel limits and a watchdog (zero motion if no packet for about 150 ms). Do not rely on the PC for safety.

## Known limits
- Tested here only with fake detectors, synthetic video and a fake serial port. Real-model speed,
  `onnx_threads` behaviour and the `picamera2` source need testing on your hardware.
- With `show_window: false` there is no way to arm, so fire stays blocked (safe).
- Two sensor slots (`ir`, `rgb`) are built in. More than two needs UI changes.
