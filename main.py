import json
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

# Load Configuration File
with open("config.json", "r") as f:
  config = json.load(f)

DEVICE = config["device"]
ir_model = YOLO(config["thermal_model_path"])
rgb_model = YOLO(config["rgb_model_path"])

cap_ir = cv2.VideoCapture(config["ir_source"])
cap_rgb = cv2.VideoCapture(config["rgb_source"])

FRAME_W = config["frame_width"]
FRAME_H = config["frame_height"]
BANNER_H = 45


class TrackerState:

  def __init__(self):
    self.last_box = None
    self.last_center = None
    self.missing_frames = 0
    self.max_coast_frames = config["max_coast_frames"]
    self.status = "SEARCHING"
    # Dictionary to track position history for independent static clutter filtering per target ID
    self.static_track_history = {}


ir_state = TrackerState()
rgb_state = TrackerState()


def process_pipeline_logic(boxes_data, state, stream_label="Target"):
  boxes = []
  confs = []
  track_ids = None

  if boxes_data is not None and len(boxes_data) > 0:
    xyxy = boxes_data.xyxy.cpu().numpy()
    conf_vals = boxes_data.conf.cpu().numpy()
    track_ids = (
        boxes_data.id.cpu().numpy() if boxes_data.id is not None else None
    )

    for i in range(len(xyxy)):
      boxes.append(xyxy[i][:4])
      confs.append(float(conf_vals[i]))

  if len(boxes) > 0:
    # --- MINIMUM SIZE FILTER (Drops tiny objects like distant birds/noise) ---
    min_w = config.get("min_box_width", 12)
    min_h = config.get("min_box_height", 12)

    # --- INDEPENDENT STATIONARY CLUTTER FILTER (For both IR and RGB) ---
    if stream_label == "IR":
      dist_thresh = config["ir_static_distance_threshold"]
      max_frames_thresh = config["ir_static_max_frames"]
    else:
      dist_thresh = config["rgb_static_distance_threshold"]
      max_frames_thresh = config["rgb_static_max_frames"]

    filtered_boxes = []
    filtered_confs = []
    current_frame_ids = set()

    for i in range(len(boxes)):
      b = boxes[i]
      c = confs[i]

      # Calculate box width and height
      box_w = b[2] - b[0]
      box_h = b[3] - b[1]

      # Reject boxes that are too small (prevents bird glitches)
      if box_w < min_w or box_h < min_h:
        continue

      tid = int(track_ids[i]) if track_ids is not None else i
      curr_center = (int((b[0] + b[2]) / 2), int((b[1] + b[3]) / 2))
      current_frame_ids.add(tid)

      if tid in state.static_track_history:
        history = state.static_track_history[tid]
        start_center = history["start_center"]
        dist = np.hypot(
            curr_center[0] - start_center[0], curr_center[1] - start_center[1]
        )
        if dist <= dist_thresh:
          history["static_frames"] += 1
        else:
          history["start_center"] = curr_center
          history["static_frames"] = 0
      else:
        state.static_track_history[tid] = {
            "start_center": curr_center,
            "static_frames": 1,
        }

      # If the object remains stationary past the max frame threshold, filter it out
      if state.static_track_history[tid]["static_frames"] < max_frames_thresh:
        filtered_boxes.append(b)
        filtered_confs.append(c)

    # Clean up stale track histories
    stale_keys = [
        k for k in state.static_track_history if k not in current_frame_ids
    ]
    for k in stale_keys:
      del state.static_track_history[k]

    boxes = filtered_boxes
    confs = filtered_confs

    if len(boxes) == 0:
      if (
          state.last_box is not None
          and state.missing_frames < state.max_coast_frames
      ):
        state.missing_frames += 1
        state.status = (
            f"COASTING ({state.missing_frames}/{state.max_coast_frames})"
        )
        return state.last_box, state.last_center, state.status, "COAST", []
      else:
        state.status = "SEARCHING"
        state.last_box = None
        state.last_center = None
        return None, None, state.status, "SEARCHING", []

    state.missing_frames = 0
    if len(boxes) == 1:
      state.last_box = boxes[0]
      state.last_center = (
          int((boxes[0][0] + boxes[0][2]) / 2),
          int((boxes[0][1] + boxes[0][3]) / 2),
      )
      state.status = "SINGLE LOCKED"
      return state.last_box, state.last_center, state.status, "SINGLE", boxes
    else:
      centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])

      clustering = DBSCAN(
          eps=config["dbscan_eps"], min_samples=config["dbscan_min_samples"]
      ).fit(centers)
      labels = clustering.labels_

      unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

      if len(unique_labels) > 0:
        main_cluster_label = unique_labels[np.argmax(counts)]
        main_swarm_boxes = [
            boxes[i] for i in range(len(boxes)) if labels[i] == main_cluster_label
        ]

        min_x = min(b[0] for b in main_swarm_boxes)
        min_y = min(b[1] for b in main_swarm_boxes)
        max_x = max(b[2] for b in main_swarm_boxes)
        max_y = max(b[3] for b in main_swarm_boxes)

        macro_box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)
        centroid = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))

        state.last_box = macro_box
        state.last_center = centroid
        state.status = "SWARM LOCKED"
        return (
            state.last_box,
            state.last_center,
            state.status,
            "SWARM",
            main_swarm_boxes,
        )
      else:
        if state.last_center is not None:
          distances = [
              np.hypot(
                  ((b[0] + b[2]) / 2) - state.last_center[0],
                  ((b[1] + b[3]) / 2) - state.last_center[1],
              )
              for b in boxes
          ]
          best_idx = np.argmin(distances)
        else:
          best_idx = np.argmax(confs)

        best_box = boxes[best_idx]
        state.last_box = best_box
        state.last_center = (
            int((best_box[0] + best_box[2]) / 2),
            int((best_box[1] + best_box[3]) / 2),
        )
        state.status = "SINGLE LOCKED"
        return (
            state.last_box,
            state.last_center,
            state.status,
            "SINGLE",
            [best_box],
        )
  else:
    if (
        state.last_box is not None
        and state.missing_frames < state.max_coast_frames
    ):
      state.missing_frames += 1
      state.status = f"COASTING ({state.missing_frames}/{state.max_coast_frames})"
      return state.last_box, state.last_center, state.status, "COAST", []
    else:
      state.status = "SEARCHING"
      state.last_box = None
      state.last_center = None
      return None, None, state.status, "SEARCHING", []


print("Starting Aegis C-UAS Command & Control Interface...")
print("Press 'q' in the video window to terminate execution.")

fps_ir = cap_ir.get(cv2.CAP_PROP_FPS)
fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS)
fps_ir = fps_ir if fps_ir > 0 else 30
fps_rgb = fps_rgb if fps_rgb > 0 else 3

interval_ir = 1.0 / fps_ir
interval_rgb = 1.0 / fps_rgb

next_ir_time = time.time()
next_rgb_time = time.time()

ret_ir, frame_ir = cap_ir.read()
ret_rgb, frame_rgb = cap_rgb.read()

res_ir_boxes = None
res_rgb_boxes = None

while cap_ir.isOpened() and cap_rgb.isOpened():
  current_time = time.time()

  if current_time >= next_ir_time:
    ret_ir, frame_ir = cap_ir.read()
    if not ret_ir:
      break
    next_ir_time = current_time + interval_ir

    if frame_ir is not None:
      frame_ir = cv2.resize(frame_ir, (FRAME_W, FRAME_H))
      res_ir = ir_model.track(
          frame_ir,
          conf=config["thermal_confidence"],
          persist=True,
          tracker="bytetrack.yaml",
          verbose=False,
          device=DEVICE,
      )[0]
      res_ir_boxes = res_ir.boxes

  if current_time >= next_rgb_time:
    ret_rgb, frame_rgb = cap_rgb.read()
    if not ret_rgb:
      break
    next_rgb_time = current_time + interval_rgb

    if frame_rgb is not None:
      frame_rgb = cv2.resize(frame_rgb, (FRAME_W, FRAME_H))
      res_rgb = rgb_model.track(
          frame_rgb,
          conf=config["rgb_confidence"],
          persist=True,
          tracker="bytetrack.yaml",
          verbose=False,
          device=DEVICE,
      )[0]
      res_rgb_boxes = res_rgb.boxes

  if frame_ir is None or frame_rgb is None:
    break

  display_ir = frame_ir.copy()
  display_rgb = frame_rgb.copy()

  ir_box, ir_center, ir_status, ir_mode, _ = process_pipeline_logic(
      res_ir_boxes, ir_state, stream_label="IR"
  )
  if ir_box is not None:
    x1, y1, x2, y2 = map(int, ir_box)
    box_color = (0, 0, 255) if ir_mode != "COAST" else (0, 165, 255)
    cv2.rectangle(display_ir, (x1, y1), (x2, y2), box_color, 2)
    if ir_center:
      cv2.circle(display_ir, ir_center, 5, (0, 255, 0), -1)
      cv2.putText(
          display_ir,
          f"IR Target {ir_center}",
          (x1, max(y1 - 10, 20)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          box_color,
          2,
      )

  rgb_box, rgb_center, rgb_status, rgb_mode, raw_boxes = process_pipeline_logic(
      res_rgb_boxes, rgb_state, stream_label="RGB"
  )
  if rgb_box is not None:
    x1, y1, x2, y2 = map(int, rgb_box)

    if rgb_mode == "SWARM":
      cv2.rectangle(display_rgb, (x1, y1), (x2, y2), (255, 0, 255), 3)
      for b in raw_boxes:
        bx1, by1, bx2, by2 = map(int, b)
        cv2.rectangle(display_rgb, (bx1, by1), (bx2, by2), (255, 255, 0), 1)
    elif rgb_mode == "SINGLE":
      cv2.rectangle(display_rgb, (x1, y1), (x2, y2), (0, 0, 255), 2)
    else:
      cv2.rectangle(display_rgb, (x1, y1), (x2, y2), (0, 165, 255), 2)

    if rgb_center:
      cv2.circle(display_rgb, rgb_center, 6, (0, 255, 0), -1)
      cv2.putText(
          display_rgb,
          f"Optical Target {rgb_center}",
          (x1, max(y1 - 10, 20)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
          2,
      )

  cv2.putText(
      display_ir,
      "SENSOR 01: THERMAL IR",
      (15, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 255, 255),
      2,
  )
  cv2.putText(
      display_rgb,
      "SENSOR 02: OPTICAL RGB",
      (15, 30),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (0, 255, 255),
      2,
  )

  cv2.putText(
      display_ir,
      f"STATUS: {ir_status}",
      (15, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (0, 255, 0),
      2,
  )
  cv2.putText(
      display_rgb,
      f"STATUS: {rgb_status}",
      (15, 60),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.55,
      (0, 255, 0),
      2,
  )

  feeds_view = np.hstack((display_ir, display_rgb))
  cv2.line(feeds_view, (FRAME_W, 0), (FRAME_W, FRAME_H), (50, 50, 50), 3)

  banner = np.zeros((BANNER_H, FRAME_W * 2, 3), dtype=np.uint8)
  banner[:] = (25, 25, 25)

  timestamp_str = time.strftime(
      "%Y-%m-%d %H:%M:%S", time.localtime(current_time)
  )
  cv2.putText(
      banner,
      "AEGIS C-UAS COMMAND & CONTROL SYSTEM",
      (20, 28),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.6,
      (255, 255, 255),
      2,
  )
  cv2.putText(
      banner,
      f"SYS TIME: {timestamp_str} | MODE: ACTIVE TRACKING",
      (FRAME_W + 40, 28),
      cv2.FONT_HERSHEY_SIMPLEX,
      0.5,
      (0, 200, 255),
      1,
  )

  combined_view = np.vstack((banner, feeds_view))
  cv2.imshow("Aegis C-UAS Command & Control Interface", combined_view)

  if cv2.waitKey(1) & 0xFF == ord("q"):
    break

cap_ir.release()
cap_rgb.release()
cv2.destroyAllWindows()