import json
import time
import cv2
import numpy as np
from sklearn.cluster import DBSCAN
from ultralytics import YOLO

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
    self.locked_track_id = None
    self.last_box = None
    self.last_center = None
    self.missing_frames = 0
    self.max_coast_frames = config["max_coast_frames"]
    self.status = "SEARCHING"
    self.track_history = {}
    self.track_spawn_pos = {}


ir_state = TrackerState()
rgb_state = TrackerState()


def process_pipeline_logic(boxes_data, state, stream_label="Target"):
  boxes, confs, track_ids = [], [], []

  if boxes_data is not None and len(boxes_data) > 0:
    xyxy = boxes_data.xyxy.cpu().numpy()
    conf_vals = boxes_data.conf.cpu().numpy()
    ids = boxes_data.id.cpu().numpy() if boxes_data.id is not None else None

    min_w = config.get("min_box_width", 12)
    min_h = config.get("min_box_height", 12)
    min_ar = config.get("min_aspect_ratio", 0.25)
    max_ar = config.get("max_aspect_ratio", 4.0)

    current_frame_ids = set()
    valid_boxes, valid_confs, valid_ids = [], [], []

    for i in range(len(xyxy)):
      b = xyxy[i][:4]
      w = b[2] - b[0]
      h = b[3] - b[1]

      # 1. Size Filter
      if w < min_w or h < min_h:
        continue

      # 2. Aspect Ratio Filter (Poles)
      aspect_ratio = w / (h + 1e-5)
      if aspect_ratio < min_ar or aspect_ratio > max_ar:
        continue

      tid = int(ids[i]) if ids is not None else i
      current_frame_ids.add(tid)

      center_x = (b[0] + b[2]) / 2
      center_y = (b[1] + b[3]) / 2

      if tid not in state.track_history:
        state.track_history[tid] = []
        state.track_spawn_pos[tid] = (center_x, center_y)

      state.track_history[tid].append((center_x, center_y))
      if len(state.track_history[tid]) > 30:
        state.track_history[tid].pop(0)

      # 3. Motion-Gate Filter: Drop tracks that stay anchored to their spawn point (Clouds/Static Clutter)
      if len(state.track_history[tid]) >= 15:
        spawn_x, spawn_y = state.track_spawn_pos[tid]
        distance_from_birth = np.hypot(center_x - spawn_x, center_y - spawn_y)
        # If it has been tracked for 15+ frames but hasn't moved at least 8 pixels from where it appeared, it's a cloud
        if distance_from_birth < 8.0:
          continue

      valid_boxes.append(b)
      valid_confs.append(float(conf_vals[i]))
      valid_ids.append(tid)

    # Cleanup stale history and spawn positions
    stale_keys = [
        k for k in state.track_history if k not in current_frame_ids
    ]
    for k in stale_keys:
      del state.track_history[k]
      if k in state.track_spawn_pos:
        del state.track_spawn_pos[k]

    boxes, confs, track_ids = valid_boxes, valid_confs, valid_ids

  dbscan_eps = (
      config["ir_dbscan_eps"]
      if stream_label == "IR"
      else config["rgb_dbscan_eps"]
  )
  dbscan_min_samples = (
      config["ir_dbscan_min_samples"]
      if stream_label == "IR"
      else config["rgb_dbscan_min_samples"]
  )

  if len(boxes) == 0:
    if (
        state.last_box is not None
        and state.missing_frames < state.max_coast_frames
    ):
      state.missing_frames += 1
      state.status = f"COASTING ({state.missing_frames})"
      return state.last_box, state.last_center, state.status, "COAST", []
    else:
      state.status, state.locked_track_id = "SEARCHING", None
      state.last_box, state.last_center = None, None
      return None, None, state.status, "SEARCHING", []

  state.missing_frames = 0

  # --- PRIORITY 1: EVALUATE SWARM CLUSTERING FIRST ---
  if len(boxes) > 1:
    centers = np.array([[(b[0] + b[2]) / 2, (b[1] + b[3]) / 2] for b in boxes])
    clustering = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit(
        centers
    )
    labels = clustering.labels_
    unique_labels, counts = np.unique(labels[labels != -1], return_counts=True)

    if len(unique_labels) > 0:
      main_cluster_label = unique_labels[np.argmax(counts)]
      main_swarm_boxes = [
          boxes[i] for i in range(len(boxes)) if labels[i] == main_cluster_label
      ]

      if len(main_swarm_boxes) > 1:
        min_x = min(b[0] for b in main_swarm_boxes)
        min_y = min(b[1] for b in main_swarm_boxes)
        max_x = max(b[2] for b in main_swarm_boxes)
        max_y = max(b[3] for b in main_swarm_boxes)
        state.last_box = np.array([min_x, min_y, max_x, max_y], dtype=np.float32)
        state.last_center = (int((min_x + max_x) / 2), int((min_y + max_y) / 2))
        state.status = "SWARM LOCKED"
        state.locked_track_id = None
        return (
            state.last_box,
            state.last_center,
            state.status,
            "SWARM",
            main_swarm_boxes,
        )

  # --- PRIORITY 2: SINGLE TARGET FALLBACK ---
  if len(boxes) == 1:
    selected_idx = 0
  elif state.locked_track_id in track_ids:
    selected_idx = track_ids.index(state.locked_track_id)
  elif state.last_center is not None:
    selected_idx = np.argmin(
        [
            np.hypot(
                ((b[0] + b[2]) / 2) - state.last_center[0],
                ((b[1] + b[3]) / 2) - state.last_center[1],
            )
            for b in boxes
        ]
    )
  else:
    selected_idx = np.argmax(confs)

  chosen_box = boxes[selected_idx]
  state.locked_track_id = track_ids[selected_idx]
  state.last_box = chosen_box
  state.last_center = (
      int((chosen_box[0] + chosen_box[2]) / 2),
      int((chosen_box[1] + chosen_box[3]) / 2),
  )
  state.status = "LOCKED"
  return state.last_box, state.last_center, state.status, "SINGLE", [chosen_box]


print("Starting Aegis C-UAS Command & Control Interface...")

fps_ir = cap_ir.get(cv2.CAP_PROP_FPS) or 30
fps_rgb = cap_rgb.get(cv2.CAP_PROP_FPS) or 3
interval_ir, interval_rgb = 1.0 / fps_ir, 1.0 / fps_rgb
next_ir_time, next_rgb_time = time.time(), time.time()

ret_ir, frame_ir = cap_ir.read()
ret_rgb, frame_rgb = cap_rgb.read()
res_ir_boxes, res_rgb_boxes = None, None

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

  display_ir, display_rgb = frame_ir.copy(), frame_rgb.copy()

  ir_box, ir_center, ir_status, ir_mode, raw_ir_boxes = process_pipeline_logic(
      res_ir_boxes, ir_state, stream_label="IR"
  )
  if ir_box is not None:
    x1, y1, x2, y2 = map(int, ir_box)
    if ir_mode == "SWARM":
      cv2.rectangle(display_ir, (x1, y1), (x2, y2), (255, 0, 255), 3)
      for b in raw_ir_boxes:
        bx1, by1, bx2, by2 = map(int, b)
        cv2.rectangle(display_ir, (bx1, by1), (bx2, by2), (255, 255, 0), 1)
    else:
      cv2.rectangle(
          display_ir,
          (x1, y1),
          (x2, y2),
          (0, 0, 255) if ir_mode == "SINGLE" else (0, 165, 255),
          2,
      )

    if ir_center:
      cv2.circle(display_ir, ir_center, 5, (0, 255, 0), -1)
      cv2.putText(
          display_ir,
          f"IR Target {ir_center}",
          (x1, max(y1 - 10, 20)),
          cv2.FONT_HERSHEY_SIMPLEX,
          0.5,
          (0, 0, 255),
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
    else:
      cv2.rectangle(
          display_rgb,
          (x1, y1),
          (x2, y2),
          (0, 0, 255) if rgb_mode == "SINGLE" else (0, 165, 255),
          2,
      )

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