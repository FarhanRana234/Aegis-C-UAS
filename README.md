
# Aegis C-UAS (Counter-Unmanned Aerial Systems)

A dual-sensor Ground Control Station (GCS) and computer vision tracking pipeline for Counter-UAS applications.  
Aegis processes asynchronous **Thermal IR** and **Optical RGB** video feeds with advanced target tracking, swarm macro-clustering, stationary clutter rejection, and ByteTrack persistence.

> **Note:** This is a bench prototype focused on dual-sensor tracking and swarm prioritisation for a conceptual wide-area but targeted High-Power Microwave (HPM) effector. It is not a fielded system.

---

## 🚀 Key Features

* **Dual-Sensor Asynchronous Pipeline**  
  Independent handling of Thermal IR and Optical RGB streams that may have mismatched frame rates. Prevents one sensor from blocking the other and keeps the UI responsive.

* **Stationary Thermal Clutter Suppression**  
  Background objects (sun-baked rocks, rooftops, asphalt, etc.) that retain heat frequently produce false positives that look like drones.  
  The system measures pixel displacement of candidate boxes. If a detection moves **less than 2 pixels across approximately 10 consecutive frames**, it is treated as static clutter and ignored.  
  This was added because the IR stream was repeatedly switching between multiple stationary heat signatures.

* **ByteTrack Persistence + Coasting**  
  YOLO11 detections are passed through ByteTrack for temporal consistency so the bounding box does not glitch on brief detection dropouts.  
  When a track is temporarily lost (occlusion or low confidence), the last known box is held for up to **15 frames** (coasting).  
  Observed coast success rate on the test sequences is approximately **80%** (visible in the LinkedIn demo video).

* **Swarm Macro-Clustering (DBSCAN)**  
  Dense groups of drones are clustered into a single macro bounding box while individual track coordinates are retained.  
  Distant or isolated drones outside the main cluster are de-prioritised when a dense swarm is present.  
  This matches the operational concept of a wide-area but targeted HPM effector — focusing energy on the densest threat group increases hit probability instead of spreading power across scattered single drones.

* **Sticky Single-Target Tracking**  
  When multiple unclustered detections exist, the system prefers spatial nearest-neighbour continuity over raw confidence scores. This prevents the classic “box jumping” / target-switching jitter.

---

## 📊 Design Decisions & Observed Performance

| Component                      | Rationale / Observation                                                                 | Result / Metric                          |
|--------------------------------|-----------------------------------------------------------------------------------------|------------------------------------------|
| IR Clutter Filter              | IR stream kept locking onto static heat sources that resembled drones                   | < 2 px movement over ~10 frames → ignore |
| ByteTrack + Coasting           | Reduce glitching and brief detection dropouts                                           | Coast success ≈ 80% on test clips        |
| Macro-Clustering (DBSCAN)      | HPM is wide-area but energy should be concentrated                                      | Prioritises dense swarm over distant singles |
| Dual asynchronous pipelines    | IR and RGB rarely share the same native FPS                                             | Independent advance without stutter      |

These figures come from the bench prototype test sequences shown in the LinkedIn demo. They are observed results rather than formal benchmark numbers.

---

## 🔗 Roboflow Dataset Sources

The custom models were trained on multi-source datasets curated via Roboflow and exported in **YOLOv11** format.

* **Thermal Dataset 1:** [Maor Ovadia – Thermal Drone (v11)](https://universe.roboflow.com/maor-ovadia/thermal-drone-47qfh/dataset/11)
* **Thermal Dataset 2:** [Nguyn Quyt – Thermal Drone (v2)](https://universe.roboflow.com/nguyn-quyt/thermal-drone/dataset/2)
* **Optical Dataset 1:** [UAV Swarm Augmentation (v1)](https://universe.roboflow.com/project-8cfvz/uavswarmv_augmentation/dataset/1)
* **Optical Dataset 2:** [Drone Detection (v1)](https://universe.roboflow.com/project-986i8/drone-uskpc/dataset/1)

### Downloading via Python

```python
!pip install roboflow
from roboflow import Roboflow

rf = Roboflow(api_key="API_key_here")

# Thermal Dataset 1
project = rf.workspace("maor-ovadia").project("thermal-drone-47qfh")
dataset_thermal_1 = project.version(11).download("yolov11")

# Thermal Dataset 2
project = rf.workspace("nguyn-quyt").project("thermal-drone")
dataset_thermal_2 = project.version(2).download("yolov11")

# Optical Dataset 1
project = rf.workspace("project-8cfvz").project("uavswarmv_augmentation")
dataset_rgb_1 = project.version(1).download("yolov11")

# Optical Dataset 2
project = rf.workspace("project-986i8").project("drone-uskpc")
dataset_rgb_2 = project.version(1).download("yolov11")
```

---

## 🛠️ Dataset Merge Script (`merge_datasets.py`)

When multiple YOLO-format datasets are downloaded from Roboflow, filenames often collide (e.g. `image_001.jpg`). This script merges them into a single directory with automatic prefixing:

```python
import os
import shutil
from pathlib import Path

def merge_yolo_datasets(source_dirs, output_dir, dataset_prefixes):
    out_path = Path(output_dir)
    splits = ["train", "valid", "test"]

    for split in splits:
        (out_path / split / "images").mkdir(parents=True, exist_ok=True)
        (out_path / split / "labels").mkdir(parents=True, exist_ok=True)

    for src_dir, prefix in zip(source_dirs, dataset_prefixes):
        src_path = Path(src_dir)
        if not src_path.exists():
            print(f"[WARNING] Source directory does not exist: {src_path}. Skipping.")
            continue

        print(f"Processing dataset '{prefix}' from: {src_path}")

        for split in splits:
            src_img_dir = src_path / split / "images"
            src_lbl_dir = src_path / split / "labels"

            if not src_img_dir.exists():
                continue

            for img_file in src_img_dir.iterdir():
                if img_file.suffix.lower() not in [".jpg", ".jpeg", ".png", ".bmp"]:
                    continue

                new_img_name = f"{prefix}_{img_file.name}"
                new_lbl_name = f"{prefix}_{img_file.stem}.txt"

                dest_img_path = out_path / split / "images" / new_img_name
                dest_lbl_path = out_path / split / "labels" / new_lbl_name
                src_lbl_path = src_lbl_dir / f"{img_file.stem}.txt"

                shutil.copy(img_file, dest_img_path)
                if src_lbl_path.exists():
                    shutil.copy(src_lbl_path, dest_lbl_path)

    print(f"\n[SUCCESS] Dataset successfully merged into: {output_dir}")

if __name__ == "__main__":
    # Example: Merge Thermal Datasets
    merge_yolo_datasets(
        source_dirs=["path/to/thermal_1", "path/to/thermal_2"],
        output_dir="dataset_merged_thermal",
        dataset_prefixes=["maor", "nguyn"]
    )

    # Example: Merge Optical Datasets
    merge_yolo_datasets(
        source_dirs=["path/to/rgb_1", "path/to/rgb_2"],
        output_dir="dataset_merged_rgb",
        dataset_prefixes=["uav_aug", "drone_uskpc"]
    )
```

---

## 🏋️ Model Training Guide (Kaggle / Google Colab)

Both models were trained with the **Ultralytics YOLO11** framework on cloud GPUs.

```python
from ultralytics import YOLO

# Load base model (YOLO11 Nano for real-time inference)
model = YOLO("yolo11n.pt")

# Train on combined Thermal dataset
model.train(
    data="dataset_merged_thermal/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    name="thermal_drone_model"
)

# Train on combined Optical RGB dataset
model.train(
    data="dataset_merged_rgb/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    device=0,
    name="rgb_drone_model"
)
```

---

## ⚙️ Installation & Usage

1. **Clone the repository**
   ```bash
   git clone https://github.com/FarhanRana234/Aegis-C-UAS.git
   cd Aegis-C-UAS
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place model weights and test videos**
   - `models/thermal_best.pt` – custom trained thermal weights  
   - `models/rgb_best.pt` – optical weights  
   - `IR_videos/ir_test.mp4` – sample IR stream  
   - `RGB_videos/rgb_test1.mp4` – sample RGB stream  

4. **Run the GCS interface**
   ```bash
   python main.py
   ```
   Press `q` in the video window to exit cleanly.

---

## 🗂️ Project Directory Structure

```text
Aegis-C-UAS/
│
├── models/
│   ├── thermal_best.pt      # Custom trained thermal weights
│   └── rgb_best.pt          # Optical weights
│
├── IR_videos/
│   └── ir_test.mp4          # Sample IR test stream
│
├── RGB_videos/
│   └── rgb_test1.mp4        # Sample RGB test stream
│
├── main.py                  # Core GCS control & tracking loop
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## Licence & Disclaimer

This repository is provided for educational and research purposes only.  
It is a software prototype and does not constitute a complete or operational Counter-UAS system.
```
