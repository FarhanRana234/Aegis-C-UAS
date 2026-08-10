# Aegis C-UAS (Counter-Unmanned Aerial Systems)

A robust, dual-sensor Ground Control Station (GCS) and computer vision tracking pipeline designed for Counter-UAS applications. Aegis integrates asynchronous **Thermal IR** and **Optical RGB** video feeds, equipped with advanced target tracking, swarm macro-clustering, and environmental clutter filtering.

---

## 🚀 Key Features

* **Dual-Sensor Fusion & Independent Asynchronous Pipeline:** Handles mismatched frame rates (e.g., high-FPS IR streams and standard optical RGB feeds) concurrently without stuttering or locking the main UI thread.
* **Stationary Thermal Clutter Suppression:** Filters out background false positives caused by *thermal inertia* (e.g., sun-baked rocks, asphalt, or rooftops that retain heat at night) by analyzing pixel displacement across frames.
* **Swarm Macro-Clustering (DBSCAN):** Utilizes Density-Based Spatial Clustering to group dense swarms of drones into a unified macro-bounding box while maintaining individual tracking coordinates.
* **Sticky Single-Target Tracking:** Implements spatial nearest-neighbor matching rather than volatile confidence scoring to prevent target-switching jitter when multiple unclustered drones are present.
* **Coasting Grace Period:** Maintains a target lock for a configurable grace period (up to 15 frames) if temporary occlusion or detection dropouts occur.

---

## 🔗 Roboflow Dataset Sources

The custom models were trained on multi-source datasets curated via Roboflow and exported in **YOLOv11** format. You can access the public datasets here:

* **Thermal Dataset 1:** [Maor Ovadia - Thermal Drone (v11)](https://www.google.com/search?q=https://universe.roboflow.com/maor-ovadia/thermal-drone-47qfh/dataset/11)
* **Thermal Dataset 2:** [Nguyn Quyt - Thermal Drone (v2)](https://www.google.com/search?q=https://universe.roboflow.com/nguyn-quyt/thermal-drone/dataset/2)
* **Optical Dataset 1:** [UAV Swarm Augmentation (v1)](https://www.google.com/search?q=https://universe.roboflow.com/project-8cfvz/uavswarmv_augmentation/dataset/1)
* **Optical Dataset 2:** [Drone Detection (v1)](https://www.google.com/search?q=https://universe.roboflow.com/project-986i8/drone-uskpc/dataset/1)

### Downloading via Python Snippets

```python
!pip install roboflow
from roboflow import Roboflow

# Thermal Dataset 1
rf = Roboflow(api_key="API_key_here")
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

When downloading multiple datasets from Roboflow in YOLO format, file names often overlap (e.g., `image_001.jpg` across folders). Use this script to merge them into a single unified directory with automatic prefixing:

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

Both models were trained using the **Ultralytics YOLO11** framework on cloud GPUs.

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

1. **Clone the Repository:**
```bash
git clone https://github.com/your-username/aegis-cuas-hpm.git
cd aegis-cuas-hpm

```


2. **Install Dependencies:**
```bash
pip install ultralytics scikit-learn opencv-python numpy roboflow

```


3. **Configure Project Files:**
* Place your trained thermal weights in `models/thermal_best.pt`.
* Place sample video files in `IR_videos/ir_test.mp4` and `RGB_videos/rgb_test1.mp4`.


4. **Run the GCS Command & Control Interface:**
```bash
python main.py

```


* *Press `q` within the video display window to terminate execution safely.*



---

## 🗂️ Project Directory Structure

```text
aegis-cuas/
│
├── models/
│   ├── thermal_best.pt      # Custom trained thermal weights
│   └── rgb_best.pt           # Base/fine-tuned optical weights
│
├── IR_videos/
│   └── ir_test.mp4          # Sample IR test stream
│
├── RGB_videos/
│   └── rgb_test1.mp4        # Sample RGB test stream
│
├── main.py                  # Core GCS control & tracking loop
└── README.md                # Project documentation

```

