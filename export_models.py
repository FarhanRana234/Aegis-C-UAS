"""Convert the trained .pt weights to ONNX for CPU deployment.

Usage:  python export_models.py            (ONNX, size taken from config.json inference_imgsz)
        python export_models.py 480        (export at 480 px instead, then set inference_imgsz to match)

Needs requirements-export.txt. Copy the resulting models/*.onnx to the Pi / mini PC.
ONNX inputs are fixed-size; main.py reads the size from the model, so a mismatch is handled.
"""
import json
import sys

from ultralytics import YOLO

with open("config.json") as f:
    cfg = json.load(f)
imgsz = int(sys.argv[1]) if len(sys.argv) > 1 else cfg["inference_imgsz"]

for name in ("thermal_best", "rgb_best"):
    out = YOLO(f"models/{name}.pt").export(format="onnx", imgsz=imgsz, simplify=True, dynamic=False)
    print(f"{name}.pt -> {out}  ({imgsz}px)")
