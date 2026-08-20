#!/usr/bin/env python3
# What this does: trains the YOLOv8 grain detector on the converted POLLEN20L-det images.

import os
from pathlib import Path

from ultralytics import YOLO


DATA = Path("/workspace/datasets/pollen_bundle/Alaska_Test/data/pollen20l/yolo/data.yaml")
OUT = Path("/workspace/datasets/pollen_bundle/Alaska_Test/outputs/detector")

# YOLOv8
MODEL = "yolov8n.pt"
EPOCHS = 100
# Image res
IMGSZ = 416
BATCH = 16
# EArly stopping
PATIENCE = 20


# Train and report
def main():
    OUT.mkdir(parents=True, exist_ok=True)
# Fail early
    if not DATA.exists():
        raise SystemExit(f"data.yaml not found: {DATA}\nRun phase5_1_1_prepare_detection_data.py first.")

    model = YOLO(MODEL)
    model.train(
        data=str(DATA),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        project=str(OUT),
        name="grain_yolov8n",
        exist_ok=True,
        seed=42,
# Augmentation
        degrees=15.0,
        fliplr=0.5,
        flipud=0.5,
        mosaic=0.5,
        verbose=True,
    )

# Score detector and report
    metrics = model.val(data=str(DATA))
    print(f"\nval mAP@0.5:     {metrics.box.map50:.4f}")
    print(f"val mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"\nbest weights -> {OUT / 'grain_yolov8n' / 'weights' / 'best.pt'}")
    print("NEXT: annotate 2 real Fairbanks slides cold, then run the mAP gate.")


if __name__ == "__main__":
    main()
