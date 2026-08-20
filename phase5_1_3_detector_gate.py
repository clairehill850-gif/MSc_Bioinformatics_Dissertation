#!/usr/bin/env python3
# What this does: runs the trained grain detector over the Fairbanks slides and scores it against manually drawn boxes

import os
import shutil
from pathlib import Path

ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
WEIGHTS = ROOT / "outputs" / "detector" / "grain_yolov8n" / "weights" / "best.pt"
FB_IMAGES = ROOT / "data" / "fairbanks" / "images"
FB_LABELS = ROOT / "data" / "fairbanks" / "labels"
GATE_DIR = ROOT / "outputs" / "detector" / "gate"
IMGSZ = 640


# Description
def build_eval_yaml():
    GATE_DIR.mkdir(parents=True, exist_ok=True)
    y = GATE_DIR / "fairbanks_eval.yaml"
    y.write_text(
        f"path: {ROOT / 'data' / 'fairbanks'}\n"
        f"train: images\n"
        f"val: images\n"
        f"nc: 1\n"
        f"names: ['grain']\n"
    )
    return y


# Run gate
def main():
# Stop early
    if not WEIGHTS.exists():
        raise SystemExit(f"detector weights not found: {WEIGHTS}")
    n_img = len(list(FB_IMAGES.glob("*.jpg")))
    n_lab = len(list(FB_LABELS.glob("*.txt")))
    print(f"Fairbanks slides: {n_img} images, {n_lab} label files")
    if n_img == 0 or n_lab == 0:
        raise SystemExit("images or labels missing - check data/fairbanks/{images,labels}")

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

# Score detector against manual boxes
    yaml = build_eval_yaml()
    print("\n=== GATE: detector vs hand-boxes on 14 Fairbanks slides ===")
    metrics = model.val(data=str(yaml), imgsz=IMGSZ, project=str(GATE_DIR),
                        name="fairbanks_gate", exist_ok=True, verbose=True)

    map50 = metrics.box.map50
    print(f"\n  mAP@0.5:      {map50:.4f}")
    print(f"  mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"  precision:    {metrics.box.mp:.4f}")
    print(f"  recall:       {metrics.box.mr:.4f}")

# Print scoring
    print("\n=== GATE DECISION ===")
    if map50 >= 0.70:
        print(f"  PASS (mAP@0.5 = {map50:.3f} >= 0.70)")
        print("  Detector transfers. No fine-tuning. All 14 samples remain blind test.")
    else:
        print(f"  BELOW THRESHOLD (mAP@0.5 = {map50:.3f} < 0.70)")
        print("  Domain gap too large. Fine-tune on a few slides")
        print("  (those slides then leave the blind test set).")

# Visual check
    print("\nsaving annotated predictions for visual check ...")
    model.predict(source=str(FB_IMAGES), imgsz=IMGSZ, conf=0.25, save=True,
                  project=str(GATE_DIR), name="fairbanks_preds", exist_ok=True)
    print(f"  annotated images -> {GATE_DIR / 'fairbanks_preds'}")


if __name__ == "__main__":
    main()
