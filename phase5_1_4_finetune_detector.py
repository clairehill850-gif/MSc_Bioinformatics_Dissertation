#!/usr/bin/env python3
# What this does: fine-tunes the grain detector on the Fairbanks slides

import random
import shutil
from pathlib import Path

ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
BASE_WEIGHTS = ROOT / "outputs" / "detector" / "grain_yolov8n" / "weights" / "best.pt"
SRC_IMAGES = ROOT / "data" / "fairbanks" / "images"
SRC_LABELS = ROOT / "data" / "fairbanks" / "labels"
FT_DIR = ROOT / "data" / "fairbanks_ft"
OUT = ROOT / "outputs" / "detector"

# Validation and seed
N_VAL = 3
SEED = 42


# No. of boxes
def box_count(label_path):
    if not label_path.exists():
        return 0
    return sum(1 for ln in label_path.read_text().splitlines() if ln.strip())

def split_slides(images):
    counted = sorted(((box_count(SRC_LABELS / (im.stem + ".txt")), im) for im in images),
                     key=lambda t: t[0])
    n = len(counted)
    val_idx = {n // 4, n // 2, (3 * n) // 4}
    val = [counted[i][1] for i in sorted(val_idx)]
    train = [im for _, im in counted if im not in val]
    return train, val


# Finetune to folder
def stage(images, split):
    (FT_DIR / "images" / split).mkdir(parents=True, exist_ok=True)
    (FT_DIR / "labels" / split).mkdir(parents=True, exist_ok=True)
    boxes = 0
    for im in images:
        shutil.copy2(im, FT_DIR / "images" / split / im.name)
        lab = SRC_LABELS / (im.stem + ".txt")
        dst = FT_DIR / "labels" / split / (im.stem + ".txt")
        dst.write_text(lab.read_text() if lab.exists() else "")
        boxes += box_count(lab)
    return boxes


# Finetune
def main():
# Stop early
    if not BASE_WEIGHTS.exists():
        raise SystemExit(f"base weights not found: {BASE_WEIGHTS}")
    images = sorted(SRC_IMAGES.glob("*.jpg"))
    if len(images) < 8:
        raise SystemExit(f"only {len(images)} images found; expected 14")

# Rebuild
    if FT_DIR.exists():
        shutil.rmtree(FT_DIR)
    train, val = split_slides(images)
    nb_tr = stage(train, "train")
    nb_va = stage(val, "val")
    print(f"fine-tune split: {len(train)} train ({nb_tr} boxes) / {len(val)} val ({nb_va} boxes)")

# Description
    (FT_DIR / "data.yaml").write_text(
        f"path: {FT_DIR}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['grain']\n")

    from ultralytics import YOLO
# Start from reference
    model = YOLO(str(BASE_WEIGHTS))

    model.train(
        data=str(FT_DIR / "data.yaml"),
        epochs=80,
        patience=25,
        imgsz=640,
        batch=8,
        lr0=0.001,
        freeze=10,
        project=str(OUT),
        name="grain_yolov8n_ft",
        exist_ok=True,
        seed=SEED,
        degrees=20.0, fliplr=0.5, flipud=0.5, scale=0.5,
        hsv_v=0.5, hsv_s=0.5, mosaic=0.3,
        verbose=True,
    )

# Score detector
    metrics = model.val(data=str(FT_DIR / "data.yaml"), imgsz=640)
    print(f"\nfine-tuned val mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"fine-tuned val mAP@0.5:0.95: {metrics.box.map:.4f}")

# Debris
    model.predict(source=str(SRC_IMAGES), imgsz=640, conf=0.25, save=True,
                  project=str(OUT), name="fairbanks_ft_preds", exist_ok=True)
    print(f"\nannotated predictions -> {OUT / 'fairbanks_ft_preds'}")
    print("Fewer bubbles/debris boxed? more real grains caught?")
    print(f"best weights -> {OUT / 'grain_yolov8n_ft' / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
