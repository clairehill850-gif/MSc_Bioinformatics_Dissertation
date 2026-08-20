#!/usr/bin/env python3
# What this does: converts the POLLEN20L-det download

import csv
import os
import random
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test/data/pollen20l")
SRC_IMAGES = None
SRC_BBOXES = None
OUT = ROOT / "yolo"

# Validation images
VAL_FRAC = 0.15
SEED = 42


# Images and box list
def resolve_sources():
    global SRC_IMAGES, SRC_BBOXES
    bb = list(ROOT.rglob("bboxes.csv"))
    if not bb:
        raise SystemExit(f"bboxes.csv not found under {ROOT}")
    SRC_BBOXES = bb[0]
    imgs = SRC_BBOXES.parent / "images"
    if not imgs.is_dir():
        cand = list(ROOT.rglob("images"))
        if not cand:
            raise SystemExit("images/ dir not found")
        imgs = cand[0]
    SRC_IMAGES = imgs
    print(f"bboxes: {SRC_BBOXES}")
    print(f"images: {SRC_IMAGES}")

def read_boxes():
    boxes = defaultdict(list)
    with open(SRC_BBOXES) as f:
        for row in csv.reader(f):
            if len(row) < 5:
                continue
            fn = row[0]
            try:
                x0, y0, x1, y1 = (int(float(v)) for v in row[1:5])
            except ValueError:
                continue
            boxes[fn].append((x0, y0, x1, y1))
    return boxes


# Convert box into detector format
def to_yolo(x0, y0, x1, y1, W, H):
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    cx = (x0 + x1) / 2.0 / W
    cy = (y0 + y1) / 2.0 / H
    w = (x1 - x0) / W
    h = (y1 - y0) / H
    clamp = lambda v: max(0.0, min(1.0, v))
    return clamp(cx), clamp(cy), clamp(w), clamp(h)


# Convert images
def main():
    resolve_sources()
    boxes = read_boxes()
    print(f"images with boxes: {len(boxes)}")

# Shuffle with fixed seed/take validation
    files = sorted(boxes.keys())
    random.Random(SEED).shuffle(files)
    n_val = int(len(files) * VAL_FRAC)
    val_set = set(files[:n_val])
    print(f"split: {len(files) - n_val} train / {n_val} val")

# Folders
    for split in ("train", "val"):
        (OUT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_boxes, n_skipped, degenerate = 0, 0, 0
    for fn in files:
        src_img = SRC_IMAGES / fn
        if not src_img.exists():
            n_skipped += 1
            continue
        split = "val" if fn in val_set else "train"
        with Image.open(src_img) as im:
            W, H = im.size

        lines = []
        for (x0, y0, x1, y1) in boxes[fn]:
            cx, cy, w, h = to_yolo(x0, y0, x1, y1, W, H)
            if w <= 0 or h <= 0:
                degenerate += 1
                continue
            lines.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            n_boxes += 1

        shutil.copy2(src_img, OUT / "images" / split / fn)
        label_name = Path(fn).with_suffix(".txt").name
        (OUT / "labels" / split / label_name).write_text("\n".join(lines))

# Description
    (OUT / "data.yaml").write_text(
        f"path: {OUT}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names: ['grain']\n"
    )

# Report
    print(f"\nboxes written: {n_boxes:,}")
    if degenerate:
        print(f"  degenerate boxes skipped (zero w/h): {degenerate}")
    if n_skipped:
        print(f"  images in csv but missing on disk: {n_skipped}")
    print(f"data.yaml -> {OUT / 'data.yaml'}")
    print(f"\nNEXT: train with phase5_1_2_train_yolo.py")


if __name__ == "__main__":
    main()
