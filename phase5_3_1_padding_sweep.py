#!/usr/bin/env python3
# What this does: tests whether the classifier's poor results on real slide crops is caused by how tightly the grains are cropped.


import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import timm

# Alaska root
ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
sys.path.insert(0, str(ROOT / "alaska_scripts"))
import phase0_determinism_Alaska as det
import phase2_1_1_dataset_Alaska as ds
from phase2_1_2_transforms_Alaska import get_val_transform

SLIDES = ROOT / "data" / "fairbanks" / "images"
OUT = ROOT / "outputs" / "pipeline"
TEMP_JSON = ROOT / "outputs" / "eval" / "temperature.json"
BASELINE = OUT / "grain_predictions.csv"

# Some settings
TAU = 0.56
SIZE = 224
DET_CONF = 0.25
PADS = [1.0, 1.25, 1.5, 2.0]

SAVE_CROPS_FOR = 1.5


# Copy preprocessing
def pad_square(arr):
    h, w = arr.shape[:2]
    if h == w:
        return arr
    side = max(h, w)
    fill = np.median(arr.reshape(-1, arr.shape[2]), axis=0).astype(arr.dtype)
    out = np.empty((side, side, arr.shape[2]), dtype=arr.dtype)
    out[:, :] = fill
    y0 = (side - h) // 2
    x0 = (side - w) // 2
    out[y0:y0 + h, x0:x0 + w] = arr
    return out


# Widen the detection box
def expand_box(box_xyxy, factor, img_w, img_h):
    x0, y0, x1, y1 = box_xyxy
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0) / 2.0 * factor, (y1 - y0) / 2.0 * factor
    nx0 = int(round(max(0.0, cx - hw)))
    ny0 = int(round(max(0.0, cy - hh)))
    nx1 = int(round(min(float(img_w), cx + hw)))
    ny1 = int(round(min(float(img_h), cy + hh)))
    if nx1 - nx0 < 2 or ny1 - ny0 < 2:
        return None
    return (nx0, ny0, nx1, ny1)


# Pad and resize
def crop_to_224(pil_img, box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    crop = pil_img.convert("RGB").crop((x0, y0, x1, y1))
    arr = pad_square(np.asarray(crop))
    return Image.fromarray(arr).resize((SIZE, SIZE), Image.LANCZOS)


# Load Alaska classifier from chkpnt
def load_model(device, n_classes):
    ck = torch.load(ROOT / "outputs" / "checkpoints" / "best.pt", map_location=device,
                    weights_only=False)
    arch = ck.get("arch", "convnext_tiny")
    model = timm.create_model(arch, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


def find_detector():
    for name in ("grain_fairbanks_ft", "grain_yolov8n_ft"):
        w = ROOT / "outputs" / "detector" / name / "weights" / "best.pt"
        if w.exists():
            return w
    raise SystemExit("fine-tuned detector not found under outputs/detector/")


def main():
    det.enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    idx = ds.load_class_index()
    inv = {v: k for k, v in idx.items()}
    model = load_model(device, len(idx))

    T = 1.0
    if TEMP_JSON.exists():
        T = json.loads(TEMP_JSON.read_text()).get("temperature", 1.0)
    print(f"classes: {len(idx)} | tau: {TAU} | temperature: {T:.3f} | device: {device}")

    tf = get_val_transform(SIZE)

    from ultralytics import YOLO
    weights = find_detector()
    yolo = YOLO(str(weights))
    print(f"detector: {weights}")

    crop_dir = OUT / f"crops_pad{SAVE_CROPS_FOR}".replace(".", "p")
    crop_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(SLIDES.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no slides found in {SLIDES}")

# Loop over boxes
    rows = []
    with torch.no_grad():
        for img_path in images:
            sample = img_path.stem.split("_jpg")[0]
            res = yolo.predict(str(img_path), imgsz=640, conf=DET_CONF, verbose=False)[0]
            pil = Image.open(img_path)
            boxes = res.boxes.xyxy.cpu().numpy()
            dconf = res.boxes.conf.cpu().numpy()

            for i, box in enumerate(boxes):
                name = f"{sample}__grain{i:03d}.png".replace(" ", "_")
                for pad in PADS:
                    eb = expand_box(box, pad, pil.width, pil.height)
                    if eb is None:
                        continue
                    want_w = (box[2] - box[0]) * pad
                    clipped = (eb[2] - eb[0]) < want_w - 1
                    crop = crop_to_224(pil, eb)
                    if pad == SAVE_CROPS_FOR:
                        crop.save(crop_dir / name)
                    x = tf(image=np.asarray(crop))["image"].unsqueeze(0).to(device)
                    prob = torch.softmax(model(x) / T, 1)[0]
                    conf, ci = float(prob.max()), int(prob.argmax())
                    rows.append({
                        "sample": sample, "crop": name, "pad": pad,
                        "pred_label": inv[ci], "confidence": round(conf, 4),
                        "indeterminate": conf < TAU,
                        "detector_conf": round(float(dconf[i]), 4),
                        "box_w": int(box[2] - box[0]), "box_h": int(box[3] - box[1]),
                        "clipped": bool(clipped),
                    })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "padding_sweep_predictions.csv", index=False)
    print(f"\nwrote {len(sweep)} rows -> padding_sweep_predictions.csv")
    print(f"crops at pad {SAVE_CROPS_FOR} -> {crop_dir}")

# Selfcheck
    if BASELINE.exists():
        base = pd.read_csv(BASELINE)[["sample", "crop", "pred_label"]]
        p1 = sweep[sweep["pad"] == 1.0][["sample", "crop", "pred_label"]]
        m = base.merge(p1, on=["sample", "crop"], suffixes=("_base", "_new"))
        agree = (m["pred_label_base"] == m["pred_label_new"]).sum()
        print(f"\nself-check vs grain_predictions.csv: {agree}/{len(m)} identical at pad 1.0")
        if len(m) != len(base):
            print(f"  WARNING: matched {len(m)} of {len(base)} baseline rows - detection differs")
        if agree != len(m):
            print("  WARNING: pad 1.0 did not reproduce the original run.")
    else:
        print("\n(no grain_predictions.csv found - self-check skipped)")

# Changes
    print("\npredicted taxon counts by padding factor:")
    piv = (sweep.pivot_table(index="pred_label", columns="pad", values="crop",
                             aggfunc="count", fill_value=0)
                .sort_values(PADS[0], ascending=False))
    print(piv.to_string())

# Below cut-off
    print("\nindeterminate rate by padding factor:")
    for pad in PADS:
        s = sweep[sweep["pad"] == pad]
        if len(s):
            print(f"  pad {pad:>4}: {s.indeterminate.sum():>3}/{len(s):>3} "
                  f"({100 * s.indeterminate.mean():.1f}%)  "
                  f"median conf {s.confidence.median():.3f}  "
                  f"clipped by slide edge: {int(s.clipped.sum())}")

    print("\nNEXT: score padding_sweep_predictions.csv against slides.")


if __name__ == "__main__":
    main()
