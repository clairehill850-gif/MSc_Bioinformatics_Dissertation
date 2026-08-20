#!/usr/bin/env python3
# What this does: tests if the detector box is cutting the marginal features off the grain

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

# The confidence cut-off
TAU = 0.55
SIZE = 224
DET_CONF = 0.25

# Margins
MARGINS = [0, 5, 10, 15, 20, 30]
SAVE_CROPS_FOR = 10


# Padding
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


# Add margin
def add_margin(box_xyxy, px, img_w, img_h):
    x0, y0, x1, y1 = box_xyxy
    nx0 = int(round(max(0.0, x0 - px)))
    ny0 = int(round(max(0.0, y0 - px)))
    nx1 = int(round(min(float(img_w), x1 + px)))
    ny1 = int(round(min(float(img_h), y1 + px)))
    if nx1 - nx0 < 2 or ny1 - ny0 < 2:
        return None
    return (nx0, ny0, nx1, ny1)


# Pad and resize
def crop_to_224(pil_img, box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    crop = pil_img.convert("RGB").crop((x0, y0, x1, y1))
    arr = pad_square(np.asarray(crop))
    return Image.fromarray(arr).resize((SIZE, SIZE), Image.LANCZOS)


# Load Alaska classifier from chckpnt
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
    print(f"detector: {weights} | det conf: {DET_CONF}")

# Folders
    crop_dir = OUT / f"crops_margin{SAVE_CROPS_FOR}"
    crop_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(SLIDES.glob("*.jpg"))
    if not images:
        raise SystemExit(f"no slides found in {SLIDES}")

# Loop
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
                bw, bh = int(box[2] - box[0]), int(box[3] - box[1])
                for px in MARGINS:
                    eb = add_margin(box, px, pil.width, pil.height)
                    if eb is None:
                        continue
                    clipped = (eb[2] - eb[0]) < bw + 2 * px - 1
                    crop = crop_to_224(pil, eb)
# Visual inspection
                    if px == SAVE_CROPS_FOR:
                        crop.save(crop_dir / name)
                    x = tf(image=np.asarray(crop))["image"].unsqueeze(0).to(device)
                    prob = torch.softmax(model(x) / T, 1)[0]
                    conf, ci = float(prob.max()), int(prob.argmax())
                    rows.append({
                        "sample": sample, "crop": name, "margin_px": px,
                        "pred_label": inv[ci], "confidence": round(conf, 4),
                        "indeterminate": conf < TAU,
                        "detector_conf": round(float(dconf[i]), 4),
                        "box_w": bw, "box_h": bh,
                        "clipped": bool(clipped),
                    })

    sweep = pd.DataFrame(rows)
    sweep.to_csv(OUT / "margin_sweep_predictions.csv", index=False)
    print(f"\nwrote {len(sweep)} rows -> margin_sweep_predictions.csv")
    print(f"crops at margin {SAVE_CROPS_FOR}px -> {crop_dir}")

# Self check
    if BASELINE.exists():
        base = pd.read_csv(BASELINE)[["sample", "crop", "pred_label"]]
        m0 = sweep[sweep["margin_px"] == 0][["sample", "crop", "pred_label"]]
        m = base.merge(m0, on=["sample", "crop"], suffixes=("_base", "_new"))
        agree = (m["pred_label_base"] == m["pred_label_new"]).sum()
        print(f"\nSELF-CHECK vs grain_predictions.csv: {agree}/{len(m)} identical at margin 0")
        if len(m) != len(base):
            print(f"  WARNING: matched {len(m)} of {len(base)} baseline rows - detection differs")
        if agree != len(m):
            print("  WARNING: margin 0 did not reproduce the original run. Most likely DET_CONF")
            print(f"  ({DET_CONF}) differs from the value the bridge was originally run at,")
            print("  which changes detection order and therefore what each grainNNN refers to.")
    else:
        print("\n(no grain_predictions.csv found - self-check skipped)")

    b = sweep[sweep["margin_px"] == 0]
    print(f"\ndetected grain size (px): median {b.box_w.median():.0f} x {b.box_h.median():.0f}"
          f" | min {b.box_w.min()} | max {b.box_w.max()}")
    print(f"grains under 100 px wide: {(b.box_w < 100).sum()} of {len(b)}")
    med = b.box_w.median()
    print("\nmargin as a share of the median grain, and grain share of the 224px frame:")
    for px in MARGINS:
        frac = (med / (med + 2 * px)) ** 2
        print(f"  +{px:>2}px per edge: box {med + 2 * px:>5.0f}px wide, "
              f"grain fills {100 * frac:>5.1f}% of frame area")

# Changes
    print("\npredicted taxon counts by margin:")
    piv = (sweep.pivot_table(index="pred_label", columns="margin_px", values="crop",
                             aggfunc="count", fill_value=0)
                .sort_values(MARGINS[0], ascending=False))
    print(piv.to_string())

# Below cut-off
    print("\nindeterminate rate by margin:")
    for px in MARGINS:
        s = sweep[sweep["margin_px"] == px]
        if len(s):
            print(f"  +{px:>2}px: {s.indeterminate.sum():>3}/{len(s):>3} "
                  f"({100 * s.indeterminate.mean():.1f}%)  "
                  f"median conf {s.confidence.median():.3f}  "
                  f"clipped by slide edge: {int(s.clipped.sum())}")

    print("\nNEXT: score margin_sweep_predictions.csv against slides.")


if __name__ == "__main__":
    main()
