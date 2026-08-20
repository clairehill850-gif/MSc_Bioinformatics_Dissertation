#!/usr/bin/env python3
# What this does: Detector to classifier 

import argparse
import os
from pathlib import Path

import numpy as np
from PIL import Image

# Alaska root
ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
SIZE = 224


def pad_square(arr):
    h, w = arr.shape[:2]
    if h == w:
        return arr
    s = max(h, w)
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]], axis=0)
    fill = np.median(border, axis=0).astype(np.uint8)
    canvas = np.full((s, s, 3), fill, np.uint8)
    y, x = (s - h) // 2, (s - w) // 2
    canvas[y:y + h, x:x + w] = arr
    return canvas


# Cut and resize
def crop_to_224(pil_img, box_xyxy):
    x0, y0, x1, y1 = (int(round(v)) for v in box_xyxy)
    x0, x1 = max(0, x0), min(pil_img.width, x1)
    y0, y1 = max(0, y0), min(pil_img.height, y1)
    if x1 <= x0 or y1 <= y0:
        return None
    crop = pil_img.convert("RGB").crop((x0, y0, x1, y1))
    arr = pad_square(np.asarray(crop))
    return Image.fromarray(arr).resize((SIZE, SIZE), Image.LANCZOS)


# Check
def validate():
    import sys
    sys.path.insert(0, str(ROOT / "alaska_scripts"))
    import torch
    import pandas as pd
    import phase2_1_1_dataset_Alaska as ds
    from phase2_1_2_transforms_Alaska import get_val_transform
    import timm

# Load classifier
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = ds.load_class_index()
    inv = {v: k for k, v in idx.items()}

    ck = torch.load(ROOT / "outputs" / "checkpoints" / "best.pt", map_location=device)
    arch = ck.get("arch", "convnext_tiny")
    model = timm.create_model(arch, pretrained=False, num_classes=len(idx))
    model.load_state_dict(ck["model_state"]); model.to(device).eval()

# Test
    tf = get_val_transform(SIZE)
    test = pd.read_csv(ds.SPLITS / "processed_test.csv")
    def resolve(pp):
        a = ds.PROCESSED_ALASKA / pp if hasattr(ds, "PROCESSED_ALASKA") else None
        if a and a.exists():
            return a
        return ds.PROCESSED_ROOT / pp

    correct = n = 0
    with torch.no_grad():
        for r in test.itertuples():
            img = Image.open(resolve(r.processed_path)).convert("RGB")
            crop = crop_to_224(img, (0, 0, img.width, img.height))
            x = tf(image=np.asarray(crop))["image"].unsqueeze(0).to(device)
            pred = inv[int(model(x).argmax(1))]
            correct += (pred == r.final_label); n += 1
            if n % 500 == 0:
                print(f"  {n} done, running acc {correct/n:.4f}")
    acc = correct / n
    print(f"\nBRIDGE VALIDATION: {acc:.4f} over {n} images")
    print(f"  reference (normal load): ~0.9043")
    if abs(acc - 0.9043) < 0.02:
        print("  PASS - bridge reproduces training accuracy. Crop convention works.")
    else:
        print("  MISMATCH - bridge differs from training preprocessing.")
        print("  (Check pad fill, resize filter, or colour handling.)")


# Crop
def crop_fairbanks(conf):
    from ultralytics import YOLO
    weights = ROOT / "outputs" / "detector" / "grain_fairbanks_ft" / "weights" / "best.pt"
    if not weights.exists():
        alt = ROOT / "outputs" / "detector" / "grain_yolov8n_ft" / "weights" / "best.pt"
        weights = alt if alt.exists() else weights
    if not weights.exists():
        raise SystemExit(f"fine-tuned detector not found at {weights}")

# Slides and crops
    images = sorted((ROOT / "data" / "fairbanks" / "images").glob("*.jpg"))
    out = ROOT / "outputs" / "pipeline" / "crops"
    out.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(weights))
    manifest = []
# Cut out boxes
    for img_path in images:
        sample = img_path.stem.split("_jpg")[0]
        res = model.predict(str(img_path), imgsz=640, conf=conf, verbose=False)[0]
        pil = Image.open(img_path)
        for i, box in enumerate(res.boxes.xyxy.cpu().numpy()):
            crop = crop_to_224(pil, box)
            if crop is None:
                continue
            fn = f"{sample}__grain{i:03d}.png".replace(" ", "_")
            crop.save(out / fn)
            manifest.append({"sample": sample, "crop": fn,
                             "conf": float(res.boxes.conf[i])})
# Save each crop
    import pandas as pd
    pd.DataFrame(manifest).to_csv(ROOT / "outputs" / "pipeline" / "crop_manifest.csv", index=False)
    print(f"cropped {len(manifest)} grains from {len(images)} slides -> {out}")
    print(f"manifest -> {ROOT / 'outputs' / 'pipeline' / 'crop_manifest.csv'}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate", action="store_true",
                    help="re-crop the frozen test set and confirm ~0.904 before real slides")
    ap.add_argument("--conf", type=float, default=0.25, help="detector confidence for crop mode")
    args = ap.parse_args()
    if args.validate:
        validate()
    else:
        crop_fairbanks(args.conf)


if __name__ == "__main__":
    main()
