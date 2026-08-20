#!/usr/bin/env python3
# What this does: prepares grain-level annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw

# Root
ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
OUT = ROOT / "outputs" / "pipeline"
ANNOT = OUT / "annotation"

CROP_DIRS = [OUT / "crops_margin10", OUT / "crops_pad1p0", OUT / "crops"]

# Slides with a single taxa
PURE = {
    "Test Assemblage 1": "Salix",
    "Test Assemblage 2": "Populus",
    "Test Assemblage 7": "Prunus",
    "Test Assemblage 10": "Poaceae",
    "Test Assemblage 12": "Salix",
    "Test Assemblage 13": "Chenopodiaceae-Amaranthaceae type",
}

# Mixed slides
MIXED = {
    "Test Assemblage 3": "1 Populus, 1 Betula",
    "Test Assemblage 4": "1 Larix, 1 Betula, 1 Populus",
    "Test Assemblage 5": "1 Betula, 1 Populus (unstained)",
    "Test Assemblage 6": "2 Populus, 12 Betula",
    "Test Assemblage 8": "7 Picea, 2 Betula, 1 Alnus, 1 Cyperaceae",
    "Test Assemblage 9": "2 Populus, 2 Alnus",
    "Test Assemblage 11": "7 Picea, 1 Alnus (out of focus)",
}

# Contact sheet layout
TILE = 220
COLS = 6
PAD = 8


def find_crop_dir():
    for d in CROP_DIRS:
        if d.exists() and any(d.glob("*.png")):
            return d
    raise SystemExit("no crop directory found - run phase5_3_2_margin_sweep.py first")


def contact_sheet(crops, title, out_path):
    n = len(crops)
    cols = min(COLS, n)
    rows = (n + cols - 1) // cols
    head = 34
    label = 22
    W = cols * (TILE + PAD) + PAD
    H = head + rows * (TILE + label + PAD) + PAD
    sheet = Image.new("RGB", (W, H), "white")
    dr = ImageDraw.Draw(sheet)
    dr.text((PAD, 8), title, fill="black")

    for i, (idx, path) in enumerate(crops):
        r, c = divmod(i, cols)
        x = PAD + c * (TILE + PAD)
        y = head + r * (TILE + label + PAD)
        try:
            im = Image.open(path).convert("RGB").resize((TILE, TILE), Image.LANCZOS)
        except Exception:
            im = Image.new("RGB", (TILE, TILE), "gainsboro")
        sheet.paste(im, (x, y))
        dr.rectangle([x, y, x + TILE - 1, y + TILE - 1], outline="black")
        dr.text((x + 4, y + TILE + 4), f"idx {idx}", fill="black")

    sheet.save(out_path)
    return out_path


def main():
    crop_dir = find_crop_dir()
    ANNOT.mkdir(parents=True, exist_ok=True)
    print(f"crops from: {crop_dir}")

    files = sorted(crop_dir.glob("*.png"))
    if not files:
        raise SystemExit(f"no crops in {crop_dir}")

    rows = []
    for i, f in enumerate(files):
# Filenames are "<sample with underscores>__grainNNN.png"
        stem = f.stem.split("__grain")[0]
        sample = stem.replace("_", " ")
        rows.append({"idx": i, "sample": sample, "crop": f.name,
                     "slide_content": PURE.get(sample, MIXED.get(sample, "")),
                     "true_label": PURE.get(sample, "")})

    df = pd.DataFrame(rows)
    csv_path = ANNOT / "slide_annotations.csv"
    df.to_csv(csv_path, index=False)

    todo = df[df.true_label == ""]
    print(f"\n{len(df)} crops total")
    print(f"  auto-labelled (single-taxon slides): {len(df) - len(todo)}")
    print(f"  needing identification:              {len(todo)}")

    made = []
    for sample, grp in todo.groupby("sample", sort=False):
        content = MIXED.get(sample, "unknown content")
        safe = sample.replace(" ", "_")
        p = contact_sheet([(r.idx, crop_dir / r.crop) for r in grp.itertuples()],
                          f"{sample}  -  identified as: {content}  ({len(grp)} crops detected)",
                          ANNOT / f"sheet_{safe}.png")
        made.append(p)
        print(f"  {sample:>22}: {len(grp):>2} crops -> {p.name}")

    print(f"\nCSV -> {csv_path}")



if __name__ == "__main__":
    main()
