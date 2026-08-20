#!/usr/bin/env python3
# What this does: convert images to standard colour, pad to a square, resize to 224x224.
from pathlib import Path
import re
import numpy as np
import pandas as pd
from PIL import Image

Image.MAX_IMAGE_PIXELS = None

# The Alaska copy of the project.
PROJECT_ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
SPLITS = PROJECT_ROOT / "outputs" / "curation" / "splits"
PROCESSED = PROJECT_ROOT / "data" / "processed_alaska"
INVOUT = PROJECT_ROOT / "outputs" / "curation" / "processed_inventory.csv"
SIZE = 224
PCT = (2, 98)


def safe(s):
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(s)).strip("_")


def load_rgb(path):
    with Image.open(path) as im:
        try:
            im.seek(0)
        except (EOFError, ValueError):
            pass
        mode = im.mode
        if mode in ("I", "I;16", "I;16B", "I;16L", "F"):
            a = np.asarray(im).astype(np.float32)
            lo, hi = np.percentile(a, PCT)
            a = np.clip((a - lo) / max(hi - lo, 1e-6), 0, 1) * 255.0
            arr = np.stack([a.astype(np.uint8)] * 3, axis=-1)
        elif mode == "RGBA":
            bg = Image.new("RGB", im.size, (255, 255, 255))
            bg.paste(im, mask=im.split()[-1])
            arr = np.asarray(bg)
        elif mode == "P":
            arr = np.asarray(im.convert("RGB"))
        elif mode == "L" or mode == "LA":
            g = np.asarray(im.convert("L"))
            arr = np.stack([g] * 3, axis=-1)
        else:
            arr = np.asarray(im.convert("RGB"))
    if arr.dtype != np.uint8:
        lo, hi = np.percentile(arr, PCT)
        arr = (np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1) * 255).astype(np.uint8)
    return arr


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


def process_one(raw_rel, out_path):
    if out_path.exists():
        return "skip"
    arr = load_rgb(RAW_DIR / raw_rel)
    arr = pad_square(arr)
    im = Image.fromarray(arr).resize((SIZE, SIZE), Image.LANCZOS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_path, format="PNG")
    return "ok"


def main():
    if not (SPLITS / "train.csv").exists():
        raise SystemExit(f"No split CSVs in {SPLITS}; run 1.3.4 first.")
    PROCESSED.mkdir(parents=True, exist_ok=True)

    manifest = {}
    for split in ("train", "val", "test"):
        df = pd.read_csv(SPLITS / f"{split}.csv")
        rows, n_ok, n_skip, n_err = [], 0, 0, 0
        for i, r in enumerate(df.itertuples(index=False), 1):
            raw_rel = r.rel_path
            flat = safe(Path(raw_rel).with_suffix("").as_posix().replace("/", "__"))
            out_rel = Path(split) / safe(r.final_label) / f"{flat}.png"
            out_path = PROCESSED / out_rel
            try:
                status = process_one(raw_rel, out_path)
                n_ok += status == "ok"; n_skip += status == "skip"
                rows.append({"processed_path": out_rel.as_posix(),
                             "raw_rel_path": raw_rel, "dataset": r.dataset,
                             "final_label": r.final_label, "label_index": r.label_index})
            except Exception as e:
                n_err += 1
                if n_err <= 10:
                    print(f"  ERR {raw_rel}: {e}")
            if i % 2000 == 0:
                print(f"  {split}: {i}/{len(df)} (ok {n_ok}, skip {n_skip}, err {n_err})")
        out_csv = SPLITS / f"processed_{split}.csv"
        pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8")
        manifest[split] = rows
        print(f"{split}: processed {n_ok}, skipped {n_skip}, errors {n_err} -> {out_csv.name}")
        
    print("\nverifying processed set ...")
    bad, inv_rows = [], []
    for split, rows in manifest.items():
        for rr in rows:
            p = PROCESSED / rr["processed_path"]
            try:
                with Image.open(p) as im:
                    ok = (im.size == (SIZE, SIZE)) and (im.mode == "RGB")
                if not ok:
                    bad.append((rr["processed_path"], im.size, im.mode))
                inv_rows.append({"split": split, "final_label": rr["final_label"],
                                 "processed_path": rr["processed_path"]})
            except Exception as e:
                bad.append((rr["processed_path"], "OPEN_FAIL", str(e)))
    inv = pd.DataFrame(inv_rows)
    inv.to_csv(INVOUT, index=False, encoding="utf-8")

    print(f"  processed images: {len(inv):,}")
    for split in ("train", "val", "test"):
        sub = inv[inv.split == split]
        exp = len(pd.read_csv(SPLITS / f'{split}.csv'))
        print(f"   {split}: {len(sub):,} processed / {exp:,} in split "
              f"({'MATCH' if len(sub)==exp else 'MISMATCH'})")
    print(f"  classes represented: {inv['final_label'].nunique()}")
    print(f"  non-conforming images (not 224x224 RGB): {len(bad)}")
    if bad:
        for b in bad[:10]:
            print("   ", b)
    print(f"\n  written: processed PNGs -> {PROCESSED}")
    print(f"  written: processed_{{train,val,test}}.csv, processed_inventory.csv")
    if not bad:
        print("  All images 224x224 RGB. Phase 1 complete - ready for the Phase 2 loader.")


if __name__ == "__main__":
    main()
