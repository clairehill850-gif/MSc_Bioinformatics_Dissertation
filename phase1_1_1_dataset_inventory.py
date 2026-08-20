#!/usr/bin/env python3
# What this does: builds inventory images. For each image it works out the plant name, records details (size, format etc) and a blur score.

from pathlib import Path
from collections import Counter
import csv
import re
import sys
import time

from PIL import Image
    
# Optional extras
try:
    import numpy as np
    HAVE_NUMPY = True
except Exception:
    HAVE_NUMPY = False

_BLUR_BACKEND = None
try:
    import cv2
    _BLUR_BACKEND = "cv2"
except Exception:
    try:
        from scipy.ndimage import laplace as _scipy_laplace
        _BLUR_BACKEND = "scipy"
    except Exception:
        _BLUR_BACKEND = None

try:
    import tifffile
    HAVE_TIFFFILE = True
except Exception:
    HAVE_TIFFFILE = False

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "inventory"
OUT_CSV = OUT_DIR / "master_inventory.csv"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

DATASET_EXCLUDE = {
    "Tananaki": "multi-grain field images — incompatible with single-grain classifier",
}
EXCLUDED_SUFFIX = "_excluded"

# Set to False to skip the blur check
COMPUTE_BLUR = True  
# Save progress to disk        
FLUSH_EVERY = 500             
PROGRESS_EVERY = 1000

# Source folders, not actual plant names
WRAPPER_TOKENS = {
    "gpp_images", "paldat_images", "pollenwiki_images", "t-dataset",
    "images_16_types", "pollen73s", "pollen23e", "unknown", "final_train",
}


# Text helpers
_NUMDOT = re.compile(r"^\s*\d+\.\s*")

def first_ws(s: str) -> str:
    s = s.strip()
    return s.split()[0] if s.split() else ""

def first_us(s: str) -> str:
    s = s.strip()
    return s.split("_")[0] if s else ""

def strip_numdot(s: str) -> str:
    return _NUMDOT.sub("", s).strip()

def norm_genus(tok: str) -> str:
    tok = tok.strip().strip(".,_-()[]").strip()
    return tok[:1].upper() + tok[1:].lower() if tok else ""


# Label readers, one per dataset because each dataset stores its plant names differently

def _anchor_index(parts, name):
# Find where a known anchor folder sits in the path
    low = [p.lower() for p in parts]
    return low.index(name.lower()) if name.lower() in low else None

# Cambridge
def parse_cambridge(parts):                      
    stem = Path(parts[-1]).stem
    return first_ws(stem), stem, None, None, ("" if len(parts) == 1 else "unexpected_depth")

# Cropped
def parse_cropped(parts):                        
    if len(parts) >= 2:
        folder = strip_numdot(parts[-2])
        return first_ws(folder), folder, None, None, ""
    stem = Path(parts[-1]).stem
    return first_ws(strip_numdot(stem)), stem, None, None, "flat_fallback"

# GPP
def parse_gpp(parts):                            
    i = _anchor_index(parts, "gpp_images")
    if i is not None and len(parts) >= i + 4:
        family = parts[i + 1]
        genus = parts[i + 2]
        species = parts[i + 3]
        return genus, f"{genus} {species}", family, species, ""
# Fallback: use the grandparent folder
    if len(parts) >= 3:
        return parts[-3], parts[-3], (parts[-4] if len(parts) >= 4 else None), parts[-2], "anchor_fallback"
    return Path(parts[-1]).stem, parts[-1], None, None, "unparsed"

# 16 Images
def parse_images16(parts):                       
    if len(parts) >= 2:
        return parts[-2], parts[-2], None, None, ""
    return Path(parts[-1]).stem, parts[-1], None, None, "flat_fallback"

# Oreme
def parse_oreme(parts):                          
    if len(parts) >= 2:
        folder = parts[-2]
        toks = folder.split()
        species = toks[1] if len(toks) > 1 else None
        return first_ws(folder), folder, None, species, ""
    stem = Path(parts[-1]).stem
    return first_ws(stem), stem, None, None, "flat_fallback"

# Other
def parse_other(parts):                          
    if len(parts) == 1:                      
        stem = Path(parts[-1]).stem
        return first_ws(stem), stem, None, None, "flat_in_other"
    folder = parts[-2]
    toks = folder.split()
    species = toks[1] if len(toks) > 1 else None
    return first_ws(folder), folder, None, species, ""

# PalDat
def parse_paldat(parts):                         
    i = _anchor_index(parts, "paldat_images")
    if i is not None and len(parts) >= i + 3:
        genus = parts[i + 1]
        species_folder = parts[i + 2]
        return genus, species_folder, None, species_folder, ""
# Fallback: use the grandparent folder
    if len(parts) >= 3:                          
        return parts[-3], parts[-2], None, parts[-2], "anchor_fallback"
    return Path(parts[-1]).stem, parts[-1], None, None, "unparsed"

# Kaggle
def parse_kaggle(parts):                         
    if len(parts) >= 2:
        folder = parts[-2]
        return first_us(folder), folder, None, folder, ""
    return Path(parts[-1]).stem, parts[-1], None, None, "flat_fallback"

# PollenWiki
def parse_pollenwiki(parts):                     
    if len(parts) >= 2:
        folder = parts[-2]
        toks = folder.split()
        species = toks[1] if len(toks) > 1 else None
        return first_ws(folder), folder, None, species, ""
    return Path(parts[-1]).stem, parts[-1], None, None, "flat_fallback"

# POLLEN23E
def parse_pollen23e(parts):                      
    stem = Path(parts[-1]).stem
    return first_us(stem), stem, None, None, ("" if len(parts) == 1 else "unexpected_depth")

# POLLEN73S
def parse_pollen73s(parts):                      
    if len(parts) >= 2:
        folder = parts[-2]
        return first_us(folder), folder, None, folder, ""
    return Path(parts[-1]).stem, parts[-1], None, None, "flat_fallback"

# GENERIC
def parse_generic(parts):                        
    if len(parts) >= 2:
        return first_ws(strip_numdot(parts[-2])), parts[-2], None, None, "generic_parser"
    stem = Path(parts[-1]).stem
    return first_ws(stem), stem, None, None, "generic_parser"

PARSERS = {
    "Cambridge": parse_cambridge,
    "Cropped Pollen Grains": parse_cropped,
    "Global Pollen Project": parse_gpp,
    "images_16_types": parse_images16,
    "oreme_processed": parse_oreme,
    "Other": parse_other,
    "PalDat": parse_paldat,
    "Pollen Image Dataset Kaggle": parse_kaggle,
    "Pollen Wiki": parse_pollenwiki,
    "POLLEN23E": parse_pollen23e,
    "POLLEN73S": parse_pollen73s,
}


def parse_taxon(dataset: str, rel_parts):
    parser = PARSERS.get(dataset, parse_generic)
    genus_raw, raw_label, family, species, flag = parser(rel_parts)
    genus = norm_genus(genus_raw)
    if not genus:
        flag = (flag + ";empty_genus").strip(";")
    elif genus.lower() in WRAPPER_TOKENS:
        flag = (flag + ";wrapper_token").strip(";")
    return genus, raw_label, family, species, flag


# Read image details and blur
_MODE_BITS = {
    "1": 1, "L": 8, "P": 8, "RGB": 8, "RGBA": 8, "CMYK": 8, "YCbCr": 8,
    "LA": 8, "I": 32, "F": 32, "I;16": 16, "I;16B": 16, "I;16L": 16,
}

def read_meta_and_blur(path: Path):
# Individual image: size, format, colour depth, file size and a blur score
    info = dict(width="", height="", mode="", bit_depth="", n_frames=1,
                file_size=path.stat().st_size, laplacian_var="", err="")
    try:
        with Image.open(path) as im:
            info["width"], info["height"] = im.size
            info["mode"] = im.mode
            info["bit_depth"] = _MODE_BITS.get(im.mode, "")
            info["n_frames"] = getattr(im, "n_frames", 1)
            if COMPUTE_BLUR and _BLUR_BACKEND and HAVE_NUMPY:
                gray = im.convert("L")
                arr = np.asarray(gray, dtype=np.float64)
                if _BLUR_BACKEND == "cv2":
                    info["laplacian_var"] = float(cv2.Laplacian(arr, cv2.CV_64F).var())
                else:
                    info["laplacian_var"] = float(_scipy_laplace(arr).var())
    except Exception as e:
        info["err"] = f"{type(e).__name__}: {e}"

# Format check if Pillow was unsure
    if HAVE_TIFFFILE and path.suffix.lower() in {".tif", ".tiff"}:
        try:
            with tifffile.TiffFile(path) as tf:
                info["n_frames"] = max(info["n_frames"], len(tf.pages))
                dt = tf.pages[0].dtype
                if dt is not None:
                    info["bit_depth"] = int(dt.itemsize * 8)
        except Exception:
            pass
    return info


# Main part of the script
FIELDS = ["dataset", "rel_path", "filename", "ext", "genus", "raw_label",
          "family", "species", "width", "height", "mode", "bit_depth",
          "n_frames", "file_size_bytes", "laplacian_var", "parse_flag"]

def included(name: str):
# True if this dataset is not on the excluded list.
    if name in DATASET_EXCLUDE:
        return False
    if name.lower().endswith(EXCLUDED_SUFFIX):
        return False
    return True

def main():
# Go through all datasets, read and measure every image, and save the inventory. Can resume.
    if not RAW_DIR.exists():
        raise SystemExit(f"RAW_DIR does not exist:\n  {RAW_DIR}\nEdit PROJECT_ROOT.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

# Resume: skip images already done
    done = set()
    if OUT_CSV.exists():
        with OUT_CSV.open("r", newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                done.add(row["rel_path"])
        print(f"Resuming — {len(done)} images already in {OUT_CSV.name}")

    datasets = sorted([d for d in RAW_DIR.iterdir() if d.is_dir()],
                      key=lambda d: d.name.lower())

# List of images left to do
    worklist = []
    for ds in datasets:
        if not included(ds.name):
            print(f"  skip (excluded): {ds.name}")
            continue
        for p in ds.rglob("*"):
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS:
                rel = str(p.relative_to(RAW_DIR))
                if rel not in done:
                    worklist.append((ds.name, p, rel))

    total = len(worklist)
    print(f"To process: {total} images "
          f"(blur={'on' if COMPUTE_BLUR and _BLUR_BACKEND else 'off'}, "
          f"backend={_BLUR_BACKEND})")
    if not total:
        print("Nothing to do.")
        return

    write_header = not OUT_CSV.exists()
    genus_counter = Counter()
    flag_counter = Counter()
    t0 = time.time()

    with OUT_CSV.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if write_header:
            w.writeheader()

        for i, (dsname, path, rel) in enumerate(worklist, 1):
            rel_parts = path.relative_to(RAW_DIR / dsname).parts
            genus, raw_label, family, species, flag = parse_taxon(dsname, rel_parts)
            meta = read_meta_and_blur(path)
            if meta["err"]:
                flag = (flag + ";read_error").strip(";")

            w.writerow({
                "dataset": dsname,
                "rel_path": rel,
                "filename": path.name,
                "ext": path.suffix.lower(),
                "genus": genus,
                "raw_label": raw_label,
                "family": family or "",
                "species": species or "",
                "width": meta["width"],
                "height": meta["height"],
                "mode": meta["mode"],
                "bit_depth": meta["bit_depth"],
                "n_frames": meta["n_frames"],
                "file_size_bytes": meta["file_size"],
                "laplacian_var": meta["laplacian_var"],
                "parse_flag": flag,
            })
            genus_counter[genus] += 1
            if flag:
                for fl in flag.split(";"):
                    flag_counter[fl] += 1

            if i % FLUSH_EVERY == 0:
                f.flush()
            if i % PROGRESS_EVERY == 0:
                rate = i / (time.time() - t0)
                eta = (total - i) / rate / 60 if rate else 0
                print(f"  {i}/{total}  ({rate:.0f} img/s, ETA {eta:.1f} min)")

    dt = time.time() - t0
    print(f"\nInventory complete in {dt/60:.1f} min.")
    print(f"  Rows written this run : {total}")
    print(f"  Unique genera         : {len(genus_counter)}")
    print(f"  CSV                    : {OUT_CSV}")
    if flag_counter:
        print("  Parse flags (review these):")
        for fl, n in flag_counter.most_common():
            print(f"     {fl}: {n}")
    print("\n  Top 15 genera by image count:")
    for g, n in genus_counter.most_common(15):
        print(f"     {g}: {n}")


if __name__ == "__main__":
    main()
