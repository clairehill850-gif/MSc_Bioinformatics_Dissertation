#!/usr/bin/env python3
# What this does: builds Alaska regional splits 
import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import pandas as pd

# Pod set up
PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", "/workspace/datasets/pollen_bundle"))
FROZEN_SPLITS = PROJECT_ROOT / "outputs" / "curation" / "splits"
ALASKA_SPLITS = PROJECT_ROOT / "Alaska_Test" / "outputs" / "curation" / "splits"
CLASS_LIST = PROJECT_ROOT / "outputs" / "alaska" / "alaska_class_list.csv"
HIER = PROJECT_ROOT / "outputs" / "taxonomy" / "taxonomy_hierarchy.csv"
RAW_COUNTS = PROJECT_ROOT / "outputs" / "taxonomy" / "raw_label_counts.csv"
LUT = PROJECT_ROOT / "outputs" / "taxonomy" / "taxonomy_lookup.csv"

SPLIT_COLS = ["processed_path", "raw_rel_path", "dataset", "final_label", "label_index"]

# Combine spores
SPORE_GBIF_CLASSES = [
    "Polypodiopsida", "Lycopodiopsida", "Equisetopsida", "Bryopsida",
    "Sphagnopsida", "Marchantiopsida", "Jungermanniopsida", "Psilotopsida",
    "Marattiopsida", "Anthocerotopsida",
]
SPORE_LABEL = "Spores-undiff"

# Produces pollen, not spores
SPORE_EXCLUDE = {"Ancistrophyllum"}


# Assign split
def assign_split(key, ratios=(0.70, 0.15, 0.15)):
    h = int(hashlib.md5(key.encode("utf-8")).hexdigest(), 16)
    frac = (h % 10_000) / 10_000.0
    if frac < ratios[0]:
        return "train"
    if frac < ratios[0] + ratios[1]:
        return "val"
    return "test"


# Load frozen splits
def load_frozen():
    frames = {}
    for sp in ("train", "val", "test"):
        p = FROZEN_SPLITS / f"processed_{sp}.csv"
        if not p.exists():
            raise SystemExit(f"missing frozen split: {p}")
        df = pd.read_csv(p)
        df["_split"] = sp
        frames[sp] = df
    return pd.concat(frames.values(), ignore_index=True)


# More spore stuff
def spore_labels(hier):
    src = LUT if LUT.exists() else None
    if src is None:
        print(f"  WARNING: {LUT.name} not found; spore pooling skipped.")
        return set()
    lut = pd.read_csv(src, low_memory=False)
    if "class" not in lut.columns:
        print(f"  WARNING: no 'class' column in {src.name}; spore pooling skipped.")
        return set()
    sp = lut[lut["class"].isin(SPORE_GBIF_CLASSES)]
    genera = set(sp["accepted_genus"].dropna().astype(str)) - SPORE_EXCLUDE
# Translate accepted genus into final label through the hierarchy, one to one
    g2l = (hier.dropna(subset=["accepted_genus"])
               .drop_duplicates("accepted_genus")
               .set_index("accepted_genus")["final_label"])
    labels = set(g2l.reindex(genera).dropna().astype(str))
    labels &= set(hier["final_label"].dropna().astype(str))
    return labels


# Build Alaska splits
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--picea-raw", default=None,
                    help="CSV of new Picea raw images with columns: processed_path,raw_rel_path,dataset")
    ap.add_argument("--salix-raw", default=None,
                    help="CSV of new Salix raw images (same columns); all go to TRAIN")
    ap.add_argument("--pool-spores", action="store_true",
                    help="fold spore genera into a single Spores-undiff class")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for p in (CLASS_LIST, HIER):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    class_list = pd.read_csv(CLASS_LIST)
    hier = pd.read_csv(HIER)
    keep = set(class_list["final_label"].astype(str))
    print(f"a priori class list: {len(keep)} classes")

    pooled_from = set()
    if args.pool_spores:
        pooled_from = spore_labels(hier)
        print(f"pooling {len(pooled_from)} spore labels -> {SPORE_LABEL}")
        keep = (keep - pooled_from) | {SPORE_LABEL}

    frozen = load_frozen()
    print(f"frozen images: {len(frozen):,} across {frozen.final_label.nunique()} classes")

# Relabel spores
    if args.pool_spores and pooled_from:
        frozen.loc[frozen.final_label.isin(pooled_from), "final_label"] = SPORE_LABEL

    sub = frozen[frozen.final_label.isin(keep)].copy()
    print(f"after restriction: {len(sub):,} images, {sub.final_label.nunique()} classes present")
    dropped = keep - set(sub.final_label) - {SPORE_LABEL}
    if dropped:
        print(f"  NOTE: {len(dropped)} listed classes have no frozen images "
              f"(new-grain classes?): {sorted(dropped)[:6]}")

# Add new grain images
    def load_new(path, name, force_split=None):
        if not path:
            return None
        df = pd.read_csv(path)
        need = {"processed_path", "raw_rel_path", "dataset"}
        if not need.issubset(df.columns):
            raise SystemExit(f"{name}: needs columns {need}, got {list(df.columns)}")
        df["final_label"] = name
        if force_split:
            df["_split"] = force_split
        else:
            df["_split"] = [assign_split(pp) for pp in df["processed_path"]]
        print(f"  +{len(df)} {name} "
              f"({'train only' if force_split else 'fresh 70/15/15: ' + str(df._split.value_counts().to_dict())})")
        return df

    additions = [x for x in (
        load_new(args.picea_raw, "Picea"),
        load_new(args.salix_raw, "Salix", force_split="train"),
    ) if x is not None]

    if additions:
        sub = pd.concat([sub] + additions, ignore_index=True)

    classes = sorted(sub.final_label.astype(str).unique())
    class_index = {c: i for i, c in enumerate(classes)}
    sub["label_index"] = sub.final_label.map(class_index)

    def fix_prefix(row):
        pp = row["processed_path"]
        parts = pp.split("/", 1)
        if len(parts) == 2 and parts[0] in ("train", "val", "test"):
            return f"{row['_split']}/{parts[1]}"
        return pp
    sub["processed_path"] = sub.apply(fix_prefix, axis=1)

# Leakage check
    dup = sub.groupby("processed_path")._split.nunique()
    leaked = dup[dup > 1]
    if len(leaked):
        raise SystemExit(f"LEAKAGE: {len(leaked)} images in >1 split, "
                         f"e.g. {list(leaked.index[:3])}")

    print(f"\nfinal: {len(sub):,} images, {len(classes)} classes")
    for sp in ("train", "val", "test"):
        n = (sub._split == sp).sum()
        print(f"  {sp:5s} {n:6,d}")

# A dry run stops here
    if args.dry_run:
        print("\ndry run: nothing written.")
        return

# Write splits
    ALASKA_SPLITS.mkdir(parents=True, exist_ok=True)
    for sp in ("train", "val", "test"):
        out = sub[sub._split == sp][SPLIT_COLS]
        out.to_csv(ALASKA_SPLITS / f"processed_{sp}.csv", index=False)
    (ALASKA_SPLITS / "class_index_final.json").write_text(
        json.dumps(class_index, indent=2), encoding="utf-8")
    print(f"\nwritten -> {ALASKA_SPLITS}")
    print("  processed_{train,val,test}.csv + class_index_final.json")


if __name__ == "__main__":
    main()
