#!/usr/bin/env python3
# What this does: fixes the 70/15/15 train/validation/test split, balanced by class and by source dataset.
from pathlib import Path
import json
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
INV = PROJECT_ROOT / "outputs" / "inventory" / "master_inventory.csv"
HIER = PROJECT_ROOT / "outputs" / "taxonomy" / "taxonomy_hierarchy.csv"
TIERS = PROJECT_ROOT / "outputs" / "taxonomy" / "forensic_relevance_tiers.csv"
INCL = PROJECT_ROOT / "outputs" / "curation" / "class_inclusion_log.csv"
DEDUP = PROJECT_ROOT / "outputs" / "curation" / "dedup_clusters.csv"
BLUR = PROJECT_ROOT / "outputs" / "curation" / "blur_flags.csv"
OUT = PROJECT_ROOT / "outputs" / "curation" / "splits"

RANDOM_SEED = 42
TEST_FRAC = 0.15
VAL_FRAC = 0.15
RESCUE_TIERS = {"High"}   
MIN_FOR_EVAL = 4         
EXCLUDE_BLUR = False        
EXCLUDE_CONFLICTS = True     


def split_group(idx, rng):
# Split a set of rows 70/15/15 into train, validation, test
    idx = idx.copy()
    rng.shuffle(idx)
    n = len(idx)
    n_test = int(round(n * TEST_FRAC))
    n_val = int(round(n * VAL_FRAC))
    return idx[n_test + n_val:], idx[n_test:n_test + n_val], idx[:n_test]


def main():
# Choose the final class list, and save the fixed split and a report
    for p in (INV, HIER, INCL):
        if not p.exists():
            raise SystemExit(f"Missing {p}")
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    inv = pd.read_csv(INV, low_memory=False)
    inv["genus"] = inv["genus"].fillna("").astype(str).str.strip()

    hier = pd.read_csv(HIER, dtype=str, keep_default_na=False)
    to_final = dict(zip(hier["original_label"], hier["final_label"]))
    inv["final_label"] = inv["genus"].map(to_final).fillna("")

    incl = pd.read_csv(INCL)
    keep = set(incl[incl.status == "keep"]["final_label"])
    rescue = set(incl[incl.status == "rescue_candidate"]["final_label"])

    tier = {}
    if TIERS.exists():
        td = pd.read_csv(TIERS)
        tier = dict(zip(td["final_label"], td["forensic_relevance_tier"]))
    rescued = {c for c in rescue if tier.get(c) in RESCUE_TIERS}
    final_classes = keep | rescued
    high_classes = {c for c, t in tier.items() if t == "High"} & final_classes
    print(f"final classes: {len(final_classes):,} "
          f"(survivors {len(keep)} + rescued-High {len(rescued)})")

    work = inv[inv["final_label"].isin(final_classes)].copy()
    n0 = len(work)

# Exclusions
    if DEDUP.exists():
        ded = pd.read_csv(DEDUP)
        if "cross_label_CONFLICT" in ded.columns:
            conflict_mask = ded["cross_label_CONFLICT"].astype(str).str.lower().isin(["true", "1"])
        else:
            conflict_mask = pd.Series(False, index=ded.index)
        remove = set(ded[ded.role == "remove_candidate"]["rel_path"])
        conflict_imgs = set(ded[conflict_mask]["rel_path"]) if EXCLUDE_CONFLICTS else set()
        dedup_only = remove - conflict_imgs
        work = work[~work["rel_path"].isin(remove | conflict_imgs)]
        print(f"  dropped dedup duplicates: {len(dedup_only):,}")
        print(f"  excluded cross-label-conflict images: {len(conflict_imgs):,}")
    else:
        print("  (no dedup_clusters.csv - run 1.3.3 first to remove duplicates)")
    if EXCLUDE_BLUR and BLUR.exists():
        bl = set(pd.read_csv(BLUR)["rel_path"])
        before = len(work)
        work = work[~work["rel_path"].isin(bl)]
        print(f"  dropped blur-flagged: {before - len(work):,}")

# Class numbering
    classes = sorted(final_classes)
    class_index = {c: i for i, c in enumerate(classes)}

# Split, balanced by class and source
    splits = {"train": [], "val": [], "test": []}
    too_small = []
    for label, cg in work.groupby("final_label"):
        if len(cg) < MIN_FOR_EVAL:
            splits["train"].extend(cg.index.tolist())
            too_small.append((label, len(cg)))
            continue
        tr, va, te = [], [], []
        for _, sg in cg.groupby("dataset"):
            a, b, c = split_group(sg.index.to_numpy(), rng)
            tr += list(a); va += list(b); te += list(c)
# Ensure validation and test each get coverage
        for bucket in (va, te):
            if not bucket and len(tr) > 1:
                bucket.append(tr.pop())
        splits["train"] += tr; splits["val"] += va; splits["test"] += te

    def frame(idxs):
        d = work.loc[idxs, ["rel_path", "dataset", "final_label"]].copy()
        d["label_index"] = d["final_label"].map(class_index)
        return d.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)

    out = {k: frame(v) for k, v in splits.items()}
    for k, d in out.items():
        d.to_csv(OUT / f"{k}.csv", index=False, encoding="utf-8")
    (OUT / "class_index.json").write_text(json.dumps(class_index, indent=1), encoding="utf-8")

# Coverage checks
    val_classes = set(out["val"]["final_label"])
    test_classes = set(out["test"]["final_label"])
    high_missing = sorted(c for c in high_classes if c not in val_classes or c not in test_classes)

    rep = []
    for c in classes:
        rep.append({"final_label": c,
                    "tier": tier.get(c, ""),
                    "rescued": c in rescued,
                    "n_train": int((out["train"].final_label == c).sum()),
                    "n_val": int((out["val"].final_label == c).sum()),
                    "n_test": int((out["test"].final_label == c).sum())})
    pd.DataFrame(rep).to_csv(OUT / "split_report.csv", index=False)

    print(f"\nsplit sizes: train {len(out['train']):,} | "
          f"val {len(out['val']):,} | test {len(out['test']):,}")
    print(f"  classes in all 3 splits: "
          f"{len(set(classes) & val_classes & test_classes):,} / {len(classes):,}")
    print(f"  classes too small to evaluate (<{MIN_FOR_EVAL}, train-only): {len(too_small)}")
    print(f"  High-relevance classes missing from val/test: {len(high_missing)}")
    if high_missing:
        print("   " + ", ".join(high_missing))
    print(f"\n  written: train/val/test.csv, class_index.json, split_report.csv -> {OUT}")
    print("  These splits are FIXED. Do not re-randomise after this point.")


if __name__ == "__main__":
    main()
