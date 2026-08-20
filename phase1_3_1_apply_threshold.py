#!/usr/bin/env python3
# What this does: applies the minimum-images-per-class.
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
INV = PROJECT_ROOT / "outputs" / "inventory" / "master_inventory.csv"
HIER_CSV = PROJECT_ROOT / "outputs" / "taxonomy" / "taxonomy_hierarchy.csv"
OUT = PROJECT_ROOT / "outputs" / "curation"
OUT.mkdir(parents=True, exist_ok=True)

#Threshold
THRESHOLD = 20       
# Rescue band
RESCUE_FLOOR = 10    


def main():
# Map to final classes
    if not INV.exists():
        raise SystemExit(f"Missing {INV}")
    if not HIER_CSV.exists():
        raise SystemExit(f"Missing {HIER_CSV}. Run phase1.2.4 first.")

    inv = pd.read_csv(INV, low_memory=False, usecols=["dataset", "genus"])
    inv["genus"] = inv["genus"].fillna("").astype(str).str.strip()

    hier = pd.read_csv(HIER_CSV, dtype=str, keep_default_na=False)
    to_final = dict(zip(hier["original_label"], hier["final_label"]))
    to_rank = dict(zip(hier["final_label"], hier["label_rank"]))

    n_total = len(inv)
    inv["final_label"] = inv["genus"].map(to_final).fillna("")
    dropped_excluded = inv[inv["final_label"] == ""]
    kept_imgs = inv[inv["final_label"] != ""].copy()
    print(f"images in inventory       : {n_total:,}")
    print(f"  dropped (excluded taxa) : {len(dropped_excluded):,}")
    print(f"  mapped to classes       : {len(kept_imgs):,}")

# Per-class counts
    agg = (kept_imgs.groupby("final_label")
           .agg(n_images=("final_label", "size"),
                n_datasets=("dataset", "nunique"),
                datasets=("dataset", lambda s: "|".join(sorted(set(s)))))
           .reset_index())
    agg["label_rank"] = agg["final_label"].map(to_rank).fillna("genus")

    def status(n):
# Sort class by its image count
        if n >= THRESHOLD:
            return "keep"
        if n >= RESCUE_FLOOR:
            return "rescue_candidate"
        return "drop"
    agg["status"] = agg["n_images"].map(status)
    agg = agg.sort_values(["status", "n_images"], ascending=[True, False])

    agg.to_csv(OUT / "class_inclusion_log.csv", index=False, encoding="utf-8")
    keep = agg[agg.status == "keep"].sort_values("n_images", ascending=False)
    rescue = agg[agg.status == "rescue_candidate"].sort_values("n_images", ascending=False)
    drop = agg[agg.status == "drop"]
    keep.to_csv(OUT / "survivors.csv", index=False, encoding="utf-8")
    rescue.to_csv(OUT / "rescue_candidates.csv", index=False, encoding="utf-8")

    print(f"\nreconciled classes        : {len(agg):,}")
    print(f"  keep (>= {THRESHOLD})           : {len(keep):,}  "
          f"({keep.n_images.sum():,} images)")
    print(f"  rescue ({RESCUE_FLOOR}-{THRESHOLD-1})        : {len(rescue):,}  "
          f"({rescue.n_images.sum():,} images)  -> forensic review")
    print(f"  drop (< {RESCUE_FLOOR})           : {len(drop):,}  "
          f"({drop.n_images.sum():,} images)")
    print(f"\n  family-level classes kept : "
          f"{int((keep.label_rank=='family').sum())} "
          f"({', '.join(keep[keep.label_rank=='family']['final_label'].tolist()) or 'none'})")
    print(f"\n  written to {OUT}\\:")
    print("    class_inclusion_log.csv, survivors.csv, rescue_candidates.csv")
    print("\n  Top 15 survivor classes:")
    for _, r in keep.head(15).iterrows():
        print(f"     {r['final_label']}: {r['n_images']} ({r['n_datasets']} sources)")


if __name__ == "__main__":
    main()
