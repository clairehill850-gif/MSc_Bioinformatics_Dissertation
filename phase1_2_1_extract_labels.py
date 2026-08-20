#!/usr/bin/env python3
# What this does: lists unique plant names taken from the image inventory, counts how many images and how many datasets each one is in. This list is checked against GBIF.
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
INV = PROJECT_ROOT / "outputs" / "inventory" / "master_inventory.csv"
OUT = PROJECT_ROOT / "outputs" / "taxonomy"
OUT.mkdir(parents=True, exist_ok=True)
THRESHOLD = 20   


def main():
# Images per plant
    df = pd.read_csv(INV, low_memory=False)
    df["genus"] = df["genus"].fillna("").astype(str).str.strip()
    df = df[df["genus"] != ""]

    g = (df.groupby("genus")
           .agg(n_images=("genus", "size"),
                n_datasets=("dataset", "nunique"),
                datasets=("dataset", lambda s: "|".join(sorted(set(s)))))
           .reset_index()
           .sort_values("n_images", ascending=False))
    g["above_threshold"] = g["n_images"] >= THRESHOLD

    (OUT / "raw_labels.txt").write_text(
        "\n".join(g["genus"].tolist()), encoding="utf-8")
    g.to_csv(OUT / "raw_label_counts.csv", index=False, encoding="utf-8")

    print(f"unique genus labels      : {len(g):,}")
    print(f"  above threshold (>={THRESHOLD}) : {int(g.above_threshold.sum()):,}")
    print(f"  below threshold         : {int((~g.above_threshold).sum()):,}")
    print(f"written: {OUT/'raw_labels.txt'}")
    print(f"written: {OUT/'raw_label_counts.csv'}")


if __name__ == "__main__":
    main()
