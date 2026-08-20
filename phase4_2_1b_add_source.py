#!/usr/bin/env python3
# What this does: adds a source column to the calibrated predictions
import os
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
CALIB = EVAL_DIR / "calibrated_test_predictions.csv"
DUMP = EVAL_DIR / "test_predictions.csv"
OUT = EVAL_DIR / "calibrated_test_predictions_src.csv"
KEY = "processed_path"


def pick_source_col(df):
# Find the source/dataset column
    for c in ("dataset", "source", "source_dataset", "src"):
        if c in df.columns:
            return c
    return None


def main():
# Add source column
    if not CALIB.exists():
        sys.exit(f"not found: {CALIB} (run phase3_3_calibration.py first)")
    if not DUMP.exists():
        sys.exit(f"not found: {DUMP} (run phase2_3b_dump_predictions.py first)")

    cal = pd.read_csv(CALIB)
    dump = pd.read_csv(DUMP)
    src = pick_source_col(dump)
    if src is None:
        sys.exit(f"no source/dataset column in {DUMP.name} (has {list(dump.columns)})")
    if KEY not in cal.columns:
        sys.exit(f"'{KEY}' not in {CALIB.name}. Re-run the patched phase3_3_calibration.py "
                 "so the calibrated file carries the join key.")
    if KEY not in dump.columns:
        sys.exit(f"'{KEY}' not in {DUMP.name} (has {list(dump.columns)}); cannot key-join.")
    for name, d in [(CALIB.name, cal), (DUMP.name, dump)]:
        dup = d[KEY].duplicated().sum()
        if dup:
            sys.exit(f"{name} has {dup} duplicate '{KEY}' values; cannot key-join safely.")

    merged = cal.merge(dump[[KEY, src]].rename(columns={src: "source"}), on=KEY, how="left")
    unmatched = merged["source"].isna().sum()
    if unmatched:
        print(f"warning: {unmatched}/{len(merged)} grains had no match in {DUMP.name} "
              "(source left blank for those).")
    else:
        print(f"clean 1:1 join on '{KEY}': all {len(merged)} grains matched.")

    merged.to_csv(OUT, index=False)
    vc = merged["source"].value_counts(dropna=False)
    print(f"\nattached 'source' from {DUMP.name}:[{src}] -> {OUT.name}")
    print(f"{vc.notna().sum() if hasattr(vc, 'notna') else len(vc)} sources; grains per source:")
    for s, n in vc.items():
        print(f"  {str(s)[:32]:32s} {n}")
    small = vc[vc < 30]
    if len(small):
        print(f"\nnote: {len(small)} source(s) have <30 grains; their assemblage profiles "
              "(and any Bray-Curtis involving them) will be noisy - interpret with care.")


if __name__ == "__main__":
    main()
