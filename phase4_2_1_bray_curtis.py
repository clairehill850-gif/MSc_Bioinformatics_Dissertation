#!/usr/bin/env python3
# What this does: Bray-Curtis dissimilarity.
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from phase4_1_1_assemblage import assemblage_profile, find_family_map, UNKNOWN
except Exception:
    UNKNOWN = "unknown"
    assemblage_profile = None
    def find_family_map(explicit=None):
        return {}

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
OUT_DIR = PROJECT_ROOT / "outputs" / "assemblage"
RESERVED = {"taxon", "family", "count", "pct_count", "cw_weight", "pct_cw",
            "mean_pct", "lo95_pct", "hi95_pct", "sd_pct"}


def bray_curtis(a, b):
# Bray-Curtis dissimilarity between two lists
    a = np.asarray(a, float); b = np.asarray(b, float)
    denom = (a + b).sum()
    return float(np.abs(a - b).sum() / denom) if denom > 0 else np.nan


def pairwise_matrix(M):
# Dissimilarity table
    names = list(M.index)
    A = M.to_numpy()
    n = len(names)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = bray_curtis(A[i], A[j])
            out[i, j] = out[j, i] = d
    return pd.DataFrame(out, index=names, columns=names)


def drop_unknown_renorm(M):
# Drop "unknown", re-scale to 100%
    if UNKNOWN in M.columns:
        M = M.drop(columns=[UNKNOWN])
    rs = M.sum(axis=1)
    rs = rs.replace(0, np.nan)
    return M.div(rs, axis=0).mul(100.0).fillna(0.0)


def build_from_profiles(path, group_col, value):
# Samples x taxa table
    df = pd.read_csv(path, keep_default_na=False)
    if "taxon" not in df.columns or value not in df.columns:
        sys.exit(f"{Path(path).name} must have 'taxon' and '{value}' columns (has {list(df.columns)})")
    if group_col is None:
        cands = [c for c in df.columns if c not in RESERVED]
        if len(cands) != 1:
            sys.exit("could not infer the sample column; pass --group-col. "
                     "Re-run phase4_1_1_assemblage.py with --group-col to get grouped profiles.")
        group_col = cands[0]
    if group_col not in df.columns:
        sys.exit(f"group column '{group_col}' not in {Path(path).name}")
    df[value] = pd.to_numeric(df[value], errors="coerce").fillna(0.0)
    M = df.pivot_table(index=group_col, columns="taxon", values=value, aggfunc="sum", fill_value=0.0)
    return M


def build_from_predictions(path, group_col, value, taxon_col, conf_col, indet_col):
# Sample assemblages
    if assemblage_profile is None:
        sys.exit("could not import assemblage_profile from phase4_1_1_assemblage.py "
                 "(keep both scripts in the same directory).")
    df = pd.read_csv(path)
    for col in (group_col, taxon_col, conf_col, indet_col):
        if col not in df.columns:
            sys.exit(f"column '{col}' not in {Path(path).name} (has {list(df.columns)})")
    rows = {}
    for g, sub in df.groupby(group_col):
        prof = assemblage_profile(sub, taxon_col, conf_col, indet_col)
        rows[g] = dict(zip(prof.taxon, prof[value]))
    M = pd.DataFrame(rows).T.fillna(0.0)
    M.index.name = group_col
    return M


def main():
# Abundance and Bray-Curtis tables
    ap = argparse.ArgumentParser()
    ap.add_argument("--profiles", default=None, help="grouped profile CSV (long form)")
    ap.add_argument("--pred", default=None, help="per-grain predictions CSV (with --group-col)")
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--value", default="pct_cw", help="abundance column (pct_cw / pct_count / mean_pct)")
    ap.add_argument("--taxon-col", default="pred_label")
    ap.add_argument("--conf-col", default="calibrated_confidence")
    ap.add_argument("--indet-col", default="indeterminate")
    ap.add_argument("--include-unknown", action="store_true",
                    help="keep 'unknown' as a shared category instead of dropping + renormalising")
    ap.add_argument("--out", default=str(OUT_DIR / "bray_curtis_matrix.csv"))
    args = ap.parse_args()

    if args.pred:
        M = build_from_predictions(args.pred, args.group_col, args.value,
                                   args.taxon_col, args.conf_col, args.indet_col)
    elif args.profiles:
        M = build_from_profiles(args.profiles, args.group_col, args.value)
    else:
        default = OUT_DIR / "assemblage_profile.csv"
        if not default.exists():
            sys.exit("provide --pred (with --group-col) or --profiles. "
                     "No default assemblage_profile.csv found.")
        M = build_from_profiles(default, args.group_col, args.value)

    if len(M) < 2:
        sys.exit(f"need >=2 samples for a dissimilarity matrix, found {len(M)}. "
                 "Group your grains into samples first (run 4.1.1 with --group-col, "
                 "or pass --pred with --group-col).")

    if not args.include_unknown:
        M = drop_unknown_renorm(M)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    M.to_csv(OUT_DIR / "abundance_matrix.csv")
    bc = pairwise_matrix(M)
    bc.to_csv(args.out)

    print(f"{len(M)} samples x {M.shape[1]} taxa | "
          f"unknown {'kept' if args.include_unknown else 'dropped + renormalised'}")
    tri = bc.where(np.triu(np.ones(bc.shape), k=1).astype(bool))
    vals = tri.stack()
    if len(vals):
        print(f"Bray-Curtis dissimilarity: min {vals.min():.3f}, "
              f"median {vals.median():.3f}, max {vals.max():.3f}")
        lo = vals.idxmin(); hi = vals.idxmax()
        print(f"  most similar : {lo[0]} <-> {lo[1]}  ({vals.min():.3f})")
        print(f"  most distinct: {hi[0]} <-> {hi[1]}  ({vals.max():.3f})")
    print(f"\nwrote bray_curtis_matrix.csv + abundance_matrix.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
