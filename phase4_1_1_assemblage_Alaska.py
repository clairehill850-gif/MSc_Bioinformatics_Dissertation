#!/usr/bin/env python3
# What this does: single grain to assemblage ALASKA.
import os

import sys
import glob
import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

# Alaksa root
PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
OUT_DIR = PROJECT_ROOT / "outputs" / "assemblage"
DEFAULT_PRED = EVAL_DIR / "calibrated_test_predictions.csv"
UNKNOWN = "unknown"

def find_family_map(explicit=None):
    if explicit:
        paths = [str(explicit)]
    else:
        paths = []
        for pat in ("**/taxonomy_hierarchy.csv", "**/*taxonomy*hierarchy*.csv", "**/*hierarchy*.csv"):
            paths += glob.glob(str(PROJECT_ROOT / pat), recursive=True)
        seen = []
        for p in paths:
            if p not in seen:
                seen.append(p)
        non_prepool = [p for p in seen if "prepool" not in Path(p).name.lower()]
        paths = non_prepool or seen
    for p in paths:
        try:
            df = pd.read_csv(p)
        except Exception:
            continue
        low = {c.lower(): c for c in df.columns}
        lab = next((low[k] for k in ("final_label", "label", "taxon", "genus", "class", "name") if k in low), None)
        fam = next((c for c in df.columns if c.lower() == "family"), None)
        if lab and fam:
            m = {str(r[lab]): str(r[fam]) for _, r in df.iterrows()
                 if pd.notna(r[lab]) and pd.notna(r[fam])}
            if m:
                return m
    return {}


def assemblage_profile(df, taxon_col="pred_label", conf_col="calibrated_confidence",
                       indet_col="indeterminate", family_map=None):
    family_map = family_map or {}
    n = len(df)
    if n == 0:
        return pd.DataFrame(columns=["taxon", "family", "count", "pct_count", "cw_weight", "pct_cw"])
    taxa = df[taxon_col].astype(str).to_numpy()
    conf = df[conf_col].astype(float).clip(0.0, 1.0).to_numpy()
    indet = df[indet_col].astype(bool).to_numpy()

    count = defaultdict(float)
    cw = defaultdict(float)
    for t, c, ind in zip(taxa, conf, indet):
        if ind:
            count[UNKNOWN] += 1.0
            cw[UNKNOWN] += 1.0
        else:
            count[t] += 1.0
            cw[t] += c
            cw[UNKNOWN] += (1.0 - c)
    rows = []
    for t in sorted(set(count) | set(cw)):
        cnt, w = count.get(t, 0.0), cw.get(t, 0.0)
        if cnt < 1e-12 and w < 1e-12:
            continue
        fam = UNKNOWN if t == UNKNOWN else family_map.get(t, "unassigned")
        rows.append({"taxon": t, "family": fam,
                     "count": round(cnt, 4), "pct_count": cnt / n * 100.0,
                     "cw_weight": round(w, 4), "pct_cw": w / n * 100.0})
    prof = pd.DataFrame(rows)
    prof["_u"] = (prof.taxon == UNKNOWN).astype(int)
    prof = (prof.sort_values(["_u", "family", "pct_cw"], ascending=[True, True, False])
                .drop(columns="_u").reset_index(drop=True))
    return prof


def summarise(prof, df, indet_col, label=""):
    n = len(df)
    n_indet = int(df[indet_col].astype(bool).sum())
    unk = prof[prof.taxon == UNKNOWN]
    unk_raw = float(unk.pct_count.iloc[0]) if len(unk) else 0.0
    unk_cw = float(unk.pct_cw.iloc[0]) if len(unk) else 0.0
    n_taxa = int((prof.taxon != UNKNOWN).sum())
    head = f"[{label}] " if label else ""
    print(f"{head}{n} grains | retained {n - n_indet} ({(n - n_indet) / n * 100:.1f}%) | "
          f"indeterminate {n_indet} ({n_indet / n * 100:.1f}%)")
    print(f"{head}taxa detected: {n_taxa} | unknown bin: {unk_raw:.1f}% raw, {unk_cw:.1f}% conf-weighted")
    top = prof[prof.taxon != UNKNOWN].head(8)
    for _, r in top.iterrows():
        print(f"    {r['taxon'][:26]:26s} {r['family'][:18]:18s} "
              f"{r['pct_count']:5.1f}% raw | {r['pct_cw']:5.1f}% cw")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(DEFAULT_PRED))
    ap.add_argument("--group-col", default=None,
                    help="optional column to split into one profile per sample/provenance")
    ap.add_argument("--taxon-col", default="pred_label")
    ap.add_argument("--conf-col", default="calibrated_confidence")
    ap.add_argument("--indet-col", default="indeterminate")
    ap.add_argument("--hierarchy", default=None,
                    help="explicit path to taxonomy_hierarchy.csv (else auto-search)")
    ap.add_argument("--out", default=str(OUT_DIR / "assemblage_profile.csv"))
    args = ap.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        sys.exit(f"predictions not found: {pred_path}\nRun phase3_3_calibration.py first.")
    df = pd.read_csv(pred_path)
    for col in (args.taxon_col, args.conf_col, args.indet_col):
        if col not in df.columns:
            sys.exit(f"column '{col}' not in {pred_path.name} (has: {list(df.columns)})")

    family_map = find_family_map(args.hierarchy)
    print(f"loaded {len(df)} grains from {pred_path.name} | "
          f"family map: {len(family_map)} taxa" if family_map else
          f"loaded {len(df)} grains from {pred_path.name} | family map: none (family='NA')")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if args.group_col and args.group_col in df.columns:
        profiles = []
        for g, sub in df.groupby(args.group_col):
            p = assemblage_profile(sub, args.taxon_col, args.conf_col, args.indet_col, family_map)
            p.insert(0, args.group_col, g)
            profiles.append(p)
            summarise(p, sub, args.indet_col, label=str(g))
            print()
        out = pd.concat(profiles, ignore_index=True)
    else:
        if args.group_col:
            print(f"(group column '{args.group_col}' not found; producing a single profile)")
        out = assemblage_profile(df, args.taxon_col, args.conf_col, args.indet_col, family_map)
        summarise(out, df, args.indet_col)

    out.to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nwrote assemblage profile -> {args.out}")


if __name__ == "__main__":
    main()
