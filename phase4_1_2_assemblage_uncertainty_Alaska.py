#!/usr/bin/env python3
# What this does: reporting ranges for each taxa, sampling and classification uncertainty. ALASKA
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from phase4_1_1_assemblage_Alaska import find_family_map, UNKNOWN
except Exception:
    UNKNOWN = "unknown"
    def find_family_map(explicit=None):
        return {}

# Alaska root
PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
OUT_DIR = PROJECT_ROOT / "outputs" / "assemblage"
DEFAULT_PRED = EVAL_DIR / "calibrated_test_predictions.csv"


def bootstrap_profiles(df, taxon_col, conf_col, indet_col, n_boot=1000, seed=42):
    taxa = df[taxon_col].astype(str).to_numpy()
    conf = df[conf_col].astype(float).clip(0.0, 1.0).to_numpy()
    indet = df[indet_col].astype(bool).to_numpy()
    N = len(df)

    uniq = sorted(set(taxa[~indet]))
    code = {t: i for i, t in enumerate(uniq)}
    K = len(uniq)
    tcode = np.array([code.get(t, -1) for t in taxa])
    retainable = (~indet) & (tcode >= 0)

    rng = np.random.default_rng(seed)
    draws = np.zeros((n_boot, K + 1), dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, N, N)
        keep = retainable[idx] & (rng.random(N) < conf[idx])
        counts = np.bincount(tcode[idx][keep], minlength=K) if keep.any() else np.zeros(K)
        draws[b, :K] = counts
        draws[b, K] = N - keep.sum()
    draws = draws / N * 100.0
    return uniq + [UNKNOWN], draws


def summarise_draws(taxa, draws, family_map):
    mean = draws.mean(axis=0)
    lo = np.percentile(draws, 2.5, axis=0)
    hi = np.percentile(draws, 97.5, axis=0)
    sd = draws.std(axis=0, ddof=1)
    rows = []
    for i, t in enumerate(taxa):
        if mean[i] < 1e-9 and hi[i] < 1e-9:
            continue
        fam = UNKNOWN if t == UNKNOWN else family_map.get(t, "unassigned")
        rows.append({"taxon": t, "family": fam, "mean_pct": mean[i],
                     "lo95_pct": lo[i], "hi95_pct": hi[i], "sd_pct": sd[i]})
    prof = pd.DataFrame(rows)
    prof["_u"] = (prof.taxon == UNKNOWN).astype(int)
    prof = (prof.sort_values(["_u", "family", "mean_pct"], ascending=[True, True, False])
                .drop(columns="_u").reset_index(drop=True))
    return prof


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", default=str(DEFAULT_PRED))
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--taxon-col", default="pred_label")
    ap.add_argument("--conf-col", default="calibrated_confidence")
    ap.add_argument("--indet-col", default="indeterminate")
    ap.add_argument("--hierarchy", default=None,
                    help="explicit path to taxonomy_hierarchy.csv (else auto-search)")
    ap.add_argument("--save-draws", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "assemblage_uncertainty.csv"))
    args = ap.parse_args()

    pred_path = Path(args.pred)
    if not pred_path.exists():
        sys.exit(f"predictions not found: {pred_path}\nRun phase3_3_calibration.py first.")
    df = pd.read_csv(pred_path)
    for col in (args.taxon_col, args.conf_col, args.indet_col):
        if col not in df.columns:
            sys.exit(f"column '{col}' not in {pred_path.name} (has: {list(df.columns)})")

    family_map = find_family_map(args.hierarchy)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"{len(df)} grains | {args.boot} bootstrap iterations | seed {args.seed}")

    groups = [(None, df)] if not (args.group_col and args.group_col in df.columns) \
        else list(df.groupby(args.group_col))
    if args.group_col and args.group_col not in df.columns:
        print(f"(group column '{args.group_col}' not found; single distribution)")

    out_rows, draw_store = [], {}
    for g, sub in groups:
        taxa, draws = bootstrap_profiles(sub, args.taxon_col, args.conf_col,
                                         args.indet_col, args.boot, args.seed)
        prof = summarise_draws(taxa, draws, family_map)
        if g is not None:
            prof.insert(0, args.group_col, g)
            draw_store[str(g)] = (taxa, draws)
        label = f"[{g}] " if g is not None else ""
        unk = prof[prof.taxon == UNKNOWN]
        if len(unk):
            u = unk.iloc[0]
            print(f"{label}unknown {u['mean_pct']:.1f}% (95% CI {u['lo95_pct']:.1f}-{u['hi95_pct']:.1f})")
        for _, r in prof[prof.taxon != UNKNOWN].head(6).iterrows():
            print(f"  {label}{r['taxon'][:24]:24s} {r['mean_pct']:5.2f}% "
                  f"(95% CI {r['lo95_pct']:.2f}-{r['hi95_pct']:.2f})")
        out_rows.append(prof)

    pd.concat(out_rows, ignore_index=True).to_csv(args.out, index=False, encoding="utf-8")
    print(f"\nwrote assemblage uncertainty -> {args.out}")
    if args.save_draws:
        if not draw_store:
            taxa, draws = bootstrap_profiles(df, args.taxon_col, args.conf_col,
                                             args.indet_col, args.boot, args.seed)
            np.savez_compressed(OUT_DIR / "assemblage_bootstrap.npz",
                                taxa=np.array(taxa, dtype=object), draws=draws)
        else:
            np.savez_compressed(OUT_DIR / "assemblage_bootstrap.npz",
                                **{f"taxa__{k}": np.array(v[0], dtype=object) for k, v in draw_store.items()},
                                **{f"draws__{k}": v[1] for k, v in draw_store.items()})
        print(f"wrote bootstrap draws -> {OUT_DIR / 'assemblage_bootstrap.npz'}")


if __name__ == "__main__":
    main()
