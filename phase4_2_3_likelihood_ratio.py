#!/usr/bin/env python3
# What this does: computes an illustrative likelihood ratio from Bray-Curtis values
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from phase4_1_2_assemblage_uncertainty import bootstrap_profiles
    from phase4_1_1_assemblage import assemblage_profile, UNKNOWN
    from phase4_2_1_bray_curtis import bray_curtis
except Exception as e:
    sys.exit(f"could not import Phase 4 helpers ({e}); keep all phase4_* scripts together.")

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
OUT_DIR = PROJECT_ROOT / "outputs" / "assemblage"

VERBAL = [(10, "weak"), (100, "moderate"), (1000, "moderately strong"),
          (10000, "strong"), (1e6, "very strong")]


def verbal(lr):
# Plain text answer
    if not np.isfinite(lr) or lr <= 0:
        return "undefined"
    if abs(np.log10(lr)) < 1e-9:
        return "no meaningful support either way"
    fav = lr if lr >= 1 else 1.0 / lr
    where = "A (H1)" if lr >= 1 else "B (H2)"
    for thr, words in VERBAL:
        if fav < thr:
            return f"{words} support for {where}"
    return f"extremely strong support for {where}"


def bc_between(taxa1, v1, taxa2, v2, include_unknown=False):
# Bray-Curtis dissimilarity
    idx = {}
    for t in list(taxa1) + list(taxa2):
        idx.setdefault(t, len(idx))
    a = np.zeros(len(idx)); b = np.zeros(len(idx))
    for t, val in zip(taxa1, v1):
        a[idx[t]] += val
    for t, val in zip(taxa2, v2):
        b[idx[t]] += val
    if not include_unknown and UNKNOWN in idx:
        k = idx[UNKNOWN]; a[k] = 0.0; b[k] = 0.0
    if a.sum() > 0:
        a = a / a.sum() * 100.0
    if b.sum() > 0:
        b = b / b.sum() * 100.0
    return bray_curtis(a, b)


def lr_from_scores(d_a, d_b, sigma, kernel="exponential"):
# The lr
    if kernel == "gaussian":
        return float(np.exp((d_b ** 2 - d_a ** 2) / (2.0 * sigma ** 2)))
# Exponential form (default)
    return float(np.exp((d_b - d_a) / sigma))


def point_profile(df, t_col, c_col, i_col):
# Confidence-weighted profile
    p = assemblage_profile(df, t_col, c_col, i_col)
    return list(p.taxon), p.pct_cw.to_numpy()


def main():
# Estimate, resampled range and fig
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True)
    ap.add_argument("--group-col", required=True)
    ap.add_argument("--q", required=True, help="group name of the questioned sample")
    ap.add_argument("--a", required=True, help="group name for hypothesis A (H1)")
    ap.add_argument("--b", required=True, help="group name for hypothesis B (H2)")
    ap.add_argument("--sigma", type=float, default=0.3,
                    help="within-source dissimilarity scale (ASSUMPTION; smaller=tighter)")
    ap.add_argument("--calib-within", default=None,
                    help="CSV/text of same-source Bray-Curtis values; sigma=their mean if given")
    ap.add_argument("--kernel", default="exponential", choices=["exponential", "gaussian"])
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--taxon-col", default="pred_label")
    ap.add_argument("--conf-col", default="calibrated_confidence")
    ap.add_argument("--indet-col", default="indeterminate")
    ap.add_argument("--plot", action="store_true")
    ap.add_argument("--out", default=str(OUT_DIR / "likelihood_ratio_sketch.json"))
    args = ap.parse_args()

    df = pd.read_csv(args.pred)
    for col in (args.group_col, args.taxon_col, args.conf_col, args.indet_col):
        if col not in df.columns:
            sys.exit(f"column '{col}' not in {Path(args.pred).name} (has {list(df.columns)})")
    groups = {g: sub for g, sub in df.groupby(args.group_col)}
    for name, who in [(args.q, "q"), (args.a, "a"), (args.b, "b")]:
        if name not in groups:
            sys.exit(f"group '{name}' (for {who}) not found. Available: {sorted(groups)}")

    sigma = args.sigma
    if args.calib_within and Path(args.calib_within).exists():
        vals = pd.read_csv(args.calib_within, header=None).to_numpy(dtype=float).ravel()
        vals = vals[np.isfinite(vals)]
        if len(vals):
            sigma = float(np.mean(vals))
            print(f"sigma estimated from {len(vals)} same-source dissimilarities: {sigma:.3f}")
    if sigma <= 0:
        sys.exit("sigma must be > 0")

    tq, vq = point_profile(groups[args.q], args.taxon_col, args.conf_col, args.indet_col)
    ta, va = point_profile(groups[args.a], args.taxon_col, args.conf_col, args.indet_col)
    tb, vb = point_profile(groups[args.b], args.taxon_col, args.conf_col, args.indet_col)
    d_qa = bc_between(tq, vq, ta, va)
    d_qb = bc_between(tq, vq, tb, vb)
    lr_point = lr_from_scores(d_qa, d_qb, sigma, args.kernel)

    bt = lambda g: bootstrap_profiles(groups[g], args.taxon_col, args.conf_col,
                                      args.indet_col, args.boot, args.seed)
    tq_, Dq = bt(args.q); ta_, Da = bt(args.a); tb_, Db = bt(args.b)
    lrs = np.empty(args.boot)
    for i in range(args.boot):
        da = bc_between(tq_, Dq[i], ta_, Da[i])
        db = bc_between(tq_, Dq[i], tb_, Db[i])
        lrs[i] = lr_from_scores(da, db, sigma, args.kernel)
    log10 = np.log10(np.clip(lrs, 1e-12, None))
    lr_med = float(np.median(lrs))
    lo, hi = float(np.percentile(lrs, 2.5)), float(np.percentile(lrs, 97.5))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    result = {
        "SKETCH_WARNING": "Illustrative score-based LR; sigma is assumed, not calibrated. "
                          "See Discussion.",
        "questioned": args.q, "hypothesis_A": args.a, "hypothesis_B": args.b,
        "kernel": args.kernel, "sigma": sigma, "sigma_source":
            "calib-within mean" if args.calib_within else "assumed",
        "d_Q_A": d_qa, "d_Q_B": d_qb,
        "LR_point": lr_point, "log10_LR_point": float(np.log10(max(lr_point, 1e-12))),
        "LR_median": lr_med, "LR_95CI": [lo, hi],
        "log10_LR_median": float(np.median(log10)),
        "favoured": "A (H1)" if lr_med >= 1 else "B (H2)",
        "verbal": verbal(lr_med), "n_boot": args.boot,
    }
    json.dump(result, open(args.out, "w"), indent=2)

    print("\n=== score-based LR SKETCH (illustrative) ===")
    print(f"Q = {args.q}   A (H1) = {args.a}   B (H2) = {args.b}")
    print(f"kernel {args.kernel} | sigma {sigma:.3f} ({result['sigma_source']})")
    print(f"d(Q,A) = {d_qa:.3f}   d(Q,B) = {d_qb:.3f}   (Bray-Curtis, identified pollen sum)")
    print(f"LR (point) = {lr_point:.3g}")
    print(f"LR (bootstrap median) = {lr_med:.3g}   95% CI [{lo:.3g}, {hi:.3g}]")
    print(f"interpretation: {result['verbal']}  (log10 LR median {result['log10_LR_median']:.2f})")
    print(f"\nwrote likelihood_ratio_sketch.json -> {OUT_DIR}")

    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(4.6, 3.2))
        ax.hist(log10, bins=40, color="#0072B2", alpha=0.85)
        ax.axvline(0, color="#999999", ls="--", lw=1, label="LR = 1 (no support)")
        ax.axvline(np.median(log10), color="#E69F00", lw=1.6,
                   label=f"median log10 LR = {np.median(log10):.2f}")
        ax.set_xlabel("log10 likelihood ratio"); ax.set_ylabel("bootstrap count")
        ax.legend(fontsize=7, frameon=False)
        ax.set_title("LR sketch (illustrative)", fontsize=9)
        fig.tight_layout(); fig.savefig(OUT_DIR / "lr_distribution.png", dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"wrote lr_distribution.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
