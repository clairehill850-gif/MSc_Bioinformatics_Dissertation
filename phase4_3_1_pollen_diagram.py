#!/usr/bin/env python3
# What this does: Assemblage fig.
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

try:
    from phase4_1_1_assemblage import assemblage_profile, find_family_map, UNKNOWN
except Exception as e:
    sys.exit(f"could not import phase4_1_1_assemblage ({e}); keep the phase4_* scripts together.")

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
OUT_DIR = PROJECT_ROOT / "outputs" / "assemblage"
DEFAULT_PROFILE = OUT_DIR / "assemblage_profile.csv"
OTHER = "Other (rare taxa)"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.8})


def load_profile(args):
    fam_map = find_family_map(args.hierarchy)
    if args.pred:
        df = pd.read_csv(args.pred)
        if args.group_col not in df.columns:
            sys.exit(f"group column '{args.group_col}' not in predictions")
        sub = df[df[args.group_col].astype(str) == str(args.sample)]
        if len(sub) == 0:
            sys.exit(f"sample '{args.sample}' not found in column '{args.group_col}'")
        prof = assemblage_profile(sub, args.taxon_col, args.conf_col, args.indet_col, fam_map)
        val = "pct_cw"
    else:
        ppath = Path(args.profile)
        if not ppath.exists():
            sys.exit(f"profile not found: {ppath}")
        prof = pd.read_csv(ppath, keep_default_na=False)
        val = args.value if args.value in prof.columns else \
            ("pct_cw" if "pct_cw" in prof.columns else "pct_count")
        if "family" not in prof.columns:
            prof["family"] = "unassigned"
    prof = prof.copy()
    prior_fam = prof["family"] if "family" in prof.columns else ["unassigned"] * len(prof)
    prof["family"] = [
        UNKNOWN if t == UNKNOWN
        else fam_map.get(str(t), f if (f and str(f) not in ("nan", "NA", "")) else "unassigned")
        for t, f in zip(prof["taxon"], prior_fam)]
    prof["pct"] = pd.to_numeric(prof[val], errors="coerce").fillna(0.0)
    return prof[["taxon", "family", "pct"]], val


def family_colours(families):
# Different colours
    base = (plt.cm.tab20.colors + plt.cm.tab20b.colors + plt.cm.tab20c.colors)
    fams = [f for f in families if f != UNKNOWN]
    return {f: base[i % len(base)] for i, f in enumerate(sorted(set(fams)))}


def main():
# Group rares
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", default=str(DEFAULT_PROFILE))
    ap.add_argument("--pred", default=None)
    ap.add_argument("--group-col", default=None)
    ap.add_argument("--sample", default=None)
    ap.add_argument("--value", default="pct_cw")
    ap.add_argument("--min-pct", type=float, default=1.0,
                    help="taxa below this %% (of pollen sum) are pooled into 'Other'")
    ap.add_argument("--no-pollen-sum", action="store_true",
                    help="plot %% of all grains instead of %% of identified pollen sum")
    ap.add_argument("--keep-unknown", action="store_true", help="show the unknown bin as a bar")
    ap.add_argument("--annot", action="store_true", help="write the % at the end of each bar")
    ap.add_argument("--uncertainty", default=None,
                    help="4.1.2 CSV (taxon, lo95_pct, hi95_pct) for CI whiskers")
    ap.add_argument("--hierarchy", default=None)
    ap.add_argument("--taxon-col", default="pred_label")
    ap.add_argument("--conf-col", default="calibrated_confidence")
    ap.add_argument("--indet-col", default="indeterminate")
    ap.add_argument("--title", default="Assemblage pollen diagram")
    ap.add_argument("--out", default=str(OUT_DIR / "pollen_diagram.png"))
    args = ap.parse_args()
    if args.pred and not (args.group_col and args.sample):
        sys.exit("--pred requires --group-col and --sample")

    prof, val = load_profile(args)
    unk = float(prof.loc[prof.taxon == UNKNOWN, "pct"].sum())
    ident = prof[prof.taxon != UNKNOWN].copy()

    pollen_sum = not args.no_pollen_sum
    tot = ident["pct"].sum()
    scale = 100.0 / tot if (pollen_sum and tot > 0) else 1.0
    ident["plot_pct"] = ident["pct"] * scale
    xlabel = "% of identified pollen sum" if pollen_sum else "% of all grains"

    ci = {}
    if args.uncertainty and Path(args.uncertainty).exists():
        u = pd.read_csv(args.uncertainty, keep_default_na=False)
        if {"taxon", "lo95_pct", "hi95_pct"}.issubset(u.columns):
            for _, r in u.iterrows():
                ci[str(r["taxon"])] = (float(r["lo95_pct"]) * scale, float(r["hi95_pct"]) * scale)

    keep = ident[ident["plot_pct"] >= args.min_pct].copy()
    rare = ident[ident["plot_pct"] < args.min_pct]
    other_pct, n_rare = float(rare["plot_pct"].sum()), len(rare)
    keep = keep.sort_values(["family", "plot_pct"], ascending=[True, False]).reset_index(drop=True)

    rows = [(r.taxon, r.family, r.plot_pct) for r in keep.itertuples(index=False)]
    if n_rare:
        rows.append((OTHER, OTHER, other_pct))
    if args.keep_unknown and unk > 0:
        rows.append((UNKNOWN, UNKNOWN, unk))

    cmap = family_colours(list(keep["family"].unique()))
    cmap[OTHER] = (0.6, 0.6, 0.6); cmap[UNKNOWN] = (0.4, 0.4, 0.4)

    labels = [r[0] for r in rows]
    vals = [r[2] for r in rows]
    fams = [r[1] for r in rows]
    colours = [cmap.get(f, (0.6, 0.6, 0.6)) for f in fams]
    err = []
    for t, p in zip(labels, vals):
        lohi = ci.get(t)
        err.append((max(p - lohi[0], 0.0), max(lohi[1] - p, 0.0)) if lohi else (0.0, 0.0))

    n = len(labels)
    y = np.arange(n)
    fig, ax = plt.subplots(figsize=(7.2, max(3.0, 0.34 * n + 1.4)))
    xerr = np.array(err).T if any(e != (0.0, 0.0) for e in err) else None
    ax.barh(y, vals, color=colours, edgecolor="white", linewidth=0.4,
            xerr=xerr,
            error_kw=dict(ecolor="#333333", elinewidth=0.8, capsize=2) if xerr is not None else {})
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel(xlabel)

# Whiskers
    ends = [v + e[1] for v, e in zip(vals, err)]
    xmax = max(ends) if ends else 1.0
    ax.set_xlim(0, xmax * (1.22 if args.annot else 1.15))

# Percentage at end of whisker
    if args.annot:
        pad = xmax * 0.012
        for yi, v, e in zip(y, vals, err):
            ax.text(v + e[1] + pad, yi, f"{v:.1f}", va="center", ha="left",
                    fontsize=6.5, color="#333333")
    ax.set_title(args.title, fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.invert_yaxis()

    legend_fams = [f for f in sorted(set(fams)) if f not in (OTHER, UNKNOWN)]
    handles = [Patch(color=cmap[f], label=f) for f in legend_fams]
    if n_rare:
        handles.append(Patch(color=cmap[OTHER], label=f"Other ({n_rare} taxa <{args.min_pct:g}%)"))
    if handles:
        ax.legend(handles=handles, fontsize=6.5, frameon=False, loc="center left",
                  bbox_to_anchor=(1.01, 0.5), title="Family", title_fontsize=7)
    note = f"unknown {unk:.1f}%" + (" (excluded from pollen sum)" if pollen_sum and not args.keep_unknown else "")
    ax.annotate(note, xy=(1.0, -0.07), xycoords="axes fraction", ha="right",
                fontsize=6.5, color="#555555")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"taxa shown: {len(keep)} (+{n_rare} rare pooled into Other) | "
          f"families: {len(legend_fams)} | unknown {unk:.1f}%")
    print(f"axis: {xlabel}" + (" | CI whiskers from 4.1.2" if ci else ""))
    print(f"wrote pollen_diagram.png -> {OUT_DIR}")


if __name__ == "__main__":
    main()
