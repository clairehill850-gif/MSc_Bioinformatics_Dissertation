#!/usr/bin/env python3
# What this does: generates dissimilarity heatmap. ALASKA
import os
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram, fcluster
from scipy.spatial.distance import squareform

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
OUT_DIR = PROJECT_ROOT / "outputs" / "assemblage"
DEFAULT_MATRIX = OUT_DIR / "bray_curtis_matrix.csv"

plt.rcParams.update({"font.family": "sans-serif",
                     "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 8, "axes.linewidth": 0.8})


def load_matrix(path):
    bc = pd.read_csv(path, index_col=0)
    bc.columns = [str(c) for c in bc.columns]
    bc.index = [str(i) for i in bc.index]
    if list(bc.index) != list(bc.columns):
        common = [c for c in bc.index if c in bc.columns]
        bc = bc.loc[common, common]
    M = bc.to_numpy(dtype=float)
    M = (M + M.T) / 2.0
    np.fill_diagonal(M, 0.0)
    M = np.clip(M, 0.0, 1.0)
    return bc.index.tolist(), M

def make_heatmap(labels, M, method, cut, out_png, annot):
    n = len(labels)
    condensed = squareform(M, checks=False)
    Z = linkage(condensed, method=method)
    order = dendrogram(Z, no_plot=True)["leaves"]
    Mo = M[np.ix_(order, order)]
    lab_o = [labels[i] for i in order]

    fig = plt.figure(figsize=(max(6.0, 0.55 * n + 2.5), max(6.2, 0.55 * n + 2.8)))
    gs = fig.add_gridspec(2, 2, width_ratios=[20, 1], height_ratios=[1, 5],
                          hspace=0.03, wspace=0.04)
    ax_d = fig.add_subplot(gs[0, 0])
    ax_h = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])
    dendrogram(Z, ax=ax_d, color_threshold=cut, above_threshold_color="#444444",
               no_labels=True)
    ax_d.set_xticks([]); ax_d.set_yticks([])
    for s in ax_d.spines.values():
        s.set_visible(False)
# The height at which the tree is cut into flat groups
    ax_d.axhline(cut, ls="--", lw=0.8, color="#B22222")
    im = ax_h.imshow(Mo, extent=[0, 10 * n, 10 * n, 0], aspect="auto",
                     cmap="viridis", vmin=0.0, vmax=1.0)
    ticks = [10 * k + 5 for k in range(n)]
    ax_h.set_xticks(ticks); ax_h.set_yticks(ticks)
    ax_h.set_xticklabels(lab_o, rotation=45, ha="right", fontsize=7)
    ax_h.set_yticklabels(lab_o, fontsize=7)
    ax_d.set_xlim(0, 10 * n); ax_h.set_xlim(0, 10 * n)
    if annot:
        for a in range(n):
            for b in range(n):
                v = Mo[a, b]
                ax_h.text(10 * b + 5, 10 * a + 5, f"{v:.2f}", ha="center", va="center",
                          fontsize=6, color="white" if v < 0.6 else "black")
    cb = fig.colorbar(im, cax=ax_c)
    cb.set_label("Bray-Curtis dissimilarity", fontsize=8)
    ax_d.set_title("Assemblage dissimilarity (UPGMA-clustered)", fontsize=9)

    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return Z, order, lab_o

def main():
# Draw figure
    ap = argparse.ArgumentParser()
    ap.add_argument("--matrix", default=str(DEFAULT_MATRIX))
    ap.add_argument("--method", default="average",
                    help="linkage method (average=UPGMA, ward, complete, single)")
    ap.add_argument("--cut", type=float, default=0.8,
                    help="dissimilarity height for flat cluster assignment")
    ap.add_argument("--annot", action="store_true", help="write the value in each cell")
    ap.add_argument("--out", default=str(OUT_DIR / "dissimilarity_heatmap.png"))
    args = ap.parse_args()


    mpath = Path(args.matrix)
    if not mpath.exists():
        sys.exit(f"matrix not found: {mpath}\nRun phase4_2_1_bray_curtis.py first.")
    labels, M = load_matrix(mpath)
    if len(labels) < 2:
        sys.exit(f"need >=2 samples, found {len(labels)}.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    Z, order, lab_o = make_heatmap(labels, M, args.method, args.cut, args.out, args.annot)

    flat = fcluster(Z, t=args.cut, criterion="distance")
    assign = (pd.DataFrame({"sample": labels, "cluster": flat})
              .sort_values(["cluster", "sample"]).reset_index(drop=True))
    assign.to_csv(OUT_DIR / "cluster_assignments.csv", index=False)

    print(f"{len(labels)} samples | {args.method} linkage | cut {args.cut}")
    print("clustered order:", " -> ".join(lab_o))
    print(f"\nflat clusters at height {args.cut}:")
    for cl in sorted(assign.cluster.unique()):
        members = assign.loc[assign.cluster == cl, "sample"].tolist()
        print(f"  cluster {cl}: {', '.join(members)}")
    print(f"\nwrote dissimilarity_heatmap.png + cluster_assignments.csv -> {OUT_DIR}")


if __name__ == "__main__":
    main()
