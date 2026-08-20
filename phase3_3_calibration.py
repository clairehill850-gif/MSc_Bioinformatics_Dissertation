#!/usr/bin/env python3
# What this does: adjusts the model's confidence, and sets "indeterminate" threshold.
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from phase0_determinism import enable
from phase2_1_1_dataset import PollenDataset, load_class_index
from phase2_1_2_transforms import get_val_transform

try:
    from torch.amp import autocast as _autocast
    def amp_ctx(enabled): return _autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast
    def amp_ctx(enabled): return _autocast(enabled=enabled)

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
COST_NPZ = EVAL_DIR / "confusion_cost_matrix.npz"

# Colourblind-friendly palette
C_BEFORE, C_AFTER, C_REF = "#E69F00", "#0072B2", "#999999"
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
                     "font.size": 9, "axes.linewidth": 0.8, "figure.dpi": 150})


@torch.no_grad()
def collect_logits(model, loader, device, use_amp):
# Get model's raw scores
    model.eval()
    logits_all, labels_all = [], []
    for imgs, labels in loader:
        imgs = imgs.to(device, non_blocking=True)
        with amp_ctx(use_amp):
            out = model(imgs)
        logits_all.append(out.float().cpu().numpy())
        labels_all.append(labels.numpy())
    return np.concatenate(logits_all), np.concatenate(labels_all)


def fit_temperature(logits, labels, max_iter=200):
# Temp value that calibrates confidence best
    lt = torch.tensor(logits, dtype=torch.float32)
    yt = torch.tensor(labels, dtype=torch.long)
    T = torch.nn.Parameter(torch.ones(1))
    opt = torch.optim.LBFGS([T], lr=0.01, max_iter=max_iter)
    nll = torch.nn.CrossEntropyLoss()

    def closure():
        opt.zero_grad()
        loss = nll(lt / T.clamp(min=1e-3), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(T.detach().clamp(min=1e-3).item())


def softmax_np(logits, T=1.0):
# Apply temp
    z = logits / T
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def calibration_bins(conf, correct, n_bins=15):
# Per bin accuracy and confidence
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    centres, accs, confs, ns = [], [], [], []
    ece, N = 0.0, len(conf)
    for b in range(n_bins):
        m = idx == b
        centres.append((edges[b] + edges[b + 1]) / 2)
        if m.sum() == 0:
            accs.append(np.nan); confs.append(np.nan); ns.append(0); continue
        a, c, n = correct[m].mean(), conf[m].mean(), int(m.sum())
        ece += (n / N) * abs(a - c)
        accs.append(a); confs.append(c); ns.append(n)
    return ece, np.array(centres), np.array(accs), np.array(confs), np.array(ns)


def reliability_diagram(conf0, conf1, correct, ece0, ece1, T, path, n_bins=15):
# Before and after reliability
    _, ctr, acc0, _, n0 = calibration_bins(conf0, correct, n_bins)
    _, _, acc1, _, n1 = calibration_bins(conf1, correct, n_bins)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.4))
    for ax, acc, ns, ece, ttl, col in [
        (axes[0], acc0, n0, ece0, f"Before (T=1)\nECE = {ece0:.3f}", C_BEFORE),
        (axes[1], acc1, n1, ece1, f"After (T={T:.2f})\nECE = {ece1:.3f}", C_AFTER)]:
        ax.plot([0, 1], [0, 1], "--", color=C_REF, lw=1, zorder=1)
        ok = ~np.isnan(acc)
        ax.bar(ctr[ok], acc[ok], width=1.0 / n_bins * 0.9, color=col, alpha=0.85, zorder=2,
               edgecolor="white", linewidth=0.4)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_xlabel("confidence"); ax.set_title(ttl, fontsize=9)
        ax.set_aspect("equal")
    axes[0].set_ylabel("accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def risk_coverage(conf, correct, error_cost=None, n_steps=101):
# Acc vs coverage curve
    taus = np.linspace(0.0, 1.0, n_steps)
    rows = []
    for t in taus:
        keep = conf >= t
        cov = float(keep.mean())
        if keep.sum() == 0:
            rows.append([t, 0.0, np.nan, np.nan, 0, np.nan]); continue
        acc = float(correct[keep].mean())
        ret_cost = float(error_cost[keep & ~correct].sum()) if error_cost is not None else np.nan
        rows.append([t, cov, acc, 1 - acc, int(keep.sum()), ret_cost])
    return pd.DataFrame(rows, columns=["threshold", "coverage", "sel_accuracy",
                                       "sel_error", "n_retained", "retained_cost"])


def pick_threshold(rc, target_acc, min_cov=0.5):
# Select lowest cutoff that meets acc and coverage scores
    ok = rc[(rc.sel_accuracy >= target_acc) & (rc.coverage >= min_cov)]
    if len(ok):
        return float(ok.iloc[ok.threshold.argmin()].threshold), True
    feas = rc[rc.coverage >= min_cov].dropna(subset=["sel_accuracy"])
    return (float(feas.iloc[feas.sel_accuracy.argmax()].threshold), False) if len(feas) else (0.5, False)


def accuracy_coverage_plot(rc, tau, target_acc, path):
# Plot acc vs coverage and draw threshold
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    v = rc.dropna(subset=["sel_accuracy"])
    ax.plot(v.coverage, v.sel_accuracy, color=C_AFTER, lw=1.8)
    ax.axhline(target_acc, ls="--", color=C_REF, lw=1, label=f"target acc {target_acc:.2f}")
    row = rc.iloc[(rc.threshold - tau).abs().argmin()]
    ax.scatter([row.coverage], [row.sel_accuracy], color=C_BEFORE, zorder=5, s=35,
               label=f"tau={tau:.2f} (cov {row.coverage:.2f}, acc {row.sel_accuracy:.3f})")
    ax.set_xlabel("coverage (fraction of grains retained)")
    ax.set_ylabel("selective accuracy")
    ax.set_xlim(0, 1.02); ax.legend(fontsize=7, loc="lower left", frameon=False)
    fig.tight_layout(); fig.savefig(path, dpi=300, bbox_inches="tight"); plt.close(fig)


def load_cost_matrix(class_index):
# Load the cost matrix
    if not COST_NPZ.exists():
        return None
    z = np.load(COST_NPZ, allow_pickle=True)
    C = z["cost"].astype(np.float32)
    labels = [str(x) for x in z["labels"]]
    order = [None] * len(class_index)
    for lab, i in class_index.items():
        order[int(i)] = str(lab)
    if labels == order:
        return C
    pos = {l: k for k, l in enumerate(labels)}
    n = len(order); out = np.ones((n, n), np.float32); np.fill_diagonal(out, 0.0)
    for a, la in enumerate(order):
        for b, lb in enumerate(order):
            if la in pos and lb in pos:
                out[a, b] = C[pos[la], pos[lb]]
    return out


def main():
# Calibrate confidence, draw the charts, choose the cut-off, and save calibrated predictions
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(CKPT_DIR / "best.pt"))
    ap.add_argument("--target-acc", type=float, default=0.95)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--num-workers", type=int, default=4)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable()
# Run at full precision
    use_amp = False
    class_index = load_class_index()
    idx_to_label = {int(i): l for l, i in class_index.items()}
    n_classes = len(class_index)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        sys.exit(f"checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    arch = state.get("arch") or state.get("config", {}).get("model", "resnet50")
    model = timm.create_model(arch, pretrained=False, num_classes=n_classes).to(device)
    model.load_state_dict(state["model_state"])
    print(f"calibrating {ckpt_path.name} (arch={arch}) on {device}")

    tf = get_val_transform()
    val_loader = DataLoader(PollenDataset("val", class_index, transform=tf),
                            batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
    test_ds = PollenDataset("test", class_index, transform=tf)
    test_loader = DataLoader(test_ds,
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=args.num_workers, pin_memory=True)

    print("collecting logits...")
    val_logits, val_labels = collect_logits(model, val_loader, device, use_amp)
    test_logits, test_labels = collect_logits(model, test_loader, device, use_amp)

# Find the temp
    T = fit_temperature(val_logits, val_labels)
    print(f"fitted temperature T = {T:.3f}  ({'over' if T > 1 else 'under'}-confident baseline)")

# Reliability chart
    pred = test_logits.argmax(1)
    correct = (pred == test_labels)
    conf0 = softmax_np(test_logits, 1.0).max(1)
    conf1 = softmax_np(test_logits, T).max(1)
    ece0, *_ = calibration_bins(conf0, correct)
    ece1, *_ = calibration_bins(conf1, correct)
    print(f"test top-1 (unchanged by T): {correct.mean():.4f} | ECE {ece0:.3f} -> {ece1:.3f}")
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    reliability_diagram(conf0, conf1, correct, ece0, ece1, T, EVAL_DIR / "reliability_diagram.png")

# Apply indeterminate cutoff
    C = load_cost_matrix(class_index)
    err_cost = None
    if C is not None:
        err_cost = np.where(correct, 0.0, C[test_labels, pred])
    rc = risk_coverage(conf1, correct, err_cost)
    rc.to_csv(EVAL_DIR / "threshold_analysis.csv", index=False)
    tau, reached = pick_threshold(rc, args.target_acc)
    row = rc.iloc[(rc.threshold - tau).abs().argmin()]
    accuracy_coverage_plot(rc, tau, args.target_acc, EVAL_DIR / "accuracy_coverage.png")
    note = "" if reached else " (target acc NOT reachable at >=50% coverage; reporting best feasible)"
    print(f"operating threshold tau = {tau:.2f}{note}: "
          f"coverage {row.coverage:.3f}, selective accuracy {row.sel_accuracy:.4f}")

# Cutoff survival
    hi_summary = {}
    if C is not None:
        is_err = ~correct
        hi = is_err & (C[test_labels, pred] >= 2.0)
        surv = hi & (conf1 >= tau)
        hi_summary = {"high_cost_errors": int(hi.sum()),
                      "high_cost_surviving_threshold": int(surv.sum()),
                      "high_cost_caught": int((hi & (conf1 < tau)).sum())}
        print(f"high-cost errors: {hi_summary['high_cost_errors']} total | "
              f"{hi_summary['high_cost_caught']} sent to indeterminate | "
              f"{hi_summary['high_cost_surviving_threshold']} still confidently reported")

    paths = test_ds.df["processed_path"].astype(str).to_numpy()
    if len(paths) != len(test_labels):
        sys.exit(f"alignment error: {len(paths)} dataset rows vs {len(test_labels)} predictions")
    df_labels = test_ds.df["final_label"].astype(str).to_numpy()
    rec_labels = np.array([idx_to_label[i] for i in test_labels])
    match = (df_labels == rec_labels).mean()
    if match < 0.999:
        sys.exit(f"alignment error: dataset order != prediction order (label match {match:.3f}); "
                 "processed_path key would be wrong.")
    pd.DataFrame({
        "processed_path": paths,
        "true_label": [idx_to_label[i] for i in test_labels],
        "pred_label": [idx_to_label[i] for i in pred],
        "calibrated_confidence": conf1,
        "indeterminate": conf1 < tau,
        "correct": correct,
    }).to_csv(EVAL_DIR / "calibrated_test_predictions.csv", index=False)

    json.dump({"checkpoint": ckpt_path.name, "arch": arch, "temperature": T,
               "ece_before": ece0, "ece_after": ece1, "test_top1": float(correct.mean()),
               "target_accuracy": args.target_acc, "operating_threshold": tau,
               "threshold_reached_target": bool(reached),
               "coverage_at_tau": float(row.coverage),
               "selective_accuracy_at_tau": float(row.sel_accuracy), **hi_summary},
              open(EVAL_DIR / "temperature.json", "w"), indent=2)
    print(f"\nwrote temperature.json, reliability_diagram.png, accuracy_coverage.png,\n"
          f"      threshold_analysis.csv, calibrated_test_predictions.csv -> {EVAL_DIR}")


if __name__ == "__main__":
    main()
