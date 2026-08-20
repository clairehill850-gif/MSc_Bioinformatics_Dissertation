#!/usr/bin/env python3
# What this does: deterministic re-scoring for every saved checkpoint on the test set.

# CUBLAS_WORKSPACE_CONFIG
import os
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import sys
import shutil
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from phase2_1_1_dataset import PollenDataset, load_class_index
from phase2_1_2_transforms import get_val_transform
from phase3_1_3_train_costsensitive import load_cost_matrix, cost_metrics

try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
BASELINE_CKPT = CKPT_DIR / "best.pt"

ARCH_CSV = EVAL_DIR / "architecture_comparison.csv"
COST_CSV = EVAL_DIR / "cost_sensitive_comparison.csv"
REPORT_CSV = EVAL_DIR / "deterministic_eval_report.csv"

ARCH_COLS = ["arch", "params_M", "top1", "top5", "macro_f1", "weighted_f1",
             "total_cost", "high_cost_errors", "infer_ms_per_img"]
COST_COLS = ["model", "checkpoint", "top1", "top5", "macro_f1", "weighted_f1",
             "errors", "total_cost", "mean_cost_per_error", "high_cost_errors",
             "same_family_errors", "neutral_errors"]
REPORT_COLS = ["top1", "top5", "macro_f1", "weighted_f1",
               "errors", "total_cost", "high_cost_errors"]

CFG = {"batch_size": 64, "num_workers": 4, "seed": 42}


def set_seed(seed):
# Seed for reproducibility.
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def set_deterministic():
# Determinism
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


@torch.no_grad()
def evaluate_test(model, loader, device, C):
    model.eval()
    top1 = top5 = n = 0
    yp_all, yt_all = [], []
    for imgs, labels in tqdm(loader, desc="test", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(imgs).float()
        _, pred5 = logits.topk(5, dim=1)
        top1 += (pred5[:, 0] == labels).sum().item()
        top5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)
        yp_all.append(pred5[:, 0].cpu().numpy()); yt_all.append(labels.cpu().numpy())
    yp = np.concatenate(yp_all); yt = np.concatenate(yt_all)
    out = {"top1": top1 / n, "top5": top5 / n,
           "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
           "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0)}
    out.update(cost_metrics(yt, yp, C))
    return out


def discover_checkpoints():
    arch_targets, cost_targets = [], []
    if BASELINE_CKPT.exists():
        arch_targets.append(("resnet50", BASELINE_CKPT))
        cost_targets.append(("baseline", BASELINE_CKPT))
    for p in sorted(CKPT_DIR.glob("best_*.pt")):
        if p.name.startswith("best_cost"):
            cost_targets.append((None, p))
        else:
            arch_targets.append((None, p))
    return arch_targets, cost_targets


def score_checkpoints(targets, class_index, C, device, cache):
# Score each checkpoint once
    n_classes = len(class_index)
    loader = DataLoader(PollenDataset("test", class_index, transform=get_val_transform()),
                        batch_size=CFG["batch_size"], shuffle=False,
                        num_workers=CFG["num_workers"], pin_memory=True)
    rows = []
    for name, ckpt_path in targets:
        key = str(ckpt_path.resolve())
        if key not in cache:
            if not ckpt_path.exists():
                print(f"  ({ckpt_path.name}: not found - skipping)")
                continue
            state = torch.load(ckpt_path, map_location=device, weights_only=False)
            arch = state.get("arch") or state.get("config", {}).get("model", "?")
            model = timm.create_model(arch, pretrained=False, num_classes=n_classes).to(device)
            model.load_state_dict(state["model_state"])
            m = evaluate_test(model, loader, device, C)
            m["arch_name"] = arch
            m["params_M"] = round(sum(p.numel() for p in model.parameters()) / 1e6, 1)
            m["cost_lambda"] = state.get("config", {}).get("cost_lambda", None)
            cache[key] = m
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            print(f"  {ckpt_path.name:22s} top1 {m['top1']:.4f} macroF1 {m['macro_f1']:.4f} "
                  f"| total_cost {m['total_cost']:.1f} high-cost errs {m['high_cost_errors']}")
        else:
            print(f"  {ckpt_path.name:22s} (already scored - reusing)")
        rows.append((name, ckpt_path, cache[key]))
    return rows


def build_arch_table(rows, old):
# Rebuild architecture_comparison.csv
    prev = {}
    if old is not None and "arch" in old.columns and "infer_ms_per_img" in old.columns:
        prev = dict(zip(old["arch"].astype(str), old["infer_ms_per_img"]))
    out = []
    for name, ckpt_path, m in rows:
        arch = name or m["arch_name"]
        out.append({"arch": arch, "params_M": m["params_M"],
                    "top1": m["top1"], "top5": m["top5"],
                    "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                    "total_cost": m["total_cost"], "high_cost_errors": m["high_cost_errors"],
                    "infer_ms_per_img": prev.get(arch, np.nan)})
    return pd.DataFrame(out)[ARCH_COLS].sort_values("macro_f1", ascending=False)


def build_cost_table(rows):
# Rebuild cost_sensitive_comparison.csv
    out = []
    for name, ckpt_path, m in rows:
        label = name
        if label is None:
            lam = m["cost_lambda"] if m["cost_lambda"] is not None else "?"
            label = f"cost_lambda={lam}"
        out.append({"model": label, "checkpoint": ckpt_path.name,
                    "top1": m["top1"], "top5": m["top5"],
                    "macro_f1": m["macro_f1"], "weighted_f1": m["weighted_f1"],
                    "errors": m["errors"], "total_cost": m["total_cost"],
                    "mean_cost_per_error": m["mean_cost_per_error"],
                    "high_cost_errors": m["high_cost_errors"],
                    "same_family_errors": m["same_family_errors"],
                    "neutral_errors": m["neutral_errors"]})
    return pd.DataFrame(out)[COST_COLS]


def build_report(old, new, key, table_name):
    rows = []
    old_idx = {} if old is None or key not in old.columns else \
        {str(k): v for k, v in old.set_index(key).to_dict("index").items()}
    for _, r in new.iterrows():
        k = str(r[key])
        prev = old_idx.get(k)
        for col in REPORT_COLS:
            if col not in new.columns:
                continue
            new_v = r[col]
            old_v = prev.get(col) if prev else None
            if old_v is None or pd.isna(old_v):
                delta = np.nan
            else:
                delta = float(new_v) - float(old_v)
            rows.append({"table": table_name, "row": k, "metric": col,
                         "old": old_v if old_v is not None else np.nan,
                         "new": new_v, "delta": delta,
                         "changed": bool(not pd.isna(delta) and abs(delta) > 1e-9)})
    return pd.DataFrame(rows)


def read_old(path):
    return pd.read_csv(path) if path.exists() else None


def backup(path):
    if not path.exists():
        return None
    bak = path.with_name(path.stem + ".pre_deterministic.csv")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"  backed up {path.name} -> {bak.name}")
    return bak


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    dry_run = "--dry-run" in flags

    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(CFG["seed"])
    set_deterministic()
    print(f"device: {device} | deterministic fp32 eval | seed {CFG['seed']}")
    if device == "cpu":
        print("warning: running on CPU - results are deterministic but will not match GPU scoring")

    class_index = load_class_index()
    C = load_cost_matrix(class_index)
    arch_targets, cost_targets = discover_checkpoints()
    if not arch_targets and not cost_targets:
        sys.exit(f"no checkpoints found under {CKPT_DIR}")

# Cache
    cache = {}
    print("\nscoring architecture checkpoints:")
    arch_rows = score_checkpoints(arch_targets, class_index, C, device, cache)
    print("\nscoring cost-sensitive checkpoints:")
    cost_rows = score_checkpoints(cost_targets, class_index, C, device, cache)

    old_arch, old_cost = read_old(ARCH_CSV), read_old(COST_CSV)
    new_arch = build_arch_table(arch_rows, old_arch)
    new_cost = build_cost_table(cost_rows)

    report = pd.concat([build_report(old_arch, new_arch, "arch", "architecture_comparison"),
                        build_report(old_cost, new_cost, "model", "cost_sensitive_comparison")],
                       ignore_index=True)

# Baseline
    base_arch = new_arch.loc[new_arch["arch"] == "resnet50"]
    base_cost = new_cost.loc[new_cost["model"] == "baseline"]
    if len(base_arch) and len(base_cost):
        agree = all(np.isclose(float(base_arch.iloc[0][c]), float(base_cost.iloc[0][c]))
                    for c in ("top1", "top5", "macro_f1", "total_cost", "high_cost_errors"))
        print(f"\nbaseline agreement across both tables: {'OK' if agree else 'STILL DIFFERS'}")
        if not agree:
            print("  the two tables still disagree - the cause is not inference nondeterminism alone")

    changed = report.loc[report["changed"]]
    print(f"\n{len(changed)} of {len(report)} reported metrics moved against the previous CSVs")
    if len(changed):
        print(changed.to_string(index=False))

    if dry_run:
        print("\n--dry-run: nothing written")
        return

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    backup(ARCH_CSV); backup(COST_CSV)
    new_arch.to_csv(ARCH_CSV, index=False)
    new_cost.to_csv(COST_CSV, index=False)
    report.to_csv(REPORT_CSV, index=False)
    print(f"\nwrote {ARCH_CSV.name}, {COST_CSV.name}, {REPORT_CSV.name} -> {EVAL_DIR}")


if __name__ == "__main__":
    main()
