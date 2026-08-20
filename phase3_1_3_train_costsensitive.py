#!/usr/bin/env python3
# What this does: retrains the classifier with a cost-sensitive lossand compares it against the baseline.
import os
import sys
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm
from sklearn.metrics import f1_score

from phase2_1_1_dataset import PollenDataset, load_class_index
from phase2_1_2_transforms import get_train_transform, get_val_transform, AUG_CONFIG

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

# Mixed-precision setup
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_scaler(enabled): return _GradScaler("cuda", enabled=enabled)
    def amp_ctx(enabled): return _autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_scaler(enabled): return _GradScaler(enabled=enabled)
    def amp_ctx(enabled): return _autocast(enabled=enabled)

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
COST_NPZ = EVAL_DIR / "confusion_cost_matrix.npz"
BASELINE_CKPT = CKPT_DIR / "best.pt"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "config_cost.yaml"


def ckpt_names(cost_lambda):
# Checkpoint names include the penalty weight
    tag = f"l{float(cost_lambda):g}"
    return CKPT_DIR / f"best_cost_{tag}.pt", CKPT_DIR / f"last_cost_{tag}.pt"

DEFAULTS = {
    "model": "resnet50",
# Start from the trained best.pt, or from scratch
    "init": "warmstart",       
    "ft_epochs": 12,        
    "ft_lr": 5e-5,             
    "weight_decay": 1e-4,
    "batch_size": 64,
    "num_workers": 4,
    "seed": 42,
    "use_amp": True,
    "label_smoothing": 0.0,
    "cost_lambda": 1.0,        
    "resume": True,
    "wandb": True,
    "wandb_project": "pollen-classifier",
    "run_name": "resnet50_costsensitive",
}


# Loss function
class CostSensitiveLoss(nn.Module):

    def __init__(self, cost_matrix, cost_lambda=1.0, label_smoothing=0.0):
        super().__init__()
        self.register_buffer("C", torch.as_tensor(cost_matrix, dtype=torch.float32))
        self.cost_lambda = float(cost_lambda)
        self.label_smoothing = float(label_smoothing)

    def forward(self, logits, target):
        ce = F.cross_entropy(logits, target, label_smoothing=self.label_smoothing)
        if self.cost_lambda == 0.0:
            return ce
# Full precision
        p = logits.float().softmax(dim=1)           
#Cost
        cost_rows = self.C.index_select(0, target)   
        expected_cost = (p * cost_rows).sum(dim=1).mean()
        return ce + self.cost_lambda * expected_cost


def load_cost_matrix(class_index):
# Return the cost matrix as a grid
    if not COST_NPZ.exists():
        sys.exit(f"cost matrix not found: {COST_NPZ}\nRun phase3_1_2_cost_matrix.py first.")
    z = np.load(COST_NPZ, allow_pickle=True)
    C = z["cost"].astype(np.float32)
    labels = [str(x) for x in z["labels"]]
    order = [None] * len(class_index)
    for lab, i in class_index.items():
        order[int(i)] = str(lab)
    if labels == order:
        return C
    pos = {l: k for k, l in enumerate(labels)}                 
    n = len(order)
    out = np.ones((n, n), dtype=np.float32); np.fill_diagonal(out, 0.0)
    for a, la in enumerate(order):
        for b, lb in enumerate(order):
            if la in pos and lb in pos:
                out[a, b] = C[pos[la], pos[lb]]
    return out


# Settings and helpers
def load_config(path):
# Load config_cost.yaml
    cfg = dict(DEFAULTS)
    if Path(path).exists():
        with open(path) as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(DEFAULTS, f, sort_keys=False)
        print(f"wrote default cost config -> {path}")
    return cfg


def set_seed(seed):
# Fix all random seeds
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loader(split, cfg, class_index, train):
# Build data loader
    tf = get_train_transform() if train else get_val_transform()
    ds = PollenDataset(split, class_index, transform=tf)
    return DataLoader(ds, batch_size=cfg["batch_size"], shuffle=train,
                      num_workers=cfg["num_workers"], pin_memory=True, drop_last=train)


# Testing
@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp, C=None):
# Standard scores
    model.eval()
    loss_sum, n, top1, top5 = 0.0, 0, 0, 0
    yp_all, yt_all = [], []
    for imgs, labels in tqdm(loader, desc="eval", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx(use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0); n += labels.size(0)
        _, pred5 = logits.topk(5, dim=1)
        top1 += (pred5[:, 0] == labels).sum().item()
        top5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        yp_all.append(pred5[:, 0].cpu().numpy()); yt_all.append(labels.cpu().numpy())
    yp = np.concatenate(yp_all); yt = np.concatenate(yt_all)
    out = {"loss": loss_sum / n, "top1": top1 / n, "top5": top5 / n,
           "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
           "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0)}
    if C is not None:
        out.update(cost_metrics(yt, yp, C))
    return out


def cost_metrics(yt, yp, C, high=2.0, low=0.2):
# Forensic-cost summary
    wrong = yt != yp
    costs = C[yt[wrong], yp[wrong]] if wrong.any() else np.array([])
    return {"errors": int(wrong.sum()),
            "total_cost": float(costs.sum()),
            "mean_cost_per_error": float(costs.mean()) if costs.size else 0.0,
            "high_cost_errors": int((costs >= high).sum()),
            "same_family_errors": int((costs <= low).sum()),
            "neutral_errors": int(((costs > low) & (costs < high)).sum())}


# Training
def train_one_epoch(model, loader, device, criterion, optimizer, scaler, use_amp):
    model.train()
    loss_sum, n = 0.0, 0
    for imgs, labels in tqdm(loader, desc="train", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        with amp_ctx(use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer); scaler.update()
        loss_sum += loss.item() * labels.size(0); n += labels.size(0)
    return loss_sum / n


def run_comparison(cfg, class_index, device, use_amp, C):
    import glob
    n_classes = len(class_index)
    test_loader = build_loader("test", cfg, class_index, train=False)
    plain = nn.CrossEntropyLoss()

    targets = [("baseline", BASELINE_CKPT)]
    for p in sorted(glob.glob(str(CKPT_DIR / "best_cost*.pt"))):
        targets.append((None, Path(p)))      

    rows = []
    for name, ckpt_path in targets:
        if not ckpt_path.exists():
            print(f"  ({name or ckpt_path.name}: not found - skipping)"); continue
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        if name is None:
            lam = state.get("config", {}).get("cost_lambda", "?")
            name = f"cost_lambda={lam}"
        model = timm.create_model(cfg["model"], pretrained=False, num_classes=n_classes).to(device)
        model.load_state_dict(state["model_state"])
        m = evaluate(model, test_loader, device, plain, use_amp, C=C)
        m["model"] = name; m["checkpoint"] = ckpt_path.name
        rows.append(m)
        print(f"  {name:18s} top1 {m['top1']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"wF1 {m['weighted_f1']:.4f} | total_cost {m['total_cost']:.1f} "
              f"high-cost errs {m['high_cost_errors']}")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if rows:
        cols = ["model", "checkpoint", "top1", "top5", "macro_f1", "weighted_f1",
                "errors", "total_cost", "mean_cost_per_error", "high_cost_errors",
                "same_family_errors", "neutral_errors"]
        df = pd.DataFrame(rows)[cols]
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(EVAL_DIR / "cost_sensitive_comparison.csv", index=False)
        print(f"\nwrote cost_sensitive_comparison.csv ({len(df)} models) -> {EVAL_DIR}")
        base = df[df.model == "baseline"]
        if len(base):
            b = base.iloc[0]
            for _, c in df[df.model != "baseline"].iterrows():
                d_cost = (c["total_cost"] - b["total_cost"]) / max(b["total_cost"], 1e-9) * 100
                print(f"  {c['model']} vs baseline: total cost {d_cost:+.1f}%, "
                      f"top-1 {(c['top1']-b['top1'])*100:+.2f} pts, "
                      f"high-cost errors {int(b['high_cost_errors'])} -> {int(c['high_cost_errors'])}")
    return rows


def main():
# Fine-tune with cost-sensitive loss
    eval_only = "--eval-only" in sys.argv
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg = load_config(args[0] if args else CONFIG_PATH)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg["use_amp"]) and device == "cuda"
    torch.backends.cudnn.benchmark = True
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | AMP: {use_amp} | cost_lambda: {cfg['cost_lambda']}")

    class_index = load_class_index()
    n_classes = len(class_index)
    C = load_cost_matrix(class_index)
    assert C.shape == (n_classes, n_classes), f"cost matrix {C.shape} != ({n_classes},{n_classes})"
    C_t = torch.as_tensor(C, device=device)

    if eval_only:
        print("eval-only: comparing existing checkpoints on the test set\n")
        run_comparison(cfg, class_index, device, use_amp, C)
        return

    train_loader = build_loader("train", cfg, class_index, train=True)
    val_loader = build_loader("val", cfg, class_index, train=False)
    print(f"classes: {n_classes} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = timm.create_model(cfg["model"], pretrained=(cfg["init"] == "scratch"),
                              num_classes=n_classes).to(device)
    criterion = CostSensitiveLoss(C, cfg["cost_lambda"], cfg["label_smoothing"]).to(device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                  lr=cfg["ft_lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["ft_epochs"])
    scaler = make_scaler(use_amp)

    cost_best, cost_last = ckpt_names(cfg["cost_lambda"])
    start_epoch, best_f1 = 1, -1.0
    resumed = False
    if cfg["resume"] and cost_last.exists():
        ck = torch.load(cost_last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optimizer.load_state_dict(ck["optim_state"]); scheduler.load_state_dict(ck["sched_state"])
        scaler.load_state_dict(ck["scaler_state"])
        start_epoch = ck["epoch"] + 1; best_f1 = ck.get("best_f1", -1.0); resumed = True
        print(f"resumed from {cost_last.name} at epoch {ck['epoch']} (best macro-F1 {best_f1:.3f})")
    elif cfg["init"] == "warmstart":
        if not BASELINE_CKPT.exists():
            sys.exit(f"warmstart requested but baseline checkpoint missing: {BASELINE_CKPT}")
        ck = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        print(f"warm-started from baseline best.pt (val macro-F1 "
              f"{ck.get('metrics', {}).get('val_macro_f1', float('nan')):.3f})")

    if start_epoch > cfg["ft_epochs"]:
        print(f"  this lambda={cfg['cost_lambda']} run already completed {cfg['ft_epochs']} epochs "
              f"({cost_best.name}); nothing to train. Delete {cost_last.name} to retrain. "
              f"Proceeding to comparison.")

    use_wandb = bool(cfg["wandb"]) and _HAS_WANDB
    if use_wandb and not resumed:
        wandb.init(project=cfg["wandb_project"], name=f"{cfg['run_name']}_l{cfg['cost_lambda']:g}",
                   config={**cfg, "n_classes": n_classes, "aug": AUG_CONFIG,
                           "cost_2.0_cells": int((C == 2.0).sum()),
                           "cost_0.2_cells": int((C == 0.2).sum())})

    for epoch in range(start_epoch, cfg["ft_epochs"] + 1):
        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
        metrics = evaluate(model, val_loader, device, criterion, use_amp, C=C)
        scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"epoch {epoch:02d}/{cfg['ft_epochs']} "
              f"train {tr_loss:.3f} | val {metrics['loss']:.3f} "
              f"| top1 {metrics['top1']:.3f} macroF1 {metrics['macro_f1']:.3f} "
              f"| val total_cost {metrics['total_cost']:.1f} hi {metrics['high_cost_errors']} "
              f"| lr {lr_now:.2e}")
        if use_wandb:
            wandb.log({"epoch": epoch, "train_loss": tr_loss, "lr": lr_now,
                       **{f"val_{k}": v for k, v in metrics.items()}})

        ckpt = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(),
                "sched_state": scheduler.state_dict(), "scaler_state": scaler.state_dict(),
                "epoch": epoch, "best_f1": best_f1, "metrics": metrics, "config": cfg,
                "class_index": class_index, "arch": cfg["model"]}
        torch.save(ckpt, cost_last)
        if metrics["macro_f1"] > best_f1:
            best_f1 = metrics["macro_f1"]; ckpt["best_f1"] = best_f1
            torch.save(ckpt, cost_best)
            print(f"   * new best macro-F1 {best_f1:.3f} -> {cost_best.name}")

    if best_f1 >= 0:
        print(f"\nfine-tune done. best val macro-F1: {best_f1:.3f}")
    print("\nscoring baseline + all fine-tuned checkpoints on the TEST set:")
    run_comparison(cfg, class_index, device, use_amp, C)
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
