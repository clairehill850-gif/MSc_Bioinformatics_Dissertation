#!/usr/bin/env python3
# What this does: trains EfficientNet-B3 and ConvNeXt-Tiny ALASKA
import os
import sys
import time
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import timm
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score

from phase2_1_1_dataset_Alaska import PollenDataset, load_class_index
from phase2_1_2_transforms_Alaska import get_val_transform, AUG_CONFIG
from phase2_2_train_Alaska import (build_loaders, set_backbone_trainable,
                            make_optimizer, train_one_epoch, evaluate)

try:
    import wandb
    _HAS_WANDB = True
except Exception:
    _HAS_WANDB = False
try:
    from tqdm import tqdm
except Exception:
    def tqdm(x, **k): return x

try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_scaler(enabled): return _GradScaler("cuda", enabled=enabled)
    def amp_ctx(enabled): return _autocast("cuda", enabled=enabled)
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    def make_scaler(enabled): return _GradScaler(enabled=enabled)
    def amp_ctx(enabled): return _autocast(enabled=enabled)

# Everything is read from and written to the Alaska copy of the project
PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
COST_NPZ = EVAL_DIR / "confusion_cost_matrix.npz"
BASELINE_CKPT = CKPT_DIR / "best.pt"

ARCH_LIST = ["efficientnet_b3", "convnext_tiny"]

CFG = {
    "epochs": 50, "freeze_epochs": 5, "batch_size": 64,
    "head_lr": 1e-3, "full_lr": 1e-4, "weight_decay": 1e-4,
    "num_workers": 4, "seed": 42, "use_amp": True,
    "use_weighted_sampler": False, "label_smoothing": 0.0,
    "wandb_project": "pollen-classifier", "wandb": True,
}


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def arch_ckpts(arch):
    return CKPT_DIR / f"best_{arch}.pt", CKPT_DIR / f"last_{arch}.pt"


def load_cost_matrix(class_index):
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



def make_scheduler(optimizer, cfg):
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg["epochs"] - cfg["freeze_epochs"])


def train_one_arch(arch, cfg, class_index, device, use_amp):
    best_ckpt, last_ckpt = arch_ckpts(arch)
    n_classes = len(class_index)
    train_loader, val_loader = build_loaders(cfg, class_index)
    model = timm.create_model(arch, pretrained=True, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = make_scaler(use_amp)


    set_backbone_trainable(model, trainable=False)
    optimizer = make_optimizer(model, cfg["head_lr"], cfg["weight_decay"])
    scheduler = None
    stage, start_epoch, best_f1 = "head", 1, -1.0

    if cfg.get("resume", True) and last_ckpt.exists():
        ck = torch.load(last_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        stage = ck["stage"]; start_epoch = ck["epoch"] + 1; best_f1 = ck.get("best_f1", -1.0)
        if stage == "full":
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, cfg["full_lr"], cfg["weight_decay"])
            scheduler = make_scheduler(optimizer, cfg)
            if ck.get("sched_state"):
                scheduler.load_state_dict(ck["sched_state"])
        optimizer.load_state_dict(ck["optim_state"]); scaler.load_state_dict(ck["scaler_state"])
        print(f"[{arch}] resumed at epoch {ck['epoch']} (stage={stage}, best macro-F1 {best_f1:.3f})")

    use_wandb = bool(cfg["wandb"]) and _HAS_WANDB
    if use_wandb:
        wandb.init(project=cfg["wandb_project"], name=arch, id=arch, resume="allow",
                   config={**cfg, "model": arch, "n_classes": n_classes, "aug": AUG_CONFIG})

    if start_epoch > cfg["epochs"]:
        print(f"[{arch}] already trained {cfg['epochs']} epochs; skipping to comparison.")
    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if epoch == cfg["freeze_epochs"] + 1 and stage != "full":
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, cfg["full_lr"], cfg["weight_decay"])
            scheduler = make_scheduler(optimizer, cfg)
            stage = "full"
            print(f"[{arch}] epoch {epoch}: unfroze backbone -> full fine-tune at lr {cfg['full_lr']}")

        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
        metrics = evaluate(model, val_loader, device, criterion, use_amp)
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]
        print(f"[{arch}] epoch {epoch:02d}/{cfg['epochs']} [{stage}] "
              f"train {tr_loss:.3f} | val {metrics['val_loss']:.3f} "
              f"top1 {metrics['val_top1']:.3f} macroF1 {metrics['val_macro_f1']:.3f} | lr {lr_now:.2e}")
        if use_wandb:
            wandb.log({"epoch": epoch, "stage": stage, "train_loss": tr_loss, "lr": lr_now, **metrics})

        ck = {"model_state": model.state_dict(), "optim_state": optimizer.state_dict(),
              "sched_state": scheduler.state_dict() if scheduler else None,
              "scaler_state": scaler.state_dict(), "epoch": epoch, "stage": stage,
              "best_f1": best_f1, "metrics": metrics, "config": {**cfg, "model": arch},
              "class_index": class_index, "arch": arch}
        torch.save(ck, last_ckpt)
        if metrics["val_macro_f1"] > best_f1:
            best_f1 = metrics["val_macro_f1"]; ck["best_f1"] = best_f1
            torch.save(ck, best_ckpt)
            print(f"   * [{arch}] new best macro-F1 {best_f1:.3f} -> {best_ckpt.name}")
    if use_wandb:
        wandb.finish()
    print(f"[{arch}] done. best val macro-F1 {best_f1:.3f}")


@torch.no_grad()
def evaluate_test(model, loader, device, use_amp, C):
    model.eval()
    top1 = top5 = n = 0
    yp_all, yt_all = [], []
    for imgs, labels in tqdm(loader, desc="test", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx(use_amp):
            logits = model(imgs)
        _, pred5 = logits.topk(5, dim=1)
        top1 += (pred5[:, 0] == labels).sum().item()
        top5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        n += labels.size(0)
        yp_all.append(pred5[:, 0].cpu().numpy()); yt_all.append(labels.cpu().numpy())
    yp = np.concatenate(yp_all); yt = np.concatenate(yt_all)
    out = {"top1": top1 / n, "top5": top5 / n,
           "macro_f1": f1_score(yt, yp, average="macro", zero_division=0),
           "weighted_f1": f1_score(yt, yp, average="weighted", zero_division=0)}
    if C is not None:
        wrong = yt != yp
        costs = C[yt[wrong], yp[wrong]] if wrong.any() else np.array([])
        out["total_cost"] = float(costs.sum())
        out["high_cost_errors"] = int((costs >= 2.0).sum())
    else:
        out["total_cost"] = np.nan; out["high_cost_errors"] = -1
    return out


@torch.no_grad()
def bench_inference(model, device, use_amp, bs=64, n_imgs=1280):
    model.eval()
    x = torch.randn(bs, 3, 224, 224, device=device)
    for _ in range(5):
        with amp_ctx(use_amp):
            model(x)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.time(); done = 0
    while done < n_imgs:
        with amp_ctx(use_amp):
            model(x)
        done += bs
    if device == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t0) / done * 1000.0


def build_comparison(cfg, class_index, device, use_amp):
    import glob
    n_classes = len(class_index)
    C = load_cost_matrix(class_index)
    test_loader = DataLoader(PollenDataset("test", class_index, transform=get_val_transform()),
                             batch_size=cfg["batch_size"], shuffle=False,
                             num_workers=cfg["num_workers"], pin_memory=True)

    targets = []
    if BASELINE_CKPT.exists():
        targets.append(("resnet50", BASELINE_CKPT))
    for p in sorted(glob.glob(str(CKPT_DIR / "best_*.pt"))):
        if Path(p).name.startswith("best_cost"):
            continue
        targets.append((None, Path(p)))

    rows = []
    for name, ckpt_path in targets:
        state = torch.load(ckpt_path, map_location=device, weights_only=False)
        arch = state.get("arch") or state.get("config", {}).get("model", "?")
        if name is None:
            name = arch
        model = timm.create_model(arch, pretrained=False, num_classes=n_classes).to(device)
        model.load_state_dict(state["model_state"])
        params_m = sum(p.numel() for p in model.parameters()) / 1e6
        m = evaluate_test(model, test_loader, device, use_amp, C)
        m["infer_ms_per_img"] = bench_inference(model, device, use_amp, cfg["batch_size"])
        m["arch"] = name; m["params_M"] = round(params_m, 1)
        rows.append(m)
        print(f"  {name:16s} {params_m:5.1f}M | top1 {m['top1']:.4f} macroF1 {m['macro_f1']:.4f} "
              f"wF1 {m['weighted_f1']:.4f} | cost {m['total_cost']:.0f} hi {m['high_cost_errors']} "
              f"| {m['infer_ms_per_img']:.2f} ms/img")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
    if rows:
        cols = ["arch", "params_M", "top1", "top5", "macro_f1", "weighted_f1",
                "total_cost", "high_cost_errors", "infer_ms_per_img"]
        df = pd.DataFrame(rows)[cols].sort_values("macro_f1", ascending=False)
        EVAL_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(EVAL_DIR / "architecture_comparison.csv", index=False)
        best = df.iloc[0]
        print(f"\nwrote architecture_comparison.csv -> {EVAL_DIR}")
        print(f"best by macro-F1: {best['arch']} "
              f"(macroF1 {best['macro_f1']:.4f}, top1 {best['top1']:.4f})")
    return rows


def main():
    flags = [a for a in sys.argv[1:] if a.startswith("--")]
    archs = [a for a in sys.argv[1:] if not a.startswith("--")]
    cfg = dict(CFG); cfg["resume"] = True
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg["use_amp"]) and device == "cuda"
    torch.backends.cudnn.benchmark = True
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    class_index = load_class_index()
    print(f"device: {device} | AMP: {use_amp} | classes: {len(class_index)}")

    if "--compare" in flags:
        print("comparison only:")
        build_comparison(cfg, class_index, device, use_amp)
        return

    to_train = archs if archs else ARCH_LIST
    for arch in to_train:
        print(f"\n===== training {arch} =====")
        set_seed(cfg["seed"])
        train_one_arch(arch, cfg, class_index, device, use_amp)

    print("\n===== architecture comparison (test set) =====")
    build_comparison(cfg, class_index, device, use_amp)


if __name__ == "__main__":
    main()
