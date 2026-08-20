#!/usr/bin/env python3
# What this does: trains the classifier for the Alaska regional model
import sys
import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
import timm
from sklearn.metrics import f1_score

from phase2_1_1_dataset_Alaska import PollenDataset, load_class_index, SPLITS, PROCESSED_ROOT
from phase2_1_2_transforms_Alaska import get_train_transform, get_train_transform_gray, get_train_transform_domain, get_val_transform, AUG_CONFIG, AUG_CONFIG_DOMAIN

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

PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
CKPT_DIR = PROJECT_ROOT / "outputs" / "checkpoints"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "config.yaml"

DEFAULTS = {
    "model": "resnet50",
    "epochs": 50,
    "freeze_epochs": 5,
    "batch_size": 64,
    "head_lr": 1e-3,
    "full_lr": 1e-4,
    "weight_decay": 1e-4,
    "num_workers": 4,
    "seed": 42,
    "use_amp": True,
    "use_weighted_sampler": False,
    "label_smoothing": 0.0,
    "resume": True,
    "wandb_project": "pollen-classifier",
    "wandb": True,
    "run_name": "resnet50_baseline",
}


def load_config(path=CONFIG_PATH):
    cfg = dict(DEFAULTS)
    if Path(path).exists():
        with open(path) as f:
            cfg.update({k: v for k, v in (yaml.safe_load(f) or {}).items()})
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.safe_dump(DEFAULTS, f, sort_keys=False)
        print(f"wrote default config -> {path}")
    return cfg


def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(cfg, class_index):
    train_ds = PollenDataset("train", class_index, transform=(get_train_transform_gray() if os.environ.get("POLLEN_GRAY")=="1" else get_train_transform_domain() if os.environ.get("POLLEN_DOMAIN_AUG")=="1" else get_train_transform()))
    val_ds = PollenDataset("val", class_index, transform=get_val_transform())
    if cfg["use_weighted_sampler"]:
        labels = train_ds.df["final_label"].map(class_index).to_numpy()
        counts = np.bincount(labels, minlength=len(class_index)).astype(float)
        w = 1.0 / np.clip(counts, 1, None)
        sample_w = w[labels]
        sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                        num_samples=len(sample_w), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], sampler=sampler,
                                  num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    else:
        train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                                  num_workers=cfg["num_workers"], pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            num_workers=cfg["num_workers"], pin_memory=True)
    return train_loader, val_loader


def set_backbone_trainable(model, trainable):
    for p in model.parameters():
        p.requires_grad = trainable
    for p in model.get_classifier().parameters():
        p.requires_grad = True


def make_optimizer(model, lr, wd):
    params = [p for p in model.parameters() if p.requires_grad]
    return torch.optim.AdamW(params, lr=lr, weight_decay=wd)


def torch_load(path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


@torch.no_grad()
def evaluate(model, loader, device, criterion, use_amp):
    model.eval()
    loss_sum, n, top1, top5 = 0.0, 0, 0, 0
    all_pred, all_true = [], []
    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with amp_ctx(use_amp):
            logits = model(imgs)
            loss = criterion(logits, labels)
        loss_sum += loss.item() * labels.size(0); n += labels.size(0)
        _, pred5 = logits.topk(5, dim=1)
        top1 += (pred5[:, 0] == labels).sum().item()
        top5 += (pred5 == labels.unsqueeze(1)).any(dim=1).sum().item()
        all_pred.append(pred5[:, 0].cpu().numpy()); all_true.append(labels.cpu().numpy())
    yp = np.concatenate(all_pred); yt = np.concatenate(all_true)
    return {"val_loss": loss_sum / n, "val_top1": top1 / n, "val_top5": top5 / n,
            "val_macro_f1": f1_score(yt, yp, average="macro", zero_division=0)}


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


def main():
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else CONFIG_PATH)
    set_seed(cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = bool(cfg["use_amp"]) and device == "cuda"
    torch.backends.cudnn.benchmark = True
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"device: {device} | AMP: {use_amp}")

    class_index = load_class_index()
    n_classes = len(class_index)
    train_loader, val_loader = build_loaders(cfg, class_index)
    print(f"classes: {n_classes} | train batches: {len(train_loader)} | val batches: {len(val_loader)}")

    model = timm.create_model(cfg["model"], pretrained=True, num_classes=n_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg["label_smoothing"])
    scaler = make_scaler(use_amp)

    use_wandb = bool(cfg["wandb"]) and _HAS_WANDB
    if use_wandb:
        wandb.init(project=cfg["wandb_project"], name=cfg["run_name"],
                   config={**cfg, "n_classes": n_classes, "aug": AUG_CONFIG})
    elif cfg["wandb"] and not _HAS_WANDB:
        print("  (wandb requested but not installed - logging to console only)")

    set_backbone_trainable(model, trainable=False)
    optimizer = make_optimizer(model, cfg["head_lr"], cfg["weight_decay"])
    scheduler = None
    stage = "head"
    start_epoch = 1
    best_f1 = -1.0

    last_ckpt = CKPT_DIR / "last.pt"
    if cfg.get("resume", True) and last_ckpt.exists():
        ck = torch_load(last_ckpt, device)
        model.load_state_dict(ck["model_state"])
        start_epoch = int(ck["epoch"]) + 1
        best_f1 = float(ck.get("best_f1", -1.0))
        if start_epoch > cfg["freeze_epochs"]:
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, cfg["full_lr"], cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - cfg["freeze_epochs"])
            stage = "full"
            steps = max(0, start_epoch - 1 - cfg["freeze_epochs"])
            if ck.get("stage") == "full" and ck.get("scheduler_state"):
                try:
                    scheduler.load_state_dict(ck["scheduler_state"])
                except Exception:
                    for _ in range(steps): scheduler.step()
            else:
                for _ in range(steps): scheduler.step()
        if ck.get("stage") == stage and ck.get("optimizer_state"):
            try:
                optimizer.load_state_dict(ck["optimizer_state"])
            except Exception:
                pass
        if ck.get("scaler_state") is not None:
            try:
                scaler.load_state_dict(ck["scaler_state"])
            except Exception:
                pass
        print(f"resumed: epoch {ck['epoch']} done -> starting at {start_epoch} "
              f"[{stage}], best macro-F1 so far {best_f1:.3f}")

    for epoch in range(start_epoch, cfg["epochs"] + 1):
        if epoch == cfg["freeze_epochs"] + 1 and stage != "full":
            set_backbone_trainable(model, trainable=True)
            optimizer = make_optimizer(model, cfg["full_lr"], cfg["weight_decay"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=cfg["epochs"] - cfg["freeze_epochs"])
            stage = "full"

        tr_loss = train_one_epoch(model, train_loader, device, criterion, optimizer, scaler, use_amp)
        metrics = evaluate(model, val_loader, device, criterion, use_amp)
        if scheduler is not None:
            scheduler.step()
        lr_now = optimizer.param_groups[0]["lr"]

        log = {"epoch": epoch, "stage": stage, "train_loss": tr_loss, "lr": lr_now, **metrics}
        print(f"epoch {epoch:02d}/{cfg['epochs']} [{stage}] "
              f"train {tr_loss:.3f} | val {metrics['val_loss']:.3f} "
              f"| top1 {metrics['val_top1']:.3f} top5 {metrics['val_top5']:.3f} "
              f"| macroF1 {metrics['val_macro_f1']:.3f} | lr {lr_now:.2e}")
        if use_wandb:
            wandb.log(log)

        improved = metrics["val_macro_f1"] > best_f1
        if improved:
            best_f1 = metrics["val_macro_f1"]
        ckpt = {"model_state": model.state_dict(),
                "optimizer_state": optimizer.state_dict(),
                "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
                "scaler_state": scaler.state_dict(),
                "epoch": epoch, "stage": stage, "best_f1": best_f1,
                "metrics": metrics, "config": cfg,
                "class_index": class_index, "arch": cfg["model"]}
        torch.save(ckpt, CKPT_DIR / "last.pt")
        if improved:
            torch.save(ckpt, CKPT_DIR / "best.pt")
            print(f"   * new best macro-F1 {best_f1:.3f} -> best.pt")

    print(f"\nDone. Best val macro-F1: {best_f1:.3f}")
    if use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
