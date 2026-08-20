#!/usr/bin/env python3
# What this does: Leave one slide out fine-tune (greyscale)
import json
import copy
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, Subset
import timm

# Alaska root
ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
sys.path.insert(0, str(ROOT / "alaska_scripts"))
import phase0_determinism_Alaska as det
import phase2_1_1_dataset_Alaska as ds
from phase2_1_2_transforms_Alaska import get_train_transform_gray as get_train_transform, get_val_transform_gray as get_val_transform

# Folders
OUT = ROOT / "outputs" / "pipeline"
ANNOT = OUT / "annotation" / "slide_annotations.csv"
BASELINE_CKPT = ROOT / "outputs" / "checkpoints_baseline" / "best.pt"
TEMP_JSON = ROOT / "outputs" / "eval" / "temperature.json"
RESULT = OUT / "loso_gray_predictions.csv"
CROP_DIRS = [OUT / "crops_margin10", OUT / "crops_pad1p0", OUT / "crops"]

# Fine-tuning settings
SIZE = 224
EPOCHS = 12
BATCH = 16
LR_HEAD = 1e-4
LR_STAGE = 1e-5
REF_PER_SLIDE_GRAIN = 2
# Drop non-pollen
DROP_LABELS = {"reject", "Larix"}
RENAME = {"Alder": "Alnus"}


def find_crop_dir():
    for d in CROP_DIRS:
        if d.exists() and any(d.glob("*.png")):
            return d
    raise SystemExit("no crop directory found")


# Slide crops and annotations
class SlideCrops(Dataset):
    def __init__(self, frame, crop_dir, class_index, transform):
        self.rows = frame.reset_index(drop=True)
        self.dir = crop_dir
        self.idx = class_index
        self.tf = transform

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows.iloc[i]
        img = Image.open(self.dir / r["crop"]).convert("RGB")
        x = self.tf(image=np.asarray(img))["image"]
        return x, self.idx[r["true_label"]]


# Load chckpnt
def load_baseline(device, n_classes):
    ck = torch.load(BASELINE_CKPT, map_location=device, weights_only=False)
    arch = ck.get("arch", "convnext_tiny")
    model = timm.create_model(arch, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ck["model_state"])
    return model.to(device), arch


# Taining
def set_trainable(model):
    for p in model.parameters():
        p.requires_grad = False
    head, stage = [], []
    for name, p in model.named_parameters():
        if name.startswith("head") or ".head." in name:
            p.requires_grad = True
            head.append(p)
        elif "stages.3" in name or "stages_3" in name:
            p.requires_grad = True
            stage.append(p)
    if not head:
        raise SystemExit("could not identify the classifier head - check the arch")
    return head, stage


# Return results 
def evaluate(model, loader, device, T):
    model.eval()
    preds, confs = [], []
    with torch.no_grad():
        for x, _ in loader:
            prob = torch.softmax(model(x.to(device)) / T, 1)
            c, i = prob.max(1)
            preds += i.cpu().tolist()
            confs += c.cpu().tolist()
    return preds, confs


# Leave one slide out loop
def main():
    det.enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    crop_dir = find_crop_dir()

    class_index = ds.load_class_index()
    inv = {v: k for k, v in class_index.items()}
    T = json.loads(TEMP_JSON.read_text()).get("temperature", 1.0) if TEMP_JSON.exists() else 1.0
    tau_src = json.loads(TEMP_JSON.read_text()) if TEMP_JSON.exists() else {}
    TAU = tau_src.get("operating_threshold", 0.56)

# Annotations
    ann = pd.read_csv(ANNOT)
    ann["true_label"] = ann["true_label"].fillna("").astype(str).str.strip()
    ann["true_label"] = ann["true_label"].replace(RENAME)
    n_all = len(ann)
    rejected = int((ann["true_label"] == "reject").sum())
    ann = ann[(ann["true_label"] != "") & (~ann["true_label"].isin(DROP_LABELS))]
    unknown = sorted(set(ann["true_label"]) - set(class_index))
    if unknown:
        print(f"dropping labels not in the class index: {unknown}")
        ann = ann[~ann["true_label"].isin(unknown)]

    print(f"crops: {n_all} annotated | {rejected} rejected (detector false positives, "
          f"{100 * rejected / n_all:.1f}%) | {len(ann)} usable")
    print(ann["true_label"].value_counts().to_string())

    slides = sorted(ann["sample"].unique())
    n_slide_of = ann.groupby("true_label")["sample"].nunique()
    multi = set(n_slide_of[n_slide_of > 1].index)
    print(f"\n{len(slides)} slides | multi-slide classes (primary result): {sorted(multi)}")

# Reference data
    ref_train = ds.PollenDataset("train", class_index, transform=get_train_transform(SIZE))
    print(f"reference train pool: {len(ref_train)} images")

    train_tf = get_train_transform(SIZE)
    val_tf = get_val_transform(SIZE)

# Train/test
    rows = []
    for held in slides:
        tr = ann[ann["sample"] != held]
        te = ann[ann["sample"] == held]

        model, arch = load_baseline(device, len(class_index))
        head, stage = set_trainable(model)
        opt = torch.optim.AdamW(
            [{"params": head, "lr": LR_HEAD},
             {"params": stage, "lr": LR_STAGE}], weight_decay=1e-4)
        lossf = nn.CrossEntropyLoss()

# Fixed seed for reproduability
        slide_ds = SlideCrops(tr, crop_dir, class_index, train_tf)
        n_ref = min(len(ref_train), REF_PER_SLIDE_GRAIN * len(slide_ds))
        g = torch.Generator().manual_seed(0)
        pick = torch.randperm(len(ref_train), generator=g)[:n_ref].tolist()
        mixed = ConcatDataset([slide_ds, Subset(ref_train, pick)])
        loader = DataLoader(mixed, batch_size=BATCH, shuffle=True, num_workers=2, drop_last=False)

# Fine-tune
        model.train()
        for _ in range(EPOCHS):
            for x, y in loader:
                opt.zero_grad()
                loss = lossf(model(x.to(device)), y.to(device))
                loss.backward()
                opt.step()

# Test on held-out slide
        te_loader = DataLoader(SlideCrops(te, crop_dir, class_index, val_tf),
                               batch_size=BATCH, shuffle=False, num_workers=2)
        preds, confs = evaluate(model, te_loader, device, T)

        for (_, r), p, c in zip(te.iterrows(), preds, confs):
            rows.append({"sample": held, "crop": r["crop"], "true_label": r["true_label"],
                         "pred_label": inv[p], "confidence": round(c, 4),
                         "indeterminate": c < TAU, "correct": inv[p] == r["true_label"]})

        acc = np.mean([x["correct"] for x in rows[-len(te):]]) if len(te) else float("nan")
        print(f"  held out {held:>22}: {len(te):>2} grains | fold accuracy {acc:.3f}")

# Free model
        del model, opt
        torch.cuda.empty_cache()

    res = pd.DataFrame(rows)
    res.to_csv(RESULT, index=False)
    print(f"\nwrote {len(res)} pooled predictions -> {RESULT.name}")

# Results
    prim = res[res["true_label"].isin(multi)]
    print("\n=== LEAVE-ONE-SLIDE-OUT RESULT ===")
    print(f"all usable grains        : {res.correct.sum()}/{len(res)} "
          f"({100 * res.correct.mean():.1f}%)")
    print(f"multi-slide classes only : {prim.correct.sum()}/{len(prim)} "
          f"({100 * prim.correct.mean():.1f}%)")
# Accuracy
    kept = res[~res.indeterminate]
    if len(kept):
        print(f"retained above tau={TAU}   : {kept.correct.sum()}/{len(kept)} "
              f"({100 * kept.correct.mean():.1f}%)  | {int(res.indeterminate.sum())} indeterminate")

    print("\nper taxon (baseline slide performance in the manuscript: Betula 1/17, Populus 2/16):")
    t = (res.groupby("true_label")
            .agg(n=("correct", "size"), correct=("correct", "sum"))
            .assign(pct=lambda d: (100 * d.correct / d.n).round(1),
                    multi_slide=lambda d: [i in multi for i in d.index])
            .sort_values("n", ascending=False))
    print(t.to_string())

# Mistakes
    print("\nwhere the errors went:")
    err = res[~res.correct].groupby(["true_label", "pred_label"]).size().sort_values(ascending=False)
    print(err.head(12).to_string() if len(err) else "  none")


if __name__ == "__main__":
    main()
