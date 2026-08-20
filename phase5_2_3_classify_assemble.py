#!/usr/bin/env python3
# What this does: Classification. ALASKA

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
import torch
import timm

import phase0_determinism_Alaska as det

# Alaska root
ROOT = Path("/workspace/datasets/pollen_bundle/Alaska_Test")
sys.path.insert(0, str(ROOT / "alaska_scripts"))
import phase2_1_1_dataset_Alaska as ds
from phase2_1_2_transforms_Alaska import get_val_transform

# Crops and results path
CROPS = ROOT / "outputs" / "pipeline" / "crops"
MANIFEST = ROOT / "outputs" / "pipeline" / "crop_manifest.csv"
OUT = ROOT / "outputs" / "pipeline"

# Confidence cut-off
TAU = 0.55
# TExclude spores
SPORE_LABEL = "Spores-undiff"
TEMP_JSON = ROOT / "outputs" / "eval" / "temperature.json"


# Load Alaska classifier from chkpnt
def load_model(device, n_classes):
    ck = torch.load(ROOT / "outputs" / "checkpoints" / "best.pt", map_location=device,
                    weights_only=False)
    arch = ck.get("arch", "convnext_tiny")
    model = timm.create_model(arch, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ck["model_state"])
    return model.to(device).eval()


# Classify and build assemblage
def main():
    det.enable()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    idx = ds.load_class_index()
    inv = {v: k for k, v in idx.items()}
    model = load_model(device, len(idx))

# TSoften confidence scores
    T = 1.0
    if TEMP_JSON.exists():
        import json
        T = json.loads(TEMP_JSON.read_text()).get("temperature", 1.0)
    print(f"classes: {len(idx)} | tau: {TAU} | temperature: {T:.3f}")

    tf = get_val_transform(224)
    man = pd.read_csv(MANIFEST)

    rows = []
    with torch.no_grad():
        for r in man.itertuples():
            img = Image.open(CROPS / r.crop).convert("RGB")
            x = tf(image=np.asarray(img))["image"].unsqueeze(0).to(device)
            logits = model(x) / T
            prob = torch.softmax(logits, 1)[0]
            conf, ci = float(prob.max()), int(prob.argmax())
            label = inv[ci]
            rows.append({
                "sample": r.sample, "crop": r.crop,
                "pred_label": label, "confidence": round(conf, 4),
                "indeterminate": conf < TAU,
                "detector_conf": round(r.conf, 4),
            })
    pred = pd.DataFrame(rows)
    pred.to_csv(OUT / "grain_predictions.csv", index=False)
    print(f"classified {len(pred)} grains -> grain_predictions.csv")

# Assemblage/slide
    n_indet = int(pred.indeterminate.sum())
    n_spore = int((pred.pred_label == SPORE_LABEL).sum())
    keep = pred[(~pred.indeterminate) & (pred.pred_label != SPORE_LABEL)]
    print(f"  indeterminate (< tau): {n_indet}  |  spores dropped: {n_spore}  "
          f"|  counted pollen: {len(keep)}")

# Counts and percentages
    prof = (keep.groupby(["sample", "pred_label"]).size()
                .rename("count").reset_index())
    totals = prof.groupby("sample")["count"].transform("sum")
    prof["pct"] = (100 * prof["count"] / totals).round(1)
    prof = prof.sort_values(["sample", "count"], ascending=[True, False])
    prof.to_csv(OUT / "assemblage_profile_auto.csv", index=False)

    wide = prof.pivot_table(index="sample", columns="pred_label",
                            values="pct", fill_value=0.0)
    wide.to_csv(OUT / "assemblage_wide_auto.csv")

    print(f"\nper-sample profile -> assemblage_profile_auto.csv")
    print(f"wide matrix       -> assemblage_wide_auto.csv")
# Top 3 taxa
    print("\n=== dominant taxon per sample (automated) ===")
    for s in sorted(prof["sample"].unique()):
        sub = prof[prof["sample"] == s].head(3)
        top = ", ".join(f"{r.pred_label} {r.pct}%" for r in sub.itertuples())
        n = int(keep[keep["sample"] == s].shape[0])
        print(f"  {s:22s} (n={n:2d}): {top}")

# Stats
    book = (pred.groupby("sample")
                .agg(grains=("crop", "size"),
                     indeterminate=("indeterminate", "sum"),
                     spores=("pred_label", lambda s: (s == SPORE_LABEL).sum()))
                .reset_index())
    book["counted"] = book["grains"] - book["indeterminate"] - book["spores"]
    book.to_csv(OUT / "sample_bookkeeping.csv", index=False)
    print(f"\nbookkeeping -> sample_bookkeeping.csv")


if __name__ == "__main__":
    main()
