#!/usr/bin/env python3
# What this does: sets up how images are loaded for training in the Alaska run
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from torch.utils.data import Dataset
    _HAS_TORCH = True
except Exception:
    Dataset = object
    _HAS_TORCH = False

try:
    import cv2
    _HAS_CV2 = True
except Exception:
    _HAS_CV2 = False

PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"
PROCESSED_ALASKA = PROJECT_ROOT / "data" / "processed_alaska"
SPLITS = PROJECT_ROOT / "outputs" / "curation" / "splits"
INDEX_PATH = SPLITS / "class_index_final.json"

def build_class_index(splits_dir=SPLITS, out_path=INDEX_PATH):
    labels = set()
    for s in ("train", "val", "test"):
        p = Path(splits_dir) / f"processed_{s}.csv"
        if p.exists():
            labels |= set(pd.read_csv(p)["final_label"].astype(str))
    classes = sorted(labels)
    index = {c: i for i, c in enumerate(classes)}
    if out_path:
        Path(out_path).write_text(json.dumps(index, indent=1), encoding="utf-8")
    return index


def load_class_index(path=INDEX_PATH):
    return json.loads(Path(path).read_text(encoding="utf-8"))



class PollenDataset(Dataset):

    def __init__(self, split, class_index, processed_root=PROCESSED_ROOT,
                 splits_dir=SPLITS, transform=None):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required to instantiate PollenDataset.")
        if not _HAS_CV2:
            raise RuntimeError("opencv (cv2) is required to load images.")
        self.df = pd.read_csv(Path(splits_dir) / f"processed_{split}.csv").reset_index(drop=True)
        self.root = Path(processed_root)
        self.class_index = class_index
        self.transform = transform
        missing = set(self.df["final_label"].astype(str)) - set(class_index)
        if missing:
            raise ValueError(f"{split}: {len(missing)} labels absent from class index "
                             f"(e.g. {sorted(missing)[:3]}). Rebuild the index.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        pp = r["processed_path"]
        cand = PROCESSED_ALASKA / pp
        path = cand if cand.exists() else (self.root / pp)
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = int(self.class_index[str(r["final_label"])])
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, label



def main():
    index = build_class_index()
    print(f"class index rebuilt: {len(index)} classes -> {INDEX_PATH}")
    for split in ("train", "val", "test"):
        p = SPLITS / f"processed_{split}.csv"
        if p.exists():
            df = pd.read_csv(p)
            unseen = set(df["final_label"].astype(str)) - set(index)
            print(f"  {split}: {len(df):,} images, {df['final_label'].nunique()} classes, "
                  f"{'all in index' if not unseen else f'{len(unseen)} MISSING'}")
    vals = sorted(index.values())
    print(f"  indices contiguous 0..{len(index)-1}: {vals == list(range(len(index)))}")


if __name__ == "__main__":
    main()
