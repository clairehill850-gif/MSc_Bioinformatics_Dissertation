
# What this does: re-runs the trained model over ALASKA test set
import os
import sys
from pathlib import Path
from phase0_determinism_Alaska import enable, null_autocast

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
PREDS_OUT = EVAL_DIR / "test_predictions.csv"
EVAL_BATCH = 128


def assemble_pred_df(df_meta, true, pred, conf, top5c, index_to_label):
    true = np.asarray(true); pred = np.asarray(pred)
    out = pd.DataFrame({
        "dataset": df_meta["dataset"].astype(str).to_numpy(),
        "true_label": [index_to_label[int(t)] for t in true],
        "pred_label": [index_to_label[int(p)] for p in pred],
        "confidence": np.round(np.asarray(conf, dtype=float), 4),
        "correct": (pred == true),
        "top5_correct": np.asarray(top5c, dtype=bool),
        "processed_path": df_meta["processed_path"].astype(str).to_numpy(),
        "raw_rel_path": (df_meta["raw_rel_path"].astype(str).to_numpy()
                         if "raw_rel_path" in df_meta.columns
                         else df_meta["processed_path"].astype(str).to_numpy()),
    })
    return out


def show_pair(pred_df, true_label, pred_label):
    sub = pred_df[(pred_df["true_label"] == true_label) &
                  (pred_df["pred_label"] == pred_label)]
    print(f"\n=== {true_label} -> {pred_label}: {len(sub)} image(s) ===")
    if len(sub) == 0:
        print("  (no such misclassifications in the test set)")
        return
    by_src = sub["dataset"].value_counts()
    print("  by source:")
    for src, c in by_src.items():
        print(f"     {src[:28]:28s} {c}")
    print("  images (open raw_rel_path to inspect):")
    for _, r in sub.sort_values("confidence", ascending=False).iterrows():
        print(f"     conf {r['confidence']:.3f} | {r['dataset'][:18]:18s} | {r['raw_rel_path']}")


def show_class_errors(pred_df, true_label):
    sub = pred_df[(pred_df["true_label"] == true_label) & (~pred_df["correct"])]
    print(f"\n=== errors where true == {true_label}: {len(sub)} ===")
    if len(sub) == 0:
        print("  (none)")
        return
    for pl, grp in sub.groupby("pred_label"):
        print(f"  -> {pl[:28]:28s} {len(grp)}")
        for _, r in grp.iterrows():
            print(f"       conf {r['confidence']:.3f} | {r['dataset'][:18]:18s} | {r['raw_rel_path']}")


def _torch_load(path, device):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def main():
    args = sys.argv[1:]
    pair = None
    cls = None
    if "--pair" in args:
        i = args.index("--pair")
        if i + 2 < len(args) + 1 and i + 2 <= len(args):
            pair = (args[i + 1], args[i + 2])
    if "--class" in args:
        i = args.index("--class")
        if i + 1 < len(args):
            cls = args[i + 1]

    if PREDS_OUT.exists() and (pair or cls):
        pred_df = pd.read_csv(PREDS_OUT)
        print(f"using existing {PREDS_OUT.name} ({len(pred_df):,} rows)")
    else:
        pred_df = _run_inference()

    if pair:
        show_pair(pred_df, pair[0], pair[1])
    if cls:
        show_class_errors(pred_df, cls)
    if not (pair or cls):
        n = len(pred_df); acc = pred_df["correct"].mean()
        print(f"\nwrote {n:,} rows | top-1 {acc:.4f} -> {PREDS_OUT}")
        print("query e.g.:  python phase2_3b_dump_predictions.py --pair Cecropia Mimosa")


def _run_inference():
    import torch
    import timm
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_1_1_dataset_Alaska import PollenDataset, load_class_index
    from phase2_1_2_transforms_Alaska import get_val_transform

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    if not CKPT_PATH.exists():
        sys.exit(f"checkpoint not found: {CKPT_PATH}\nRun phase2_2_train.py first.")

    ckpt = _torch_load(CKPT_PATH, device)
    class_index = ckpt.get("class_index") or load_class_index()
    class_index = {str(k): int(v) for k, v in class_index.items()}
    index_to_label = {v: k for k, v in class_index.items()}
    arch = ckpt.get("arch", "resnet50")
    n_classes = len(class_index)
    print(f"checkpoint epoch {ckpt.get('epoch','?')} | arch {arch} | classes {n_classes} | device {device}")

    test_ds = PollenDataset("test", class_index, transform=get_val_transform())
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=4, pin_memory=(device == "cuda"))

    model = timm.create_model(arch, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    autocast = null_autocast

    true, pred, conf, top5c = [], [], [], []
    done = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                logits = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)
            maxp, p = probs.max(dim=1)
            t5 = probs.topk(5, dim=1).indices
            lab = labels.numpy()
            true.append(lab); pred.append(p.cpu().numpy()); conf.append(maxp.cpu().numpy())
            top5c.append((t5.cpu().numpy() == lab[:, None]).any(axis=1))
            done += len(lab)
            if done % (EVAL_BATCH * 10) < EVAL_BATCH:
                print(f"  {done}/{len(test_ds)}")

    true = np.concatenate(true); pred = np.concatenate(pred)
    conf = np.concatenate(conf); top5c = np.concatenate(top5c)
    pred_df = assemble_pred_df(test_ds.df, true, pred, conf, top5c, index_to_label)
    pred_df.to_csv(PREDS_OUT, index=False, encoding="utf-8")
    return pred_df


if __name__ == "__main__":
    main()
