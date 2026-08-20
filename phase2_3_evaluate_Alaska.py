
# What this does: takes the best trained model and tests it ALASKA
import os
import sys
import json
from pathlib import Path
from phase0_determinism_Alaska import enable, null_autocast

import numpy as np
import pandas as pd

from sklearn.metrics import (
    confusion_matrix as sk_confusion_matrix,
    precision_recall_fscore_support,
    f1_score,
)

PROJECT_ROOT = Path(r"D:\Pollen\Alaska_Test") if os.name == "nt" else Path("/workspace/datasets/pollen_bundle/Alaska_Test")
CKPT_PATH = PROJECT_ROOT / "outputs" / "checkpoints" / "best.pt"
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
CONFIG_PATH = PROJECT_ROOT / "scripts" / "config.yaml"
# Forensic tiers
TIERS_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "curation" / "forensic_relevance_tiers.csv",
    PROJECT_ROOT / "outputs" / "forensic" / "forensic_relevance_tiers.csv",
    PROJECT_ROOT / "outputs" / "forensic_relevance_tiers.csv",
]


EVAL_BATCH = 128   
NUM_WORKERS = 4
TOP_PAIRS = 30    
TOP_HEATMAP = 40


def standard_metrics(true, pred, top5, n_classes):
    true = np.asarray(true); pred = np.asarray(pred); top5 = np.asarray(top5)
    labels = list(range(n_classes))
    top1 = float((pred == true).mean())
    top5_acc = float((top5 == true[:, None]).any(axis=1).mean())
    macro = float(f1_score(true, pred, labels=labels, average="macro", zero_division=0))
    weighted = float(f1_score(true, pred, labels=labels, average="weighted", zero_division=0))
    return {"n_images": int(len(true)), "n_classes": int(n_classes),
            "top1": top1, "top5": top5_acc, "macro_f1": macro, "weighted_f1": weighted}


def per_class_table(true, pred, n_classes, index_to_label):
    labels = list(range(n_classes))
    p, r, f, s = precision_recall_fscore_support(
        true, pred, labels=labels, zero_division=0)
    return pd.DataFrame({
        "class_index": labels,
        "label": [index_to_label[i] for i in labels],
        "precision": p, "recall": r, "f1": f, "support": s.astype(int),
    })


def confusion_and_pairs(true, pred, n_classes, index_to_label, top_n=TOP_PAIRS):
    labels = list(range(n_classes))
    cm = sk_confusion_matrix(true, pred, labels=labels)
    support = cm.sum(axis=1)
    off = cm.copy()
    np.fill_diagonal(off, 0)
    flat = np.argsort(off.ravel())[::-1]
    pairs = []
    for idx in flat[: top_n * 4]:
        i, j = divmod(int(idx), n_classes)
        c = int(off[i, j])
        if c == 0:
            break
        pairs.append({
            "true_index": i, "true_label": index_to_label[i],
            "pred_index": j, "pred_label": index_to_label[j],
            "count": c,
            "true_support": int(support[i]),
            "pct_of_true": round(100.0 * c / support[i], 2) if support[i] else 0.0,
        })
        if len(pairs) >= top_n:
            break
    return cm, pd.DataFrame(pairs)


def per_source_table(true, pred, top5, sources):
    true = np.asarray(true); pred = np.asarray(pred); top5 = np.asarray(top5)
    sources = np.asarray(sources, dtype=object)
    hit1 = (pred == true)
    hit5 = (top5 == true[:, None]).any(axis=1)
    rows = []
    for src in sorted(set(sources.tolist())):
        m = sources == src
        rows.append({
            "source": src,
            "n_images": int(m.sum()),
            "n_classes": int(len(set(true[m].tolist()))),
            "top1": round(float(hit1[m].mean()), 4),
            "top5": round(float(hit5[m].mean()), 4),
        })
    df = pd.DataFrame(rows).sort_values("n_images", ascending=False).reset_index(drop=True)
    return df


def attach_tiers(per_class_df):
    path = next((p for p in TIERS_CANDIDATES if p.exists()), None)
    if path is None:
        return per_class_df
    try:
        t = pd.read_csv(path)
    except Exception:
        return per_class_df
    lab_col = next((c for c in t.columns
                    if c.lower() in ("final_label", "label", "genus", "family", "class")), None)
    tier_col = next((c for c in t.columns
                     if "tier" in c.lower() or "relevance" in c.lower()), None)
    if lab_col is None or tier_col is None:
        return per_class_df
    t = t[[lab_col, tier_col]].rename(columns={lab_col: "label", tier_col: "forensic_tier"})
    t["label"] = t["label"].astype(str)
    t = t.drop_duplicates("label")
    return per_class_df.merge(t, on="label", how="left")


def _torch_load(path, device):
    import torch
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _load_config():
    cfg = {}
    if CONFIG_PATH.exists():
        try:
            import yaml
            cfg = yaml.safe_load(CONFIG_PATH.read_text()) or {}
        except Exception:
            cfg = {}
    return cfg


def main():
    import torch
    import timm
    from torch.utils.data import DataLoader

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from phase2_1_1_dataset_Alaska import PollenDataset, load_class_index, PROCESSED_ROOT, SPLITS
    from phase2_1_2_transforms_Alaska import get_val_transform

    device = "cuda" if torch.cuda.is_available() else "cpu"
    enable()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if not CKPT_PATH.exists():
        sys.exit(f"checkpoint not found: {CKPT_PATH}\nRun phase2_2_train.py first.")

    ckpt = _torch_load(CKPT_PATH, device)
    class_index = ckpt.get("class_index") or load_class_index()
    class_index = {str(k): int(v) for k, v in class_index.items()}
    n_classes = len(class_index)
    index_to_label = {v: k for k, v in class_index.items()}
    arch = ckpt.get("arch", "resnet50")
    saved_epoch = ckpt.get("epoch", "?")
    saved_f1 = ckpt.get("best_f1", ckpt.get("metrics", {}).get("val_macro_f1", float("nan")))

    print(f"checkpoint: {CKPT_PATH.name} | arch {arch} | epoch {saved_epoch} "
          f"| val macro-F1 {saved_f1}")
    print(f"device: {device} | classes: {n_classes}")

    cfg = _load_config()
    num_workers = int(cfg.get("num_workers", NUM_WORKERS))

    test_ds = PollenDataset("test", class_index, transform=get_val_transform())
    sources = test_ds.df["dataset"].astype(str).to_numpy()
    loader = DataLoader(test_ds, batch_size=EVAL_BATCH, shuffle=False,
                        num_workers=num_workers, pin_memory=(device == "cuda"))

    model = timm.create_model(arch, pretrained=False, num_classes=n_classes)
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()

    autocast = null_autocast

    all_true, all_pred, all_top5, all_maxp = [], [], [], []
    n_done = 0
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            with autocast():
                logits = model(imgs)
            probs = torch.softmax(logits.float(), dim=1)
            maxp, pred = probs.max(dim=1)
            top5 = probs.topk(5, dim=1).indices
            all_true.append(labels.numpy())
            all_pred.append(pred.cpu().numpy())
            all_top5.append(top5.cpu().numpy())
            all_maxp.append(maxp.cpu().numpy())
            n_done += len(labels)
            if n_done % (EVAL_BATCH * 10) < EVAL_BATCH:
                print(f"  {n_done}/{len(test_ds)}")

    true = np.concatenate(all_true)
    pred = np.concatenate(all_pred)
    top5 = np.concatenate(all_top5)
    maxp = np.concatenate(all_maxp)
    assert len(true) == len(sources), "row alignment broke (loader must be shuffle=False)"

    summary = standard_metrics(true, pred, top5, n_classes)
    summary.update({"split": "test", "checkpoint_epoch": saved_epoch})
    pd.DataFrame([summary])[["split", "n_images", "n_classes", "top1", "top5",
                             "macro_f1", "weighted_f1", "checkpoint_epoch"]] \
        .to_csv(EVAL_DIR / "baseline_results.csv", index=False, encoding="utf-8")

    pc = per_class_table(true, pred, n_classes, index_to_label)
    pc = attach_tiers(pc)
    pc = pc.sort_values("f1").reset_index(drop=True)
    pc.to_csv(EVAL_DIR / "per_class_f1.csv", index=False, encoding="utf-8")

    cm, pairs = confusion_and_pairs(true, pred, n_classes, index_to_label)
    np.savez_compressed(EVAL_DIR / "confusion_matrix.npz",
                        cm=cm, labels=np.array([index_to_label[i] for i in range(n_classes)],
                                               dtype=object))
    pairs.to_csv(EVAL_DIR / "confusion_pairs.csv", index=False, encoding="utf-8")

    per_src = per_source_table(true, pred, top5, sources)
    per_src.to_csv(EVAL_DIR / "per_source_results.csv", index=False, encoding="utf-8")

    _plot_confusion(cm, index_to_label, pairs, EVAL_DIR)

    print("\n=== TEST-SET BASELINE ===")
    print(f"  top-1        {summary['top1']:.4f}")
    print(f"  top-5        {summary['top5']:.4f}")
    print(f"  macro-F1     {summary['macro_f1']:.4f}")
    print(f"  weighted-F1  {summary['weighted_f1']:.4f}")
    print(f"  images       {summary['n_images']:,}  | classes {summary['n_classes']}")
    print("\n  lowest 8 F1 overall (support in parens; tiny classes are noisy):")
    for _, row in pc.head(8).iterrows():
        print(f"    {row['label'][:34]:34s} F1={row['f1']:.3f} ({int(row['support'])})")
    big = pc[pc["support"] >= 20]
    print("\n  lowest 8 F1 among classes with support >= 20 (real model failures):")
    for _, row in big.head(8).iterrows():
        print(f"    {row['label'][:34]:34s} F1={row['f1']:.3f} ({int(row['support'])})")
    print("\n  top 8 confusion pairs (true -> pred, count, % of true class):")
    for _, row in pairs.head(8).iterrows():
        print(f"    {row['true_label'][:22]:22s} -> {row['pred_label'][:22]:22s} "
              f"{row['count']:4d}  {row['pct_of_true']:.1f}%")
    print("\n  per-source top-1:")
    for _, row in per_src.iterrows():
        print(f"    {row['source'][:24]:24s} {row['top1']:.3f}  (n={row['n_images']:,})")
    print(f"\nwrote outputs to {EVAL_DIR}")


# Confusion heatmaps
def _plot_confusion(cm, index_to_label, pairs, out_dir):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"(matplotlib unavailable, skipping heatmaps: {e})")
        return
    n = cm.shape[0]
    row_sums = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, np.where(row_sums == 0, 1, row_sums))
    fig, ax = plt.subplots(figsize=(10, 9))
    im = ax.imshow(norm, cmap="magma", vmin=0, vmax=1, aspect="auto")
    ax.set_title(f"Confusion matrix (row-normalised), {n} classes")
    ax.set_xlabel("Predicted class index"); ax.set_ylabel("True class index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of true class")
    fig.tight_layout()
    fig.savefig(out_dir / "confusion_matrix.png", dpi=200)
    plt.close(fig)

    if len(pairs):
        focus = []
        for _, r in pairs.iterrows():
            for k in (int(r["true_index"]), int(r["pred_index"])):
                if k not in focus:
                    focus.append(k)
            if len(focus) >= TOP_HEATMAP:
                break
        focus = sorted(focus[:TOP_HEATMAP])
        sub = cm[np.ix_(focus, focus)]
        rs = sub.sum(axis=1, keepdims=True)
        subn = np.divide(sub, np.where(rs == 0, 1, rs))
        labs = [index_to_label[i][:18] for i in focus]
        fig, ax = plt.subplots(figsize=(13, 12))
        im = ax.imshow(subn, cmap="magma", vmin=0, vmax=1)
        ax.set_xticks(range(len(focus))); ax.set_xticklabels(labs, rotation=90, fontsize=6)
        ax.set_yticks(range(len(focus))); ax.set_yticklabels(labs, fontsize=6)
        ax.set_title(f"Most-confused classes ({len(focus)}), row-normalised")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(out_dir / "confusion_matrix_top.png", dpi=200)
        plt.close(fig)


if __name__ == "__main__":
    main()
