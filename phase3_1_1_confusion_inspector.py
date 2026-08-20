# What this does: starter table for annotating mix-ups.

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
NPZ_PATH = EVAL_DIR / "confusion_matrix.npz"
PER_CLASS_PATH = EVAL_DIR / "per_class_f1.csv"
SEED_OUT = EVAL_DIR / "confusion_annotation_seed.csv"

SEED_MIN_COUNT = 2    
TOPK = 6                 
DEFAULT_FOCUS = ["Plantago", "Quercus", "Stachys", "Aloe", "Cecropia", "Plectranthus"]


def load_cm(npz_path=NPZ_PATH):
# Load the confusion matrix and class order
    if not npz_path.exists():
        sys.exit(f"not found: {npz_path}\nRun phase2_3_evaluate.py first.")
    data = np.load(npz_path, allow_pickle=True)
    cm = data["cm"].astype(np.int64)
    labels = [str(x) for x in data["labels"].tolist()]
    return cm, labels


def load_support_f1(per_class_path=PER_CLASS_PATH):
    if not per_class_path.exists():
        return {}
    try:
        df = pd.read_csv(per_class_path)
    except Exception:
        return {}
    tier_col = "forensic_tier" if "forensic_tier" in df.columns else None
    out = {}
    for _, r in df.iterrows():
        out[str(r["label"])] = {
            "f1": float(r.get("f1", float("nan"))),
            "support": int(r.get("support", 0)),
            "tier": (str(r[tier_col]) if tier_col and pd.notna(r[tier_col]) else ""),
        }
    return out


def _resolve(name, labels):
# Look up a class name
    lut = {l.lower(): l for l in labels}
    if name.lower() in lut:
        return lut[name.lower()]
    near = [l for l in labels if name.lower() in l.lower()]
    hint = f" Did you mean: {', '.join(near[:6])}?" if near else ""
    print(f"  [skip] '{name}' is not a class label.{hint}")
    return None


def class_profile(cm, labels, name, ctx, topk=TOPK):
# Print what a class gets confused with
    real = _resolve(name, labels)
    if real is None:
        return
    i = labels.index(real)
    row = cm[i].copy()
    col = cm[:, i].copy()
    correct = int(cm[i, i])
    support = int(row.sum())           
    predicted_as = int(col.sum())            
    recall = correct / support if support else 0.0
    precision = correct / predicted_as if predicted_as else 0.0
    info = ctx.get(real, {})
    tier = info.get("tier", "")
    f1 = info.get("f1", float("nan"))

    header = f"=== {real} ==="
    meta = f"support {support} | recall {recall:.3f} | precision {precision:.3f}"
    if not (isinstance(f1, float) and np.isnan(f1)):
        meta += f" | F1 {f1:.3f}"
    if tier:
        meta += f" | tier {tier}"
    print(f"\n{header}\n  {meta}")

# Where do true grains end up
    row[i] = 0
    order = np.argsort(row)[::-1]
    leaks = [(labels[j], int(row[j])) for j in order if row[j] > 0][:topk]
    if leaks:
        print(f"  misread AS (true {real} -> predicted X):")
        for lab, c in leaks:
            print(f"     -> {lab[:30]:30s} {c:3d}  ({100*c/support:.1f}% of true {real})")
    else:
        print(f"  misread AS: none (all {support} predicted correctly)")

# What is wrongly assigned
    col[i] = 0
    order = np.argsort(col)[::-1]
    intruders = [(labels[j], int(col[j])) for j in order if col[j] > 0][:topk]
    if intruders:
        print(f"  mistaken FOR (true X -> predicted {real}):")
        for lab, c in intruders:
            denom = predicted_as if predicted_as else 1
            print(f"     <- {lab[:30]:30s} {c:3d}  ({100*c/denom:.1f}% of {real} predictions)")
    else:
        print(f"  mistaken FOR: none")


def build_seed(cm, labels, ctx, min_count=SEED_MIN_COUNT):
    row_tot = cm.sum(axis=1)
    col_tot = cm.sum(axis=0)
    rows = []
    n = len(labels)
    for i in range(n):
        for j in range(n):
            if i == j or cm[i, j] < min_count:
                continue
            ti = ctx.get(labels[i], {})
            rows.append({
                "true_label": labels[i],
                "pred_label": labels[j],
                "count": int(cm[i, j]),
                "pct_of_true": round(100 * cm[i, j] / row_tot[i], 1) if row_tot[i] else 0.0,
                "pct_of_pred": round(100 * cm[i, j] / col_tot[j], 1) if col_tot[j] else 0.0,
                "true_support": int(row_tot[i]),
                "true_tier": ti.get("tier", ""),
# Y / N
                "morphologically_expected": "",
# Y / N   
                "same_family": "",          
                "forensic_consequence": "",   
                "cost_suggestion": "",            
            })
    df = pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    return df


def main():
# Print confusion profiles
    args = [a for a in sys.argv[1:]]
    cm, labels = load_cm()
    ctx = load_support_f1()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    seed_only = "--seed-only" in args
    args = [a for a in args if a != "--seed-only"]

    focus = None
    if "--worst" in args:
        k = 10
        idx = args.index("--worst")
        if idx + 1 < len(args) and args[idx + 1].isdigit():
            k = int(args[idx + 1])
        if ctx:
            big = sorted(((v["f1"], lab) for lab, v in ctx.items() if v["support"] >= 20))
            focus = [lab for _, lab in big[:k]]
        else:
            print("  (per_class_f1.csv not found; cannot rank by F1)")
            focus = []
    elif args:
        focus = args

    if not seed_only:
        targets = focus if focus is not None else DEFAULT_FOCUS
        for name in targets:
            class_profile(cm, labels, name, ctx)

    seed = build_seed(cm, labels, ctx)
    seed.to_csv(SEED_OUT, index=False, encoding="utf-8")
    print(f"\nannotation seed: {len(seed)} off-diagonal cells (count >= {SEED_MIN_COUNT}) "
          f"-> {SEED_OUT}")


if __name__ == "__main__":
    main()
