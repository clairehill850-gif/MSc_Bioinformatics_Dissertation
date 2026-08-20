# What this does: builds the cost matrix.
import os
import sys
import json
import glob
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"
CLASS_INDEX = PROJECT_ROOT / "outputs" / "curation" / "splits" / "class_index_final.json"
OVERRIDES = EVAL_DIR / "confusion_annotation_seed.csv"

COST_SAME_FAMILY = 0.2
COST_HIGH_DISTINCT = 2.0
COST_DEFAULT = 1.0
# Identification levels that allow the higher 2.0 cost
DISTINGUISHABLE = ("genus", "species")   


def _find(patterns):
    for pat in patterns:
        hits = glob.glob(str(PROJECT_ROOT / "outputs" / "**" / pat), recursive=True)
        if hits:
            return Path(sorted(hits, key=len)[0])
    return None


def _pick_col(cols, *, label=False, kind=None):
    low = {c.lower(): c for c in cols}
    if label:
        for k in ("final_label", "label", "taxon", "class", "genus", "family", "name"):
            if k in low:
                return low[k]
    if kind == "tier":
        for c in cols:
            if "tier" in c.lower() or "relevance" in c.lower():
                return c
    if kind == "resolution":
        for c in cols:
            cl = c.lower()
            if "resolution" in cl or "id_res" in cl or cl in ("resolution", "id_resolution"):
                return c
    if kind == "family":
        for c in cols:
            if c.lower() == "family":
                return c
        for c in cols:
            if "family" in c.lower():
                return c
    return None


def _load_map(path, kind):
    if path is None or not Path(path).exists():
        return {}, None, None
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}, None, None
    lab = _pick_col(df.columns, label=True)
    val = _pick_col(df.columns, kind=kind)
    if lab is None or val is None:
        return {}, lab, val
    m = {}
    for _, r in df.iterrows():
        if pd.notna(r[lab]) and pd.notna(r[val]):
            m[str(r[lab])] = str(r[val])
    return m, lab, val


def _load_overrides(path):
# Load any manual cost overrides
    if not Path(path).exists():
        return {}
    try:
        df = pd.read_csv(path)
    except Exception:
        return {}
    need = {"true_label", "pred_label", "cost_suggestion"}
    if not need.issubset(df.columns):
        return {}
    out = {}
    for _, r in df.iterrows():
        v = r["cost_suggestion"]
        if pd.isna(v) or str(v).strip() == "":
            continue
        try:
            out[(str(r["true_label"]), str(r["pred_label"]))] = float(v)
        except (ValueError, TypeError):
            continue
    return out


def build_cost_matrix(labels, fam, tier, idres, overrides, observed=None, min_count=1):
# Build the cost matrix
    n = len(labels)
    cost = np.full((n, n), COST_DEFAULT, dtype=np.float32)
    np.fill_diagonal(cost, 0.0)
    long_rows = []
    counts = {"manual": 0, "same_family": 0, "high_tier_distinguishable": 0}

    def distinguishable(lab):
        r = idres.get(lab, "").lower()
        return any(k in r for k in DISTINGUISHABLE)

    for i, li in enumerate(labels):
        fi = fam.get(li)
        is_high = tier.get(li, "").strip().lower() == "high"
        i_distinct = distinguishable(li)
        for j, lj in enumerate(labels):
            if i == j:
                continue
            reason = None
            same_fam = fi is not None and fi == fam.get(lj)
            occurs = False if observed is None else (observed[i, j] >= min_count)
            if (li, lj) in overrides:
                c = overrides[(li, lj)]; reason = "manual"; counts["manual"] += 1
            elif same_fam:
                c = COST_SAME_FAMILY; reason = "same_family"; counts["same_family"] += 1
            elif is_high and i_distinct and occurs:
                c = COST_HIGH_DISTINCT; reason = "high_tier_distinguishable"
                counts["high_tier_distinguishable"] += 1
            else:
                c = COST_DEFAULT
            cost[i, j] = c
            if reason is not None or c != COST_DEFAULT:
                long_rows.append({"true_label": li, "pred_label": lj,
                                  "cost": c, "reason": reason or "default"})
    return cost, pd.DataFrame(long_rows), counts


def main():
# Find the Phase 1 tables, build the cost matrix, and save the outputs
    if not CLASS_INDEX.exists():
        sys.exit(f"class index not found: {CLASS_INDEX}")
    class_index = json.loads(CLASS_INDEX.read_text())
    labels = [None] * len(class_index)
    for lab, idx in class_index.items():
        labels[int(idx)] = str(lab)
    n = len(labels)

    tier_path = _find(["*forensic*relevance*tier*.csv", "*forensic*tier*.csv", "*tier*.csv"])
    idres_path = _find(["*id_resolution*.csv", "*id*resolution*.csv", "*resolution*.csv"])
    fam_path = _find(["*taxonomy*hierarchy*.csv", "*hierarchy*.csv", "*taxonomy*.csv"])

    tier, tlab, tval = _load_map(tier_path, "tier")
    idres, ilab, ival = _load_map(idres_path, "resolution")
    fam, flab, fval = _load_map(fam_path, "family")
    overrides = _load_overrides(OVERRIDES)

    cm_path = EVAL_DIR / "confusion_matrix.npz"
    observed = None
    if cm_path.exists():
        try:
            z = np.load(cm_path, allow_pickle=True)
            cm = z["matrix"] if "matrix" in z.files else z[z.files[0]]
            cm_labels = [str(x) for x in z["labels"]] if "labels" in z.files else labels
            if list(cm_labels) == list(labels):
                observed = cm
            else:  
                idx = {l: k for k, l in enumerate(cm_labels)}
                observed = np.zeros((n, n), dtype=cm.dtype)
                for a, la in enumerate(labels):
                    for b, lb in enumerate(labels):
                        if la in idx and lb in idx:
                            observed[a, b] = cm[idx[la], idx[lb]]
        except Exception as e:
            print(f"  ! could not load confusion matrix ({e}); 2.0 rule will NOT fire")

    print("inputs located:")
    print(f"  tiers       : {tier_path}  [{tlab} / {tval}]  ({len(tier)} mapped)")
    print(f"  id_resolution: {idres_path}  [{ilab} / {ival}]  ({len(idres)} mapped)")
    print(f"  hierarchy   : {fam_path}  [{flab} / {fval}]  ({len(fam)} mapped)")
    print(f"  confusion mx : {cm_path if observed is not None else 'NOT FOUND - 2.0 rule disabled'}")
    print(f"  overrides   : {OVERRIDES.name if OVERRIDES.exists() else 'none'}  ({len(overrides)} cells)")
    if observed is None:
        print("  ! no baseline confusion matrix -> high-cost (2.0) cells come ONLY from manual "
              "overrides; auto 2.0 is suppressed to avoid blanket-row bloat.")
# Coverage warnings
    miss_fam = sum(1 for l in labels if l not in fam)
    miss_tier = sum(1 for l in labels if l not in tier)
    if miss_fam:
        print(f"  ! {miss_fam}/{n} classes have no family (treated as own family; no same-family 0.2)")
    if miss_tier:
        print(f"  ! {miss_tier}/{n} classes have no tier (treated as not-High)")
    if not fam:
        print("  ! WARNING: no family map -> same-family 0.2 rule is INACTIVE. "
              "Check the hierarchy file/columns above.")

    cost, long_df, counts = build_cost_matrix(labels, fam, tier, idres, overrides,
                                              observed=observed, min_count=1)

    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(EVAL_DIR / "confusion_cost_matrix.npz",
                        cost=cost, labels=np.array(labels, dtype=object))
    long_df.sort_values(["cost", "true_label"]).to_csv(
        EVAL_DIR / "confusion_cost_matrix_long.csv", index=False, encoding="utf-8")

    off = n * n - n
    print(f"\ncost matrix {n}x{n} built (off-diagonal cells: {off:,})")
    print(f"  0.2 same-family            : {counts['same_family']:,}")
    print(f"  2.0 high-tier distinguishable: {counts['high_tier_distinguishable']:,}")
    print(f"  manual overrides applied   : {counts['manual']:,}")
    print(f"  1.0 default (remainder)    : {off - counts['same_family'] - counts['high_tier_distinguishable'] - counts['manual']:,}")
    print(f"\n  sample same-family pairs (cost 0.2):")
    for _, r in long_df[long_df.reason == "same_family"].head(6).iterrows():
        print(f"     {r['true_label'][:24]:24s} -> {r['pred_label'][:24]:24s}")
    if counts["high_tier_distinguishable"]:
        print(f"  sample high-cost pairs (cost 2.0):")
        for _, r in long_df[long_df.reason == "high_tier_distinguishable"].head(6).iterrows():
            print(f"     {r['true_label'][:24]:24s} -> {r['pred_label'][:24]:24s}")
    print(f"\nwrote confusion_cost_matrix.npz + confusion_cost_matrix_long.csv to {EVAL_DIR}")


if __name__ == "__main__":
    main()
