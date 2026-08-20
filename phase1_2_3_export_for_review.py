#!/usr/bin/env python3
# What this does: splits the checked names into auto-accepted and manual review.
from pathlib import Path
import importlib.util
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
TAX = PROJECT_ROOT / "outputs" / "taxonomy"
LOOKUP_CSV = TAX / "taxonomy_lookup.csv"
REVIEW_CSV = TAX / "manual_taxonomy_review.csv"
AUTO_CSV = TAX / "auto_accept_log.csv"

THRESHOLD = 20

DECISION_VOCAB = "accept | redirect | keep_as_family | exclude"


def _load_suggested_action():
# Borrow "suggested action" logic
    p = Path(__file__).with_name("phase1.2.2_taxonomy_reconciliation.py")
    spec = importlib.util.spec_from_file_location("recon_1_2_2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.suggested_action


def main():
# Split into auto-accept vs review
    if not LOOKUP_CSV.exists():
        raise SystemExit(f"Missing {LOOKUP_CSV}. Run phase1.2.2 first.")
    suggested_action = _load_suggested_action()

    df = pd.read_csv(LOOKUP_CSV, low_memory=False)
    def _strip_resolved_rank(reason):
        toks = [t for t in str(reason).split(";") if t and t != "rank_genus"]
        return ";".join(toks)

    df["review_reason"] = df["review_reason"].fillna("").map(_strip_resolved_rank)
    df["needs_review"] = df["review_reason"].str.strip().ne("")
    df["n_images"] = pd.to_numeric(df["n_images"], errors="coerce").fillna(0).astype(int)

    review = df[df["needs_review"]].copy()
    auto = df[~df["needs_review"]].copy()

# Ordering
    review["above_threshold"] = review["n_images"] >= THRESHOLD
    review = review.sort_values(["above_threshold", "n_images"],
                                ascending=[False, False])
    review["suggested_action"] = review["review_reason"].apply(suggested_action)

# Keep any decisions in review file
    prior = {}
    if REVIEW_CSV.exists():
        try:
            pdf = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)
            for _, r in pdf.iterrows():
                vals = {c: r.get(c, "") for c in ("decision", "corrected_genus", "notes")}
                if any(str(v).strip() for v in vals.values()):
                    prior[str(r.get("genus_raw", ""))] = vals
        except Exception:
            prior = {}

    def _prior(g, col):
        return prior.get(str(g), {}).get(col, "")

    review.insert(0, "decision",        review["genus_raw"].map(lambda g: _prior(g, "decision")))
    review.insert(1, "corrected_genus", review["genus_raw"].map(lambda g: _prior(g, "corrected_genus")))
    review.insert(2, "notes",           review["genus_raw"].map(lambda g: _prior(g, "notes")))
    n_preserved = int((review["decision"].astype(str).str.strip() != "").sum())

    front = ["decision", "corrected_genus", "notes", "genus_raw", "n_images",
             "above_threshold", "suggested_action", "review_reason",
             "matchType", "confidence", "rank", "status", "accepted_genus",
             "family"]
    rest = [c for c in review.columns if c not in front]
    review = review[front + rest]

    def _safe_to_csv(frame, path):
        try:
            frame.to_csv(path, index=False, encoding="utf-8")
        except PermissionError:
            raise SystemExit(
                f"\nERROR: cannot write {path.name} — it's open in Excel "
                f"(or another program). Close it and re-run.")

    _safe_to_csv(review, REVIEW_CSV)
    _safe_to_csv(auto, AUTO_CSV)

    n_priority = int((review["above_threshold"]).sum())
    print(f"review queue   : {len(review):,} labels  "
          f"({n_priority:,} above threshold — review these first)")
    print(f"auto-accepted  : {len(auto):,} labels")
    if n_preserved:
        print(f"preserved      : {n_preserved:,} prior decisions carried over")
    print(f"\nDecision vocabulary for the `decision` column:\n  {DECISION_VOCAB}")
    print("\nReason breakdown in review queue:")
    for r, n in review["review_reason"].value_counts().items():
        print(f"  {r}: {n}")
    print(f"\nEDIT: {REVIEW_CSV}")
    print(f"audit: {AUTO_CSV}")


if __name__ == "__main__":
    main()
