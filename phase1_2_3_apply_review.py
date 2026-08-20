#!/usr/bin/env python3
# What this does: takes manual review decisions and puts into the name table.
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
TAX = PROJECT_ROOT / "outputs" / "taxonomy"
LOOKUP_CSV = TAX / "taxonomy_lookup.csv"
REVIEW_CSV = TAX / "manual_taxonomy_review.csv"
REVIEWED_CSV = TAX / "taxonomy_lookup_reviewed.csv"
WARN_CSV = TAX / "apply_warnings.csv"

KINGDOM = "Plantae"
OUT_FIELDS = ["genus_raw", "source", "decision", "status", "label_rank",
              "final_label", "accepted_genus", "family", "n_images",
              "kingdom", "matchType", "recovery_note"]


def _s(v):
    return "" if v is None else str(v).strip()


def resolve(row, decision, corrected):
# Final label
    acc = _s(row.get("accepted_genus"))
    fam = _s(row.get("family"))
    kingdom = _s(row.get("kingdom"))
    usable_genus = bool(acc) and kingdom == KINGDOM

    if decision == "exclude":
        return "excluded", "", ""
    if decision == "redirect":
        if corrected:
            return "redirected", "genus", corrected
        return "excluded_no_target", "", ""        
    if decision == "keep_as_family":
        if fam:
            return "kept_family", "family", fam
        return "excluded_no_family", "", ""
    if decision == "accept":
        if usable_genus:
            return "accepted", "genus", acc
        return "excluded_no_genus", "", ""
# Empty / no decision
    if usable_genus:
        return "accepted_blank", "genus", acc
    return "excluded_blank_no_genus", "", ""


def main():
# Apply label decision
    if not LOOKUP_CSV.exists():
        raise SystemExit(f"Missing {LOOKUP_CSV}")
    if not REVIEW_CSV.exists():
        raise SystemExit(f"Missing {REVIEW_CSV}")

    lk = pd.read_csv(LOOKUP_CSV, dtype=str, keep_default_na=False)
    rv = pd.read_csv(REVIEW_CSV, dtype=str, keep_default_na=False)

    decisions = {r["genus_raw"]: (_s(r.get("decision")).lower(),
                                  _s(r.get("corrected_genus")))
                 for _, r in rv.iterrows()}
    review_set = set(decisions)

    out_rows, warnings = [], []
    for _, row in lk.iterrows():
        g = row["genus_raw"]
        if g in review_set:
            decision, corrected = decisions[g]
            source = "review"
        else:
            decision, corrected, source = "", "", "auto"
        status, label_rank, final_label = resolve(row, decision, corrected)

        rec = {
            "genus_raw": g, "source": source, "decision": decision,
            "status": status, "label_rank": label_rank, "final_label": final_label,
            "accepted_genus": _s(row.get("accepted_genus")),
            "family": _s(row.get("family")), "n_images": _s(row.get("n_images")),
            "kingdom": _s(row.get("kingdom")), "matchType": _s(row.get("matchType")),
            "recovery_note": _s(row.get("recovery_note")),
        }
        out_rows.append(rec)
        if status.startswith("excluded") and status != "excluded":
            warnings.append(rec)

    out = pd.DataFrame(out_rows, columns=OUT_FIELDS)
    out.to_csv(REVIEWED_CSV, index=False, encoding="utf-8")
    if warnings:
        pd.DataFrame(warnings, columns=OUT_FIELDS).to_csv(WARN_CSV, index=False, encoding="utf-8")

    out["n_images_int"] = pd.to_numeric(out["n_images"], errors="coerce").fillna(0).astype(int)
    kept = out[out["final_label"] != ""]
    genus_labels = kept[kept.label_rank == "genus"]["final_label"].nunique()
    family_labels = kept[kept.label_rank == "family"]["final_label"].nunique()

    print(f"raw labels processed : {len(out):,}")
    print("  by status:")
    for s, n in out["status"].value_counts().items():
        print(f"     {s}: {n}")
    print(f"\n  raw labels kept      : {len(kept):,}  "
          f"({kept['n_images_int'].sum():,} images)")
    print(f"  excluded             : {(out['final_label']=='').sum():,}")
    print(f"\n  distinct final labels: {kept['final_label'].nunique():,}  "
          f"({genus_labels} genus + {family_labels} family)")
    print(f"\n  written: {REVIEWED_CSV}")
    if warnings:
        print(f"  WARNINGS ({len(warnings)}) — excluded without an explicit "
              f"exclude decision: {WARN_CSV}")


if __name__ == "__main__":
    main()
