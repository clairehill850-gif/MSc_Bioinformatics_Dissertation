#!/usr/bin/env python3
# What this does: checks every plant name against GBIF. Records the official name, its rank and family, and flags stuff needing manual review. Cached so it can stop and restart.
from pathlib import Path
import argparse
import json
import time

import pandas as pd

try:
    from pygbif import species as _gbif_species
except Exception:
    _gbif_species = None

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
TAX = PROJECT_ROOT / "outputs" / "taxonomy"
RAW_LABELS = TAX / "raw_labels.txt"
COUNTS_CSV = TAX / "raw_label_counts.csv"
LOOKUP_CSV = TAX / "taxonomy_lookup.csv"
CACHE_JSON = TAX / "gbif_cache.json"

KINGDOM = "Plantae"
# Exact matches below this flagged for manual review
CONFIDENCE_REVIEW_BELOW = 90       
SAVE_CACHE_EVERY = 50
# Pause between calls (to be polite!)
POLITE_SLEEP = 0.02                

LOOKUP_FIELDS = [
    "genus_raw", "n_images", "n_datasets",
    "matchType", "confidence", "rank", "status", "synonym",
    "accepted_canonical", "accepted_genus", "family", "order", "class",
    "kingdom", "usageKey", "needs_review", "review_reason",
]


def gbif_lookup(name: str) -> dict:
# Make a live call to the GBIF database
    if _gbif_species is None:
        raise RuntimeError(
            "pygbif not installed. Run: pip install pygbif \"requests-cache<1.2\"")
    return _gbif_species.name_backbone(name=name, kingdom=KINGDOM, strict=False)


def build_record(name: str, n_images, n_datasets, resp: dict) -> dict:
# GBIF's reply
    resp = resp or {}
    rec = {
        "genus_raw": name,
        "n_images": n_images,
        "n_datasets": n_datasets,
        "matchType": resp.get("matchType", "NONE"),
        "confidence": resp.get("confidence", ""),
        "rank": resp.get("rank", ""),
        "status": resp.get("status", ""),
        "synonym": bool(resp.get("synonym", False)),
        "accepted_canonical": resp.get("canonicalName", "") or resp.get("scientificName", ""),
        "accepted_genus": resp.get("genus", ""),
        "family": resp.get("family", ""),
        "order": resp.get("order", ""),
        "class": resp.get("class", ""),
        "kingdom": resp.get("kingdom", ""),
        "usageKey": resp.get("usageKey", ""),
    }
    needs, reason = classify_review(rec)
    rec["needs_review"] = needs
    rec["review_reason"] = reason
    return rec


def classify_review(rec: dict):
# Decide if manual review required
    reasons = []
    mt = (rec.get("matchType") or "NONE").upper()
    rank = (rec.get("rank") or "").upper()
    status = (rec.get("status") or "").upper()
    kingdom = (rec.get("kingdom") or "")
    conf = rec.get("confidence")
    try:
        conf = float(conf)
    except (TypeError, ValueError):
        conf = None
    if mt == "NONE":
        reasons.append("no_match")
    if mt == "FUZZY":
        reasons.append("fuzzy_match")          
    if mt == "HIGHERRANK" or (rank and rank != "GENUS"):
        reasons.append(f"rank_{rank.lower() or 'higher'}")  
    if rec.get("synonym") or status == "SYNONYM":
        reasons.append("synonym_redirect")
    if kingdom and kingdom != KINGDOM:
        reasons.append(f"non_plant_{kingdom.lower()}")
    if mt == "EXACT" and conf is not None and conf < CONFIDENCE_REVIEW_BELOW:
        reasons.append("low_confidence")

    return (len(reasons) > 0, ";".join(reasons))


def suggested_action(reason: str) -> str:
# Review suggestion
    r = reason or ""
    if "no_match" in r:               return "correct_spelling_or_exclude"
    if r.startswith("non_plant") or "non_plant" in r: return "exclude_non_plant"
    if "rank_" in r:                  return "family_level_label_keep_or_exclude"
    if "synonym_redirect" in r:       return "accept_redirect_to_accepted_name"
    if "fuzzy_match" in r:            return "confirm_fuzzy_correction"
    if "low_confidence" in r:         return "verify_low_confidence_match"
    return "review"


def load_cache():
# Load previous GBIF replies
    if CACHE_JSON.exists():
        try:
            return json.loads(CACHE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(cache):
# Save GBIF replies for reuse
    CACHE_JSON.write_text(json.dumps(cache), encoding="utf-8")


def run_test():
# Check that GBIF is available
    print("GBIF connectivity test (pygbif installed:", _gbif_species is not None, ")")
    for name in ["Pinus", "Aeculus"]:
        try:
            r = gbif_lookup(name)
            print(f"  {name}: matchType={r.get('matchType')} "
                  f"-> {r.get('canonicalName')} (rank={r.get('rank')}, "
                  f"conf={r.get('confidence')}, family={r.get('family')})")
        except Exception as e:
            print(f"  {name}: ERROR {type(e).__name__}: {e}")


def main():
# Check all the names against GBIF and save the results
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true", help="connectivity check only")
    args = ap.parse_args()
    if args.test:
        run_test()
        return

    if not RAW_LABELS.exists():
        raise SystemExit(f"Missing {RAW_LABELS}. Run phase1.2.1 first.")

    names = [n.strip() for n in RAW_LABELS.read_text(encoding="utf-8").splitlines()
             if n.strip()]

    counts = {}
    if COUNTS_CSV.exists():
        cdf = pd.read_csv(COUNTS_CSV)
        counts = {r["genus"]: (r["n_images"], r["n_datasets"])
                  for _, r in cdf.iterrows()}

    cache = load_cache()
    print(f"Reconciling {len(names):,} labels (cached: {len(cache):,})")

    rows = []
    t0 = time.time()
    live_calls = 0
    for i, name in enumerate(names, 1):
        if name in cache:
            resp = cache[name]
        else:
            try:
                resp = gbif_lookup(name)
            except Exception as e:
                resp = {"matchType": "NONE", "_error": f"{type(e).__name__}: {e}"}
            cache[name] = resp
            live_calls += 1
            time.sleep(POLITE_SLEEP)
            if live_calls % SAVE_CACHE_EVERY == 0:
                save_cache(cache)

        n_img, n_ds = counts.get(name, ("", ""))
        rows.append(build_record(name, n_img, n_ds, resp))

        if i % 250 == 0:
            print(f"  {i}/{len(names)}  ({live_calls} live calls)")

    save_cache(cache)
    out = pd.DataFrame(rows, columns=LOOKUP_FIELDS)
    out.to_csv(LOOKUP_CSV, index=False, encoding="utf-8")

    dt = time.time() - t0
    print(f"\nReconciliation complete in {dt/60:.1f} min "
          f"({live_calls} live GBIF calls, rest cached).")
    print(f"  rows                : {len(out):,}")
    print(f"  needs review        : {int(out.needs_review.sum()):,}")
    print(f"  auto-accepted       : {int((~out.needs_review).sum()):,}")
    print("  match types:")
    for mt, n in out["matchType"].value_counts().items():
        print(f"     {mt}: {n}")
    print(f"  written: {LOOKUP_CSV}")


if __name__ == "__main__":
    main()
