#!/usr/bin/env python3
# What this does: re-queries GBIF in a more detailed mode.
from pathlib import Path
import argparse
import importlib.util
import json
import time

import pandas as pd

try:
    from pygbif import species as _gbif_species
except Exception:
    _gbif_species = None

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
TAX = PROJECT_ROOT / "outputs" / "taxonomy"
LOOKUP_CSV = TAX / "taxonomy_lookup.csv"
VCACHE_JSON = TAX / "gbif_cache_verbose.json"

KINGDOM = "Plantae"
COARSE_RANKS = {"KINGDOM", "PHYLUM", "CLASS", "ORDER", "FAMILY", "SUBFAMILY", "TRIBE"}
POLITE_SLEEP = 0.03
SAVE_EVERY = 25


def _load_122_helpers():
# Reuse logic
    p = Path(__file__).with_name("phase1.2.2_taxonomy_reconciliation.py")
    spec = importlib.util.spec_from_file_location("recon_1_2_2", p)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def gbif_verbose(name: str) -> dict:
# Query GBIF again in verbose mode
    if _gbif_species is None:
        raise RuntimeError('pygbif not installed. pip install pygbif "requests-cache<1.2"')
    return _gbif_species.name_backbone(name=name, kingdom=KINGDOM,
                                       strict=False, verbose=True)


def pick_genus_candidate(resp: dict):
# Pick the best match
    resp = resp or {}
    cands = [resp] + list(resp.get("alternatives") or [])
    genus_cands = [
        c for c in cands
        if (str(c.get("rank", "")).upper() == "GENUS"
            and c.get("kingdom") == KINGDOM
            and c.get("genus"))
    ]
    if not genus_cands:
        return None
    genus_cands.sort(
        key=lambda c: (str(c.get("status", "")).upper() == "ACCEPTED",
                       c.get("confidence") or 0),
        reverse=True)
    return genus_cands[0]


def load_cache():
    if VCACHE_JSON.exists():
        try:
            return json.loads(VCACHE_JSON.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_cache(c):
    VCACHE_JSON.write_text(json.dumps(c), encoding="utf-8")


def run_test():
# Recovery check more difficult names
    print("Verbose re-query test (pygbif installed:", _gbif_species is not None, ")\n")
    for name in ["Cedrus", "Hosta", "Helleborus", "Albizia",
                 "Eucalipto", "Tuber", "arecaceae"]:
        try:
            resp = gbif_verbose(name)
            cand = pick_genus_candidate(resp)
            if cand:
                print(f"  {name:12s} -> RECOVERED genus {cand.get('genus')} "
                      f"(family {cand.get('family')}, conf {cand.get('confidence')}, "
                      f"status {cand.get('status')})")
            else:
                n_alt = len(resp.get("alternatives") or [])
                print(f"  {name:12s} -> no genus candidate "
                      f"(top rank {resp.get('rank')}, {n_alt} alternatives) -> manual")
        except Exception as e:
            print(f"  {name:12s} -> ERROR {type(e).__name__}: {e}")


def main():
# Re-query GBIF
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", action="store_true")
    args = ap.parse_args()
    if args.test:
        run_test()
        return

    if not LOOKUP_CSV.exists():
        raise SystemExit(f"Missing {LOOKUP_CSV}. Run phase1.2.2 first.")
    helpers = _load_122_helpers()

    df = pd.read_csv(LOOKUP_CSV, dtype=str, keep_default_na=False)
    if "recovery_note" not in df.columns:
        df["recovery_note"] = ""

    mask = (df["matchType"].astype(str).str.upper() == "HIGHERRANK") & \
           (df["rank"].astype(str).str.upper().isin(COARSE_RANKS))
    targets = df[mask]
    print(f"coarse-rank rows to re-query: {len(targets)}")

    cache = load_cache()
    recovered = unresolved = live = 0
    t0 = time.time()

    for i, (idx, row) in enumerate(targets.iterrows(), 1):
        name = str(row["genus_raw"])
        if name in cache:
            resp = cache[name]
        else:
            try:
                resp = gbif_verbose(name)
            except Exception as e:
                resp = {"_error": f"{type(e).__name__}: {e}"}
            cache[name] = resp
            live += 1
            time.sleep(POLITE_SLEEP)
            if live % SAVE_EVERY == 0:
                save_cache(cache)

        cand = pick_genus_candidate(resp)
        if cand:
            rec = helpers.build_record(name, row.get("n_images", ""),
                                       row.get("n_datasets", ""), cand)
            for k, v in rec.items():
                if k in df.columns:
                    df.at[idx, k] = "" if v is None else str(v)
            df.at[idx, "recovery_note"] = "recovered_from_higherrank"
            recovered += 1
        else:
            df.at[idx, "recovery_note"] = "unresolved_manual"
            unresolved += 1

        if i % 25 == 0:
            print(f"  {i}/{len(targets)} (recovered {recovered}, manual {unresolved})")

    save_cache(cache)
    df.to_csv(LOOKUP_CSV, index=False, encoding="utf-8")

    print(f"\nDone in {(time.time()-t0)/60:.1f} min ({live} live calls).")
    print(f"  recovered to genus : {recovered}")
    print(f"  still manual       : {unresolved}")
    print(f"  updated: {LOOKUP_CSV}")
    print("\nRe-run phase1.2.3_export_for_review.py to regenerate the (smaller) review queue.")


if __name__ == "__main__":
    main()
