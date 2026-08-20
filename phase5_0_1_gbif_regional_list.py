#!/usr/bin/env python3
# What this does: Fairbanks (ALASKA) GBIF list.

import argparse
import json
import math
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Moving to Alaska
PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
HIER = PROJECT_ROOT / "outputs" / "taxonomy" / "taxonomy_hierarchy.csv"
TRAINED = PROJECT_ROOT / "outputs" / "eval" / "per_class_f1_convnext_tiny.csv"
RAW_COUNTS = PROJECT_ROOT / "outputs" / "taxonomy" / "raw_label_counts.csv"
OUT = PROJECT_ROOT / "outputs" / "alaska"

SITE_NAME = "Fairbanks, Alaska"
SITE_LAT = 64.8378
SITE_LON = -147.7164
# Radius from central point
RADIUS_KM = 100.0
MIN_OCCURRENCES = 5
YEAR_FROM = 1950
KINGDOM_KEY = 6

BASIS_OF_RECORD = [
    "PRESERVED_SPECIMEN",
    "HUMAN_OBSERVATION",
    "MACHINE_OBSERVATION",
    "OBSERVATION",
    "MATERIAL_SAMPLE",
]

EARTH_R_KM = 6371.0
# GBIF taxa limit
FACET_LIMIT = 1200


# GBIF area outline
def build_wkt_circle(lat, lon, radius_km, n_points=64):
    pts = []
    for i in range(n_points):
        bearing = 2.0 * math.pi * i / n_points
        d = radius_km / EARTH_R_KM
        lat1 = math.radians(lat)
        lon1 = math.radians(lon)
        lat2 = math.asin(math.sin(lat1) * math.cos(d) +
                         math.cos(lat1) * math.sin(d) * math.cos(bearing))
        lon2 = lon1 + math.atan2(math.sin(bearing) * math.sin(d) * math.cos(lat1),
                                 math.cos(d) - math.sin(lat1) * math.sin(lat2))
        pts.append((round(math.degrees(lon2), 6), round(math.degrees(lat2), 6)))
    pts.reverse()
    pts.append(pts[0])
    coords = ", ".join(f"{x} {y}" for x, y in pts)
    return f"POLYGON(({coords}))"


# Assemble query
def build_predicate(wkt):
    return {
        "type": "and",
        "predicates": [
            {"type": "equals", "key": "KINGDOM_KEY", "value": str(KINGDOM_KEY)},
            {"type": "within", "geometry": wkt},
            {"type": "equals", "key": "HAS_COORDINATE", "value": "true"},
            {"type": "equals", "key": "HAS_GEOSPATIAL_ISSUE", "value": "false"},
            {"type": "equals", "key": "OCCURRENCE_STATUS", "value": "PRESENT"},
            {"type": "greaterThanOrEquals", "key": "YEAR", "value": str(YEAR_FROM)},
            {"type": "in", "key": "BASIS_OF_RECORD", "values": BASIS_OF_RECORD},
        ],
    }


# Query GBIF for record no.s
def fetch_genus_counts(wkt):
    from pygbif import occurrences

    params = dict(
        kingdomKey=KINGDOM_KEY,
        geometry=wkt,
        hasCoordinate=True,
        hasGeospatialIssue=False,
        occurrenceStatus="PRESENT",
        year=f"{YEAR_FROM},{datetime.now().year}",
        basisOfRecord=BASIS_OF_RECORD,
        facet="genusKey",
        facetLimit=FACET_LIMIT,
        limit=0,
    )
    res = occurrences.search(**params)

    total = res.get("count", 0)
    facets = res.get("facets", [])
    counts = []
    for f in facets:
        if str(f.get("field", "")).upper() in ("GENUS_KEY", "GENUSKEY"):
            counts = f.get("counts", [])
            break
    if not counts:
        raise SystemExit("GBIF returned no genusKey facet; check the query parameters.")
    if len(counts) >= FACET_LIMIT:
        print(f"  WARNING: facet returned {len(counts)} genera, at the limit of {FACET_LIMIT}.")
        print("  Some genera may be missing. Reduce RADIUS_KM or raise FACET_LIMIT.")
    out = [{"genus_key": int(c["name"]), "n_occurrences": int(c["count"])} for c in counts]
    return out, total


# Taxa codes to names
def resolve_genera(rows):
    from pygbif import species

    resolved = []
    for i, r in enumerate(rows, 1):
        try:
            u = species.name_usage(key=r["genus_key"])
        except Exception as e:
            print(f"  ERR key {r['genus_key']}: {e}")
            continue
        name = u.get("genus") or u.get("canonicalName")
        kingdom = u.get("kingdom")
        if not name:
            continue
        resolved.append({
            "gbif_genus": name,
            "gbif_family": u.get("family"),
            "gbif_kingdom": kingdom,
            "genus_key": r["genus_key"],
            "n_occurrences": r["n_occurrences"],
            "kingdom_ok": kingdom == "Plantae",
        })
        if i % 100 == 0:
            print(f"  resolved {i}/{len(rows)}")
    return pd.DataFrame(resolved)


# Match the GBIF to dataset
def map_to_labels(gdf, hier, trained_labels, raw_counts):
    lut = (hier.dropna(subset=["accepted_genus"])
               .drop_duplicates(subset=["accepted_genus"])
               .set_index("accepted_genus")[["final_label", "label_rank", "family"]])

# Attach label
    gdf = gdf.merge(lut, how="left", left_on="gbif_genus", right_index=True)
    gdf["in_hierarchy"] = gdf["final_label"].notna()
    gdf["is_trained_class"] = gdf["final_label"].isin(trained_labels)

    raw = raw_counts.set_index("genus")["n_images"] if raw_counts is not None else None
    gdf["raw_images_for_genus"] = (gdf["gbif_genus"].map(raw).fillna(0).astype(int)
                                   if raw is not None else 0)
    return gdf


# Build class list and gap report
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="write the WKT and predicate only; make no GBIF calls")
    args = ap.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    wkt = build_wkt_circle(SITE_LAT, SITE_LON, RADIUS_KM)
    predicate = build_predicate(wkt)

# Record query for citation
    query_record = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "site_name": SITE_NAME,
        "site_lat": SITE_LAT,
        "site_lon": SITE_LON,
        "radius_km": RADIUS_KM,
        "min_occurrences": MIN_OCCURRENCES,
        "year_from": YEAR_FROM,
        "basis_of_record": BASIS_OF_RECORD,
        "kingdom_key": KINGDOM_KEY,
        "wkt": wkt,
        "download_predicate": predicate,
        "gbif_download_doi": "PASTE_DOI_HERE",
    }
    (OUT / "gbif_query_record.json").write_text(json.dumps(query_record, indent=2),
                                                encoding="utf-8")
    (OUT / "gbif_download_predicate.json").write_text(json.dumps(predicate, indent=2),
                                                      encoding="utf-8")
    print(f"written: gbif_query_record.json, gbif_download_predicate.json -> {OUT}")
    print(f"  site {SITE_NAME}  radius {RADIUS_KM} km  min_occ {MIN_OCCURRENCES}  year >= {YEAR_FROM}")

# Dry run stops here
    if args.dry_run:
        print("\ndry run: no GBIF calls made.")
        return

    for p in (HIER, TRAINED):
        if not p.exists():
            raise SystemExit(f"missing required input: {p}")

    print("\nquerying GBIF ...")
    rows, total = fetch_genus_counts(wkt)
    print(f"  {total:,} occurrences in radius; {len(rows)} genera before filtering")

# Drop low record taxa 
    rows = [r for r in rows if r["n_occurrences"] >= MIN_OCCURRENCES]
    print(f"  {len(rows)} genera with >= {MIN_OCCURRENCES} occurrences")

# Codes to names, exclude non-plants
    print("\nresolving genus keys and verifying kingdom ...")
    gdf = resolve_genera(rows)
    n_bad = int((~gdf["kingdom_ok"]).sum())
    if n_bad:
        print(f"  dropped {n_bad} non-Plantae genera (homonym check):")
        for r in gdf[~gdf["kingdom_ok"]].itertuples():
            print(f"     {r.gbif_genus} ({r.gbif_kingdom})")
    gdf = gdf[gdf["kingdom_ok"]].copy()

    hier = pd.read_csv(HIER)
    trained = pd.read_csv(TRAINED)
    trained_labels = set(trained["label"].astype(str))
    raw_counts = pd.read_csv(RAW_COUNTS) if RAW_COUNTS.exists() else None

    gdf = map_to_labels(gdf, hier, trained_labels, raw_counts)
    gdf = gdf.sort_values("n_occurrences", ascending=False)
    gdf.to_csv(OUT / "gbif_regional_genera.csv", index=False, encoding="utf-8")

# Class list
    keep = gdf[gdf["is_trained_class"]].copy()
    cls = (keep.groupby("final_label")
               .agg(label_rank=("label_rank", "first"),
                    family=("family", "first"),
                    n_gbif_genera=("gbif_genus", "nunique"),
                    gbif_genera=("gbif_genus", lambda s: "|".join(sorted(set(s)))),
                    n_occurrences=("n_occurrences", "sum"))
               .reset_index()
               .sort_values("n_occurrences", ascending=False))
    sup = trained.set_index("label")["support"]
    cls["test_support"] = cls["final_label"].map(sup).fillna(0).astype(int)
    cls["est_total_images"] = (cls["test_support"] / 0.15).round().astype(int)
    cls.to_csv(OUT / "alaska_class_list.csv", index=False, encoding="utf-8")

# Gap report
    gap = gdf[~gdf["is_trained_class"]].copy()
# Reasons for being missing
    gap["reason"] = gap.apply(
        lambda r: "genus not in taxonomy hierarchy" if not r["in_hierarchy"]
        else ("pooled label not trained" if r["label_rank"] == "family"
              else "below image threshold"), axis=1)
    gap = gap[["gbif_genus", "gbif_family", "n_occurrences", "final_label",
               "label_rank", "raw_images_for_genus", "reason"]]
    gap.to_csv(OUT / "alaska_gap_report.csv", index=False, encoding="utf-8")

    print(f"\nclass list: {len(cls)} classes, ~{cls['est_total_images'].sum():,} images")
    print(f"gap report: {len(gap)} regional genera the corpus cannot classify")
    print("\n  top 12 regional genera with no usable class:")
    for r in gap.head(12).itertuples():
        print(f"     {str(r.gbif_genus):18s} occ={r.n_occurrences:6,d}  "
              f"raw_imgs={r.raw_images_for_genus:3d}  {r.reason}")
    print(f"\n  written -> {OUT}")
    print("  NEXT: run the same predicate through the GBIF download form, "
          "paste the DOI into gbif_query_record.json, and cite it in Methods.")


if __name__ == "__main__":
    main()
