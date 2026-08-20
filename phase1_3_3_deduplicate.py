#!/usr/bin/env python3
# What this does: finds near-duplicate images using visual pHash and dHash.
from pathlib import Path
import json
import time
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image
import imagehash

Image.MAX_IMAGE_PIXELS = None

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
RAW_DIR = PROJECT_ROOT / "data" / "raw"
INV = PROJECT_ROOT / "outputs" / "inventory" / "master_inventory.csv"
INCLUSION = PROJECT_ROOT / "outputs" / "curation" / "class_inclusion_log.csv"
HIER = PROJECT_ROOT / "outputs" / "taxonomy" / "taxonomy_hierarchy.csv"
OUT = PROJECT_ROOT / "outputs" / "curation"
CACHE = OUT / "image_hashes_v2.json"

HASH_SIZE = 8          
NEAR_THRESHOLD = 3        
REPORT_MAX = 12              
SAVE_EVERY = 1000

_POP = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def to_u64(hexstr):
    return np.uint64(int(hexstr, 16))


def hamming_matrix(ints):
# Table of how different every pair is
    a = np.asarray(ints, dtype=np.uint64)
    xor = np.bitwise_xor(a[:, None], a[None, :])
    b = np.ascontiguousarray(xor).view(np.uint8).reshape(len(a), len(a), 8)
    return _POP[b].sum(axis=2).astype(np.uint16)


class UnionFind:
# Grouping helper
    def __init__(self): self.p = {}
    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x
    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb: self.p[ra] = rb


def main():
# Analyse the images
    for p in (INV, INCLUSION, HIER):
        if not p.exists():
            raise SystemExit(f"Missing {p}")
    OUT.mkdir(parents=True, exist_ok=True)

    inv = pd.read_csv(INV, low_memory=False)
    inv["genus"] = inv["genus"].fillna("").astype(str).str.strip()
    for c in ("width", "height"):
        inv[c] = pd.to_numeric(inv[c], errors="coerce").fillna(0).astype(int)
    hier = pd.read_csv(HIER, dtype=str, keep_default_na=False)
    to_final = dict(zip(hier["original_label"], hier["final_label"]))
    incl = pd.read_csv(INCLUSION)
    keep_labels = set(incl[incl.status.isin(["keep", "rescue_candidate"])]["final_label"])
    inv["final_label"] = inv["genus"].map(to_final).fillna("")
    work = inv[inv["final_label"].isin(keep_labels)].copy()
    print(f"images in keep+rescue classes: {len(work):,}")

# Analysis
    cache = {}
    if CACHE.exists():
        try: cache = json.loads(CACHE.read_text())
        except Exception: cache = {}
    todo = [r for r in work["rel_path"] if r not in cache]
    print(f"to hash: {len(todo):,} (cached {len(cache):,})  [pHash + dHash]")
    t0 = time.time()
    for i, rel in enumerate(todo, 1):
        try:
            with Image.open(RAW_DIR / rel) as im:
                g = im.convert("L")
                cache[rel] = {"p": str(imagehash.phash(g, hash_size=HASH_SIZE)),
                              "d": str(imagehash.dhash(g, hash_size=HASH_SIZE))}
        except Exception:
            cache[rel] = {"p": "", "d": ""}
        if i % SAVE_EVERY == 0:
            CACHE.write_text(json.dumps(cache))
            r = i / (time.time() - t0)
            print(f"  {i}/{len(todo)} ({r:.0f}/s, ETA {(len(todo)-i)/r/60:.1f} min)")
    CACHE.write_text(json.dumps(cache))

    work = work[work["rel_path"].map(lambda r: bool(cache.get(r, {}).get("p")))].copy()
    work["ph"] = work["rel_path"].map(lambda r: cache[r]["p"])
    work["dh"] = work["rel_path"].map(lambda r: cache[r]["d"])

    uf = UnionFind()
    edges = 0

# Exact: both images identical
    for _, grp in work.groupby(["ph", "dh"]):
        if len(grp) > 1:
            paths = grp["rel_path"].tolist()
            for p in paths[1:]:
                uf.union(paths[0], p); edges += 1

# Near: within a class, both images agree
    n_exact_pairs = edges
    n_near_pairs = 0
    near_hist = Counter()
    for label, grp in work.groupby("final_label"):
        if len(grp) < 2 or len(grp) > 6000:
            continue
        paths = grp["rel_path"].tolist()
        pm = hamming_matrix([to_u64(h) for h in grp["ph"]])
        dm = hamming_matrix([to_u64(h) for h in grp["dh"]])
        iu = np.triu_indices(len(paths), k=1)
        pp, dd = pm[iu], dm[iu]
        maxd = np.maximum(pp, dd)
        cand = (pp < REPORT_MAX) & (dd < REPORT_MAX) & ((pp + dd) > 0)
        for k in np.where(cand)[0]:
            md = int(maxd[k])
            near_hist[md] += 1
            if md < NEAR_THRESHOLD:
                i, j = iu[0][k], iu[1][k]
                uf.union(paths[i], paths[j]); n_near_pairs += 1

    work["cluster"] = work["rel_path"].map(uf.find)
    clustered = work[work.groupby("cluster")["cluster"].transform("size") > 1].copy()

    rows = []
    for cid, grp in clustered.groupby("cluster"):
        grp = grp.sort_values(["width", "height"], ascending=False)
        keeper = grp.iloc[0]["rel_path"]
        spans_ds = grp["dataset"].nunique() > 1
        spans_lbl = grp["final_label"].nunique() > 1
        for _, r in grp.iterrows():
            rows.append({"cluster_id": cid, "cluster_size": len(grp),
                         "rel_path": r["rel_path"], "dataset": r["dataset"],
                         "final_label": r["final_label"],
                         "width": r["width"], "height": r["height"],
                         "role": "keep" if r["rel_path"] == keeper else "remove_candidate",
                         "cross_dataset": spans_ds, "cross_label_CONFLICT": spans_lbl})
    ded = pd.DataFrame(rows)
    if len(ded):
        ded["cluster_id"] = ded.groupby("cluster_id", sort=False).ngroup() + 1
        ded = ded.sort_values(["cross_label_CONFLICT", "cross_dataset", "cluster_id"],
                              ascending=[False, False, True])
    ded.to_csv(OUT / "dedup_clusters.csv", index=False, encoding="utf-8")

    n_clusters = ded["cluster_id"].nunique() if len(ded) else 0
    n_remove = int((ded["role"] == "remove_candidate").sum()) if len(ded) else 0
    n_xds = ded[ded.cross_dataset]["cluster_id"].nunique() if len(ded) else 0
    n_conf = ded[ded.cross_label_CONFLICT]["cluster_id"].nunique() if len(ded) else 0
    pd.DataFrame([{"images_considered": len(work), "duplicate_clusters": n_clusters,
                   "remove_candidates": n_remove, "cross_dataset_clusters": n_xds,
                   "cross_label_conflict_clusters": n_conf}]).to_csv(OUT / "dedup_summary.csv", index=False)

    print(f"\nduplicate clusters       : {n_clusters:,}")
    print(f"  remove-candidate images: {n_remove:,}  ({n_remove/max(len(work),1)*100:.1f}% of kept)")
    print(f"  cross-dataset clusters : {n_xds:,}  (the train/test leak risk)")
    print(f"  cross-LABEL conflicts  : {n_conf:,}  (same grain, different genus - inspect)")
    if len(ded):
        sizes = ded.groupby("cluster_id").size()
        print(f"\n  cluster sizes: max {int(sizes.max())}, "
              f"clusters >10 imgs: {int((sizes>10).sum())} "
              f"(large clusters = possible chaining, inspect)")
    print(f"  duplicate pairs: {n_exact_pairs:,} exact + {n_near_pairs:,} near (<{NEAR_THRESHOLD})")
    print(f"  near-pair distance distribution (max of pHash/dHash dist):")
    for d in range(REPORT_MAX):
        if near_hist.get(d):
            mark = "  <- KEPT (clustered)" if d < NEAR_THRESHOLD else ""
            print(f"     dist {d}: {near_hist[d]:,}{mark}")
    print(f"\n  written: dedup_clusters.csv, dedup_summary.csv")


if __name__ == "__main__":
    main()
