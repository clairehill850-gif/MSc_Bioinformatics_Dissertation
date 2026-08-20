#!/usr/bin/env python3
# What this does: gives each surviving (and rescue-band) class a starting forensic-importance tier and an identification level (family / genus / species).
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
CUR = PROJECT_ROOT / "outputs" / "curation"
TAX = PROJECT_ROOT / "outputs" / "taxonomy"
TAX.mkdir(parents=True, exist_ok=True)

# HIGH-importance taxa
HIGH_ABUNDANT = {"Pinus", "Betula", "Corylus", "Quercus", "Carpinus", "Fraxinus", "Poaceae"}
HIGH_HABITAT = {"Alnus", "Salix", "Populus", "Carex", "Typha", "Cyperus", "Scirpus",
                "Eleocharis", "Phragmites", "Calluna", "Erica", "Sphagnum",
                "Cyperaceae", "Juncaceae"}
HIGH_OPEN_GROUND = {"Plantago", "Rumex", "Artemisia", "Urtica", "Taraxacum",
                    "Chenopodium", "Atriplex", "Brassica", "Papaver", "Trifolium",
                    "Brassicaceae", "Cerealia-type", "Chenopodiaceae-Amaranthaceae type"}
HIGH_UK = HIGH_ABUNDANT | HIGH_HABITAT | HIGH_OPEN_GROUND

MEDIUM_UK = {"Acer", "Prunus", "Tilia", "Ulmus", "Fagus", "Juglans", "Castanea",
             "Sorbus", "Crataegus", "Malus", "Pyrus", "Sambucus", "Rubus", "Rosa",
             "Sinapis", "Centaurea", "Carduus", "Cirsium", "Hieracium", "Crepis",
             "Sonchus", "Tripleurospermum", "Polygonum", "Persicaria", "Lotus",
             "Vicia", "Lathyrus", "Medicago", "Achillea", "Taxus", "Abies", "Picea",
             "Ericaceae", "Rosaceae", "Fabaceae", "Apiaceae", "Ranunculaceae",
             "Asteraceae", "Cupressaceae"}

GRASS_GENERA = {"Anthoxanthum", "Dactylis", "Lolium", "Phleum", "Festuca", "Poa",
                "Bromus", "Holcus", "Agrostis", "Avena", "Hordeum", "Triticum", "Secale"}
FAMILY_ONLY = {"Poaceae", "Cyperaceae", "Amaranthaceae"}
SPECIES_RESOLVABLE = {"Cannabis"}


def seed_relevance(label, rank):
# Assign a forensic-importance tier
    if label in HIGH_ABUNDANT:
        return "High", "abundant, distinctive UK producer; high transfer probability + easy ID", "rule"
    if label in HIGH_HABITAT:
        return "High", "UK habitat/seasonal indicator (wetland/heath/riparian/bog)", "rule"
    if label in HIGH_OPEN_GROUND:
        return "High", "UK grassland/arable/disturbed-ground indicator", "rule"
    if label in MEDIUM_UK:
        return "Medium", "secondary UK taxon; aids hedgerow/woodland/pasture reconstruction", "rule"
    return "Low", "presumed tropical/exotic or poor producer - low UK forensic relevance; promote if UK-relevant", "default"


def seed_idres(label, rank):
# Assign an identification level
    if rank in ("family", "morphotype"):
        return "Family", "pooled to family/morphotype; LM resolves to this level only", "rule"
    if label in SPECIES_RESOLVABLE:
        return "Species", "morphologically distinctive; species-level possible under optimal LM", "rule"
    if label in GRASS_GENERA or label in FAMILY_ONLY:
        return "Family", "grass/cryptic type; LM reliably resolves to family only", "rule"
    return "Genus", "default - genus-level identification assumed under LM", "default"


def load_classes():
# Load for annotation
    frames = []
    for name, status in [("survivors.csv", "keep"), ("rescue_candidates.csv", "rescue_candidate")]:
        p = CUR / name
        if p.exists():
            df = pd.read_csv(p)
            if "status" not in df.columns:
                df["status"] = status
            frames.append(df)
    if not frames:
        raise SystemExit(f"No survivors.csv / rescue_candidates.csv in {CUR}. Run phase1.3.1 first.")
    out = pd.concat(frames, ignore_index=True)
    if "label_rank" not in out.columns:
        out["label_rank"] = "genus"
    return out


def main():
# Apply the tier and identification, save both annotation tables
    df = load_classes().sort_values(["status", "n_images"], ascending=[True, False])

    rel = df.copy()
    rel[["forensic_relevance_tier", "relevance_rationale", "seed_basis"]] = \
        rel.apply(lambda r: pd.Series(seed_relevance(r["final_label"], r.get("label_rank", "genus"))), axis=1)
    rel = rel[["final_label", "label_rank", "status", "n_images", "n_datasets",
               "forensic_relevance_tier", "relevance_rationale", "seed_basis"]]
    rel.to_csv(TAX / "forensic_relevance_tiers.csv", index=False, encoding="utf-8")

    idr = df.copy()
    idr[["id_resolution", "id_resolution_note", "seed_basis"]] = \
        idr.apply(lambda r: pd.Series(seed_idres(r["final_label"], r.get("label_rank", "genus"))), axis=1)
    idr = idr[["final_label", "label_rank", "status", "n_images",
               "id_resolution", "id_resolution_note", "seed_basis"]]
    idr.to_csv(TAX / "id_resolution.csv", index=False, encoding="utf-8")

    present = set(df["final_label"])
    print(f"classes annotated: {len(df):,}")
    print("\nforensic_relevance_tiers.csv - tier seeds:")
    for t in ("High", "Medium", "Low"):
        print(f"   {t}: {int((rel.forensic_relevance_tier == t).sum())}")
    print(f"   (Low = presumed-tropical default: {int((rel.seed_basis=='default').sum())} "
          f"-> scan for UK taxa to promote)")
    print("\nid_resolution.csv - resolution seeds:")
    for t, n in idr["id_resolution"].value_counts().items():
        print(f"   {t}: {n}")
    print(f"\nwritten: {TAX/'forensic_relevance_tiers.csv'}")
    print(f"written: {TAX/'id_resolution.csv'}")
    missing = sorted((HIGH_UK | MEDIUM_UK) - present)
    if missing:
        print(f"\n  note: {len(missing)} UK-spec taxa are NOT among your kept classes "
              f"(absent or below threshold):")
        print("   " + ", ".join(missing))


if __name__ == "__main__":
    main()
