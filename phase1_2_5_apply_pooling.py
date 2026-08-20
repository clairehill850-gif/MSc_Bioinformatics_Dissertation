#!/usr/bin/env python3
# What this does: for UK/temperate plants, groups some taxa up to family or morphological type level.
from pathlib import Path
import shutil
import pandas as pd

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")
TAX = PROJECT_ROOT / "outputs" / "taxonomy"
HIER = TAX / "taxonomy_hierarchy.csv"
INV = PROJECT_ROOT / "outputs" / "inventory" / "master_inventory.csv"

# Normalise to GBIF nomlenclature
FAM_SYN = {"Leguminosae": "Fabaceae", "Compositae": "Asteraceae",
           "Cruciferae": "Brassicaceae", "Umbelliferae": "Apiaceae",
           "Gramineae": "Poaceae", "Chenopodiaceae": "Amaranthaceae"}

POOL_NAME = {
    "Poaceae": "Poaceae", "Cyperaceae": "Cyperaceae", "Juncaceae": "Juncaceae",
    "Cupressaceae": "Cupressaceae", "Ericaceae": "Ericaceae", "Rosaceae": "Rosaceae",
    "Fabaceae": "Fabaceae", "Brassicaceae": "Brassicaceae", "Apiaceae": "Apiaceae",
    "Ranunculaceae": "Ranunculaceae", "Asteraceae": "Asteraceae",
    "Amaranthaceae": "Chenopodiaceae-Amaranthaceae type",
}
MORPHOTYPE = {"Chenopodiaceae-Amaranthaceae type", "Cerealia-type"}

CEREALIA = {"Triticum", "Hordeum", "Secale", "Avena"}
KEEP_FLAGGED = {"Phragmites"}      

KEEP_AS_GENUS = {
    "Betula", "Alnus", "Corylus", "Carpinus", "Ostrya", "Fagus", "Quercus",
    "Castanea", "Salix", "Populus", "Ulmus", "Fraxinus", "Acer", "Tilia", "Ilex",
    "Pinus", "Picea", "Abies", "Larix", "Taxus", "Calluna", "Erica", "Prunus",
    "Malus", "Crataegus", "Trifolium", "Lotus", "Plantago", "Rumex", "Polygonum",
    "Urtica", "Artemisia", "Hedera", "Sambucus",
}

# Groupings
UK_GENERA = {
    "Poaceae": {"Agrostis", "Alopecurus", "Anthoxanthum", "Arrhenatherum", "Briza",
        "Bromus", "Calamagrostis", "Cynosurus", "Dactylis", "Deschampsia", "Elymus",
        "Festuca", "Glyceria", "Holcus", "Lolium", "Melica", "Milium", "Molinia",
        "Nardus", "Phalaris", "Phleum", "Poa", "Puccinellia", "Sesleria", "Trisetum",
        "Vulpia", "Catabrosa", "Danthonia", "Koeleria", "Helictotrichon", "Bromopsis"},
    "Cyperaceae": {"Carex", "Cyperus", "Scirpus", "Eleocharis", "Bolboschoenus",
        "Schoenoplectus", "Eriophorum", "Blysmus", "Rhynchospora", "Schoenus",
        "Cladium", "Isolepis", "Trichophorum"},
    "Juncaceae": {"Juncus", "Luzula"},
    "Cupressaceae": {"Juniperus", "Cupressus", "Chamaecyparis", "Thuja"},
    "Ericaceae": {"Vaccinium", "Empetrum", "Andromeda", "Arctostaphylos", "Arbutus",
        "Rhododendron", "Gaultheria", "Phyllodoce", "Loiseleuria", "Daboecia", "Pieris"},
    "Rosaceae": {"Rosa", "Rubus", "Sorbus", "Pyrus", "Filipendula", "Potentilla",
        "Geum", "Fragaria", "Agrimonia", "Sanguisorba", "Alchemilla", "Aphanes",
        "Aruncus", "Cotoneaster", "Mespilus", "Spiraea", "Dryas", "Comarum", "Amelanchier"},
    "Fabaceae": {"Vicia", "Lathyrus", "Medicago", "Melilotus", "Ononis", "Anthyllis",
        "Astragalus", "Hippocrepis", "Ornithopus", "Ulex", "Genista", "Cytisus",
        "Lupinus", "Robinia", "Laburnum", "Coronilla"},
    "Brassicaceae": {"Brassica", "Sinapis", "Capsella", "Cardamine", "Arabidopsis",
        "Sisymbrium", "Raphanus", "Alliaria", "Lepidium", "Thlaspi", "Draba",
        "Erophila", "Barbarea", "Rorippa", "Nasturtium", "Hesperis", "Erysimum",
        "Cakile", "Crambe", "Cochlearia", "Arabis", "Lunaria", "Diplotaxis"},
    "Apiaceae": {"Daucus", "Heracleum", "Anthriscus", "Conium", "Pastinaca", "Apium",
        "Aegopodium", "Angelica", "Chaerophyllum", "Torilis", "Sanicula", "Smyrnium",
        "Foeniculum", "Pimpinella", "Oenanthe", "Cicuta", "Carum", "Conopodium",
        "Myrrhis", "Scandix", "Sium", "Berula", "Crithmum", "Bupleurum", "Aethusa",
        "Silaum", "Meum", "Petroselinum", "Eryngium", "Hydrocotyle"},
    "Ranunculaceae": {"Ranunculus", "Caltha", "Anemone", "Clematis", "Aquilegia",
        "Helleborus", "Aconitum", "Actaea", "Thalictrum", "Trollius", "Ficaria",
        "Myosurus", "Adonis", "Consolida", "Delphinium", "Hepatica", "Pulsatilla", "Eranthis"},
    "Asteraceae": {"Aster", "Bellis", "Solidago", "Senecio", "Jacobaea", "Tussilago",
        "Petasites", "Centaurea", "Carduus", "Cirsium", "Arctium", "Onopordum",
        "Taraxacum", "Sonchus", "Hieracium", "Pilosella", "Crepis", "Lapsana",
        "Leontodon", "Picris", "Hypochaeris", "Cichorium", "Lactuca", "Mycelis",
        "Tragopogon", "Achillea", "Tripleurospermum", "Matricaria", "Anthemis",
        "Leucanthemum", "Tanacetum", "Pulicaria", "Inula", "Eupatorium", "Bidens",
        "Erigeron", "Filago", "Gnaphalium", "Antennaria", "Doronicum", "Carlina",
        "Serratula", "Saussurea", "Scorzonera", "Cota", "Galinsoga", "Conyza"},
    "Amaranthaceae": {"Chenopodium", "Atriplex", "Amaranthus", "Salsola", "Beta",
        "Suaeda", "Salicornia", "Spinacia", "Blitum", "Bassia", "Halimione",
        "Chenopodiastrum", "Oxybasis", "Lipandra", "Sarcocornia"},
}


def pooled_label(genus, family):
# Return new (possibly grouped) label and rank
    fam = FAM_SYN.get(family, family)
    if genus in KEEP_AS_GENUS or genus in KEEP_FLAGGED:
        return genus, "genus"
    if genus in CEREALIA:
        return "Cerealia-type", "morphotype"
    if fam in POOL_NAME and genus in UK_GENERA.get(fam, set()):
        name = POOL_NAME[fam]
        return name, ("morphotype" if name in MORPHOTYPE else "family")
    return genus, "genus"


def main():
    if not HIER.exists():
        raise SystemExit(f"Missing {HIER}")
    backup = HIER.with_suffix(".prepool.csv")
    src = backup if backup.exists() else HIER
    hier = pd.read_csv(src, dtype=str, keep_default_na=False)
    print(f"pooling source: {src.name}; columns: {list(hier.columns)}")

    fcol = next((c for c in ("family", "accepted_family") if c in hier.columns), None)
    if fcol is None:
        raise SystemExit("No family column in hierarchy; cannot pool.")
    hier["base_genus"] = hier["final_label"].astype(str).str.strip()
    if not backup.exists():
        shutil.copy(HIER, backup)   


# Image counts per genus
    counts = {}
    if INV.exists():
        inv = pd.read_csv(INV, low_memory=False)
        inv["genus"] = inv["genus"].fillna("").astype(str).str.strip()
        o2b = dict(zip(hier["original_label"], hier["base_genus"]))
        inv["bg"] = inv["genus"].map(o2b)
        counts = inv["bg"].value_counts().to_dict()

# Do the grouping
    new_labels, new_ranks, changed = [], [], []
    for _, r in hier.iterrows():
        g, fam = r["base_genus"], r[fcol]
        old = r["final_label"]
        nl, rank = pooled_label(g, fam)
        new_labels.append(nl); new_ranks.append(rank)
        changed.append(nl != old)
    hier["final_label"] = new_labels
    if "label_rank" in hier.columns:
        hier["label_rank"] = new_ranks
    else:
        hier["label_rank"] = new_ranks
    hier.to_csv(HIER, index=False, encoding="utf-8")

# Write the name-map python file
    label_map = dict(zip(hier["original_label"], hier["final_label"]))
    pyf = TAX / "taxonomy_hierarchy.py"
    pyf.write_text("# auto-generated by phase1.2.5_apply_pooling.py\n"
                   "LABEL_MAP = " + repr(label_map) + "\n", encoding="utf-8")

# Logs
    pooled_rows, family_view = [], []
    seen = set()
    for _, r in hier.iterrows():
        g, fam = r["base_genus"], FAM_SYN.get(r[fcol], r[fcol])
        if g in seen:
            continue
        seen.add(g)
        nl, rank = pooled_label(g, r[fcol])
        n = counts.get(g, 0)
        if nl != g:                       
            pooled_rows.append({"genus": g, "family": fam, "pooled_into": nl,
                                "rank": rank, "n_images": n})
        if fam in POOL_NAME and g != fam:   
            family_view.append({"family": fam, "genus": g, "pooled": nl != g,
                                "result": nl, "n_images": n,
                                "reason": ("kept_as_genus" if (g in KEEP_AS_GENUS or g in KEEP_FLAGGED)
                                           else ("pooled" if nl != g else "not_in_UK_list_left_at_genus"))})
    pl = pd.DataFrame(pooled_rows).sort_values(["pooled_into", "n_images"],
                                               ascending=[True, False]) if pooled_rows else pd.DataFrame()
    fv = pd.DataFrame(family_view).sort_values(["family", "pooled", "n_images"],
                                               ascending=[True, False, False]) if family_view else pd.DataFrame()
    pl.to_csv(TAX / "pooling_log.csv", index=False, encoding="utf-8")
    fv.to_csv(TAX / "pooling_family_view.csv", index=False, encoding="utf-8")

# Report
    print(f"\ngenera pooled: {len(pl)}")
    if len(pl):
        print("\nresulting pooled classes (by image count):")
        agg = pl.groupby("pooled_into").agg(genera=("genus", "count"),
                                            images=("n_images", "sum")).sort_values("images", ascending=False)
        for cls, row in agg.iterrows():
            print(f"   {cls}: {int(row['images']):,} images from {int(row['genera'])} genera")
# Sanity check
    if len(fv):
        left = fv[(~fv.pooled) & (fv.reason == "not_in_UK_list_left_at_genus")]
        print(f"\nin pool families but left at genus (tropical/unlisted): {len(left)} genera "
              f"({int(left.n_images.sum()):,} images)")
        print("   -> review pooling_family_view.csv: any UK genus wrongly here should be added to its UK list")
    print(f"\n  written: taxonomy_hierarchy.csv (backup .prepool.csv), pooling_log.csv, pooling_family_view.csv")
    print("  Next: re-run 1.3.1 -> 1.2.6 -> 1.3.3 -> 1.3.4 (all fast; hashes cached).")


if __name__ == "__main__":
    main()
