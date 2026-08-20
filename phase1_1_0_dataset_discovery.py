#!/usr/bin/env python3
# What this does: takes a look at each folder of images and makes a report on the folder layout, image file types, how deeply nested things are etc.

from pathlib import Path
from collections import Counter
import csv

PROJECT_ROOT = Path(r"D:\Pollen\pollen_project")

RAW_DIR = PROJECT_ROOT / "data" / "raw"
OUT_DIR = PROJECT_ROOT / "outputs" / "discovery"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif", ".webp"}

# Datasets excluded
DATASET_EXCLUDE = {
    "Tananaki": "multi-grain field images — incompatible with single-grain classifier",
}

EXCLUDED_SUFFIX = "_excluded"
# Example file paths to record per dataset
SAMPLE_PATHS_PER_DATASET = 12          
MAX_SAMPLES_PER_DEPTH    = 4           


# Helper functions
def is_image(p: Path) -> bool:
# True if file is an image (based on extension)
    return p.suffix.lower() in IMAGE_EXTS


def classify_dataset(name: str):
# Should dataset be included
    if name in DATASET_EXCLUDE:
        return False, DATASET_EXCLUDE[name]
    if name.lower().endswith(EXCLUDED_SUFFIX):
        return False, "manual-cleaning audit sibling (*_excluded)"
    return True, ""


def scan_dataset(ds_root: Path):
# Look through dataset folder and get basic info
    depth_counter = Counter()          
    ext_counter = Counter()            
    samples_by_depth = {}             
    total = 0

    for p in ds_root.rglob("*"):
        if not p.is_file() or not is_image(p):
            continue
        total += 1
        rel = p.relative_to(ds_root)

        depth = len(rel.parts)        
        depth_counter[depth] += 1
        ext_counter[p.suffix.lower()] += 1
        bucket = samples_by_depth.setdefault(depth, [])
        if len(bucket) < MAX_SAMPLES_PER_DEPTH:
            bucket.append(str(rel))

    return {
        "total": total,
        "depth_counter": depth_counter,
        "ext_counter": ext_counter,
        "samples_by_depth": samples_by_depth,
    }


def label_location_hint(depth_counter: Counter) -> str:
# Guess which part of the path holds the plant name.
    if not depth_counter:
        return "no images found"
    modal_depth = depth_counter.most_common(1)[0][0]
    if modal_depth == 1:
        return "FLAT — taxon likely encoded in the FILENAME"
    if modal_depth == 2:
        return "taxon likely the IMMEDIATE PARENT folder"
    return (f"DEEP (modal depth {modal_depth}) — taxon likely one of the "
            f"parent folders; confirm which level holds the genus")


# Main part of the script
def main():
# Go through each dataset folder, describe it, and save the report.
    if not RAW_DIR.exists():
        raise SystemExit(
            f"RAW_DIR does not exist:\n  {RAW_DIR}\n"
            f"Edit PROJECT_ROOT at the top of this script."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    datasets = sorted(
        [d for d in RAW_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name.lower(),
    )

    summary_rows = []
    sample_rows = []
    report_lines = []

    report_lines.append("# Phase 1.1.0 — Dataset Discovery\n")
    report_lines.append(f"Raw directory: `{RAW_DIR}`\n")
    report_lines.append(
        "Read-only. Excluded datasets are listed but not "
        "scanned in depth.\n"
    )

    grand_total = 0

    for ds in datasets:
        included, reason = classify_dataset(ds.name)

        if not included:
            report_lines.append(f"\n## {ds.name}  —  EXCLUDED")
            report_lines.append(f"- Reason: {reason}")
            summary_rows.append({
                "dataset": ds.name,
                "included": "No",
                "exclusion_reason": reason,
                "total_images": "",
                "modal_depth": "",
                "extensions": "",
                "label_location_hint": "",
            })
            continue

        stats = scan_dataset(ds)
        grand_total += stats["total"]

        depth_c = stats["depth_counter"]
        modal_depth = depth_c.most_common(1)[0][0] if depth_c else ""
        ext_str = ", ".join(
            f"{ext}:{n}" for ext, n in stats["ext_counter"].most_common()
        )
        hint = label_location_hint(depth_c)

# Build report
        report_lines.append(f"\n## {ds.name}")
        report_lines.append(f"- Images: **{stats['total']}**")
        if depth_c:
            depth_str = ", ".join(
                f"depth {d}: {n}" for d, n in sorted(depth_c.items())
            )
            report_lines.append(f"- Nesting depth (image count): {depth_str}")
        report_lines.append(f"- Extensions: {ext_str or '(none)'}")
        report_lines.append(f"- Label-location hint: {hint}")
        report_lines.append(f"- Sample paths:")
        shown = 0
        for d in sorted(stats["samples_by_depth"]):
            for rel in stats["samples_by_depth"][d]:
                report_lines.append(f"    - `{rel}`")
                sample_rows.append({"dataset": ds.name, "depth": d,
                                    "relative_path": rel})
                shown += 1
                if shown >= SAMPLE_PATHS_PER_DATASET:
                    break
            if shown >= SAMPLE_PATHS_PER_DATASET:
                break

        summary_rows.append({
            "dataset": ds.name,
            "included": "Yes",
            "exclusion_reason": "",
            "total_images": stats["total"],
            "modal_depth": modal_depth,
            "extensions": ext_str,
            "label_location_hint": hint,
        })

    report_lines.append(f"\n---\n\n**Grand total (included datasets): "
                        f"{grand_total} images**\n")

# Save report and data files
    (OUT_DIR / "discovery_report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )

    with (OUT_DIR / "discovery_summary.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=[
            "dataset", "included", "exclusion_reason", "total_images",
            "modal_depth", "extensions", "label_location_hint",
        ])
        w.writeheader()
        w.writerows(summary_rows)

    with (OUT_DIR / "sample_paths.csv").open(
            "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["dataset", "depth", "relative_path"])
        w.writeheader()
        w.writerows(sample_rows)

# Summary
    print("Discovery complete.")
    print(f"  Datasets found     : {len(datasets)}")
    n_excl = sum(1 for r in summary_rows if r["included"] == "No")
    print(f"  Included           : {len(summary_rows) - n_excl}")
    print(f"  Excluded           : {n_excl}")
    print(f"  Total images (incl): {grand_total}")
    print(f"  Outputs written to : {OUT_DIR}")
    print()
    print("Excluded datasets:")
    for r in summary_rows:
        if r["included"] == "No":
            print(f"  - {r['dataset']}: {r['exclusion_reason']}")


if __name__ == "__main__":
    main()
