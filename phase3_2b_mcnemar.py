#!/usr/bin/env python3
# What this does: tests whether two models are statistically different
import os
import sys
import json
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

PROJECT_ROOT = Path(os.environ.get("POLLEN_ROOT", r"D:\Pollen\pollen_project"))
EVAL_DIR = PROJECT_ROOT / "outputs" / "eval"

KEY = "processed_path"
CORRECT = "correct"


def load_correctness(path: Path, name: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    for col in (KEY, CORRECT):
        if col not in df.columns:
            raise SystemExit(f"{path.name} has no '{col}' column (found: {list(df.columns)})")
    out = df[[KEY, CORRECT]].copy()
    out[KEY] = out[KEY].astype(str)
    out[CORRECT] = out[CORRECT].map(_as_bool)
    if out[CORRECT].isna().any():
        raise SystemExit(f"{path.name}: '{CORRECT}' column is not true/false")
    if not out[KEY].is_unique:
        raise SystemExit(f"{path.name}: '{KEY}' repeats, so the two files cannot be matched up 1-to-1")
    return out.rename(columns={CORRECT: name})


def _as_bool(v):
# Turn CSV values to bool
    if isinstance(v, (bool, np.bool_)):
        return bool(v)
    if isinstance(v, (int, float, np.integer, np.floating)):
        return bool(v) if v in (0, 1) else None
    s = str(v).strip().lower()
    if s in ("true", "1", "yes", "t"):
        return True
    if s in ("false", "0", "no", "f"):
        return False
    return None


def pair_up(a: pd.DataFrame, b: pd.DataFrame, a_name: str, b_name: str) -> pd.DataFrame:
# Match the two files on image path, keeping only grains that appear in both
    m = a.merge(b, on=KEY, how="inner")
    if len(m) == 0:
        raise SystemExit("no grains in common - are these predictions on the same test split?")
    return m


def contingency(m: pd.DataFrame, a_name: str, b_name: str) -> dict:
# The four possible outcomes summed
    ca, cb = m[a_name].to_numpy(), m[b_name].to_numpy()
    return {
        "n": int(len(m)),
        "both_correct": int((ca & cb).sum()),
        "both_incorrect": int((~ca & ~cb).sum()),
        "only_b_correct": int((~ca & cb).sum()),
        "only_a_correct": int((ca & ~cb).sum()),
    }


def mcnemar(only_a: int, only_b: int) -> dict:
    n_disc = only_a + only_b
    if n_disc == 0:
        return {"chi2": float("nan"), "p_chi2": float("nan"),
                "p_exact": float("nan"), "n_discordant": 0}
    chi2 = (abs(only_b - only_a) - 1) ** 2 / n_disc
    return {
        "chi2": float(chi2),
        "p_chi2": float(stats.chi2.sf(chi2, 1)),
        "p_exact": float(stats.binomtest(only_b, n_disc, 0.5).pvalue),
        "n_discordant": int(n_disc),
    }


def report_p(p: float) -> str:
    if np.isnan(p):
        return "n/a"
    return "< 0.001" if p < 0.001 else f"= {p:.3f}"


def main():
# Count the disagreements, run the test, and save the result
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="first predictions CSV")
    ap.add_argument("--b", required=True, help="second predictions CSV")
    ap.add_argument("--a-name", default="A", help="name of the first model, for the report")
    ap.add_argument("--b-name", default="B", help="name of the second model, for the report")
    ap.add_argument("--out", default=None, help="where to write the JSON (default outputs/eval/)")
    args = ap.parse_args()

    a_name, b_name = args.a_name, args.b_name
    if a_name == b_name:
        raise SystemExit("--a-name and --b-name must differ")

    a_path, b_path = Path(args.a), Path(args.b)
    for p in (a_path, b_path):
        if not p.exists():
            raise SystemExit(f"predictions not found: {p}")

    a = load_correctness(a_path, a_name)
    b = load_correctness(b_path, b_name)
    m = pair_up(a, b, a_name, b_name)

    dropped = max(len(a), len(b)) - len(m)
    if dropped:
        print(f"warning: {dropped} grain(s) were not in both files and were left out")

    t = contingency(m, a_name, b_name)
    res = mcnemar(t["only_a_correct"], t["only_b_correct"])

    acc_a = float(m[a_name].mean())
    acc_b = float(m[b_name].mean())

    print(f"\npaired on {KEY}: n = {t['n']:,}")
    print(f"  {a_name:<16s} accuracy {acc_a:.4f}")
    print(f"  {b_name:<16s} accuracy {acc_b:.4f}")
    print("\n  agreement:")
    print(f"    both correct            {t['both_correct']:,}")
    print(f"    both incorrect          {t['both_incorrect']:,}")
    print("\n  disagreement (what the test uses):")
    print(f"    only {b_name} correct    {t['only_b_correct']:,}")
    print(f"    only {a_name} correct    {t['only_a_correct']:,}")
    print(f"\n  McNemar chi2(1) = {res['chi2']:.1f}, p {report_p(res['p_chi2'])} "
          f"(exact p {report_p(res['p_exact'])})")
    better = b_name if t["only_b_correct"] > t["only_a_correct"] else a_name
    if res["p_chi2"] < 0.05:
        print(f"  -> the difference is unlikely to be chance; {better} is better")
    else:
        print("  -> no significant difference between the two models")

    out = Path(args.out) if args.out else EVAL_DIR / f"mcnemar_{_slug(a_name)}_vs_{_slug(b_name)}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"a_name": a_name, "b_name": b_name,
               "a_file": str(a_path), "b_file": str(b_path),
               "a_accuracy": acc_a, "b_accuracy": acc_b, **t, **res}
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


def _slug(s: str) -> str:
# Make model name OK to use as filename
    return "".join(ch if ch.isalnum() else "_" for ch in s).strip("_").lower()


if __name__ == "__main__":
    main()
