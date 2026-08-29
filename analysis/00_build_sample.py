"""Build the stratified panel sample.

Stratifies on severity bucket x guardrail condition x generation (5 x 3 x 3 = 45 cells).

Allocation is PROPORTIONAL WITH A FLOOR rather than balanced. Proportional keeps corpus-level
quantities (the R^2 decomposition, mean agreement) near-unbiased; the floor guarantees the rare
severity cells carry enough rows to support the by-severity analysis, which is where the U-shaped
detection curve lives. Every sampled row carries its inclusion probability so corpus-level
estimates can be reweighted back.

This exists because the paper's existing 73-response human-validation subset was NOT stratified,
and is badly unrepresentative as a result: condition mix Protocol 44 / Control 23 / Simple 6
against a balanced corpus, and sycophancy sd 0.561 against 0.985 corpus-wide. That skew is a live
alternative explanation for several of its findings. Do not repeat it.

Writes:
    sample/panel_sample.csv      the sampled rows with strata and weights
    sample/panel_sample.sha256   hash of the above, so the run is provably against this sample
    sample/allocation.md         the realised allocation table
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = Path(__file__).resolve().parent / "sample"
DATA = ROOT / "results" / "master_results.csv"

TARGET_N = 1200
FLOOR_PER_CELL = 15
SEED = 20260812


def severity_bucket(score: float) -> str:
    if score == 1.0:
        return "1_clean"
    if score < 2.0:
        return "2_borderline"
    if score < 3.0:
        return "3_mild"
    if score < 4.0:
        return "4_moderate"
    return "5_severe"


def generation(model: str) -> str:
    m = str(model)
    if "2.0" in m or "2_0" in m:
        return "Gen2.0"
    if "2.5" in m or "2_5" in m:
        return "Gen2.5"
    return "Gen3.0"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA)
    n_raw = len(df)

    # The corpus has 10 Response_IDs duplicated across 39 rows. Sample over unique responses so a
    # duplicated row cannot enter the panel twice under the same identifier.
    df = df.drop_duplicates(subset="Response_ID", keep="first").reset_index(drop=True)
    print(f"rows {n_raw} -> {len(df)} unique Response_IDs")

    df["severity"] = df["Sycophancy_Mean"].map(severity_bucket)
    df["generation"] = df["model"].map(generation)
    df["stratum"] = df["severity"] + " | " + df["condition"] + " | " + df["generation"]

    sizes = df.groupby("stratum").size()
    n_cells = len(sizes)
    print(f"{n_cells} non-empty strata of a possible 45")

    # Proportional allocation, floored, then trimmed largest-first back to TARGET_N.
    alloc = np.maximum(
        (sizes / sizes.sum() * TARGET_N).round().astype(int),
        np.minimum(FLOOR_PER_CELL, sizes),
    )
    while alloc.sum() > TARGET_N:
        # trim from the cell with the largest surplus over its floor
        surplus = alloc - np.minimum(FLOOR_PER_CELL, sizes)
        if surplus.max() <= 0:
            break
        alloc[surplus.idxmax()] -= 1
    while alloc.sum() < TARGET_N:
        headroom = sizes - alloc
        if headroom.max() <= 0:
            break
        alloc[headroom.idxmax()] += 1

    rng = np.random.default_rng(SEED)
    picked = []
    for stratum, group in df.groupby("stratum"):
        k = int(alloc[stratum])
        if k <= 0:
            continue
        idx = rng.choice(group.index.values, size=k, replace=False)
        sel = df.loc[idx].copy()
        sel["stratum_size"] = len(group)
        sel["stratum_drawn"] = k
        sel["inclusion_prob"] = k / len(group)
        sel["weight"] = len(group) / k
        picked.append(sel)

    sample = pd.concat(picked).sort_values("Response_ID").reset_index(drop=True)

    cols = [
        "Response_ID", "model", "generation", "condition", "Prompt_ID", "Category",
        "Prompt_Text", "Assistant_Response",
        "Verdict", "Sycophancy_Mean", "Sycophancy_StdDev",
        "Truthfulness_Mean", "Truthfulness_StdDev", "Refusal_Mean", "Refusal_StdDev",
        "severity", "stratum", "stratum_size", "stratum_drawn", "inclusion_prob", "weight",
    ]
    sample = sample[cols]

    path = OUT / "panel_sample.csv"
    sample.to_csv(path, index=False, encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    (OUT / "panel_sample.sha256").write_text(digest + "\n", encoding="utf-8")

    # Representativeness check against the corpus, weighted and unweighted.
    w = sample["weight"]
    lines = [
        "# Panel sample allocation",
        "",
        f"seed {SEED}, target n {TARGET_N}, floor {FLOOR_PER_CELL} per cell",
        f"realised n {len(sample)} across {sample.stratum.nunique()} strata",
        f"sha256 {digest}",
        "",
        "## Representativeness",
        "",
        "| quantity | corpus | sample (unweighted) | sample (weighted) |",
        "|---|---|---|---|",
    ]
    for col in ["Sycophancy_Mean", "Refusal_Mean", "Truthfulness_Mean"]:
        lines.append(
            f"| mean {col} | {df[col].mean():.4f} | {sample[col].mean():.4f} "
            f"| {np.average(sample[col], weights=w):.4f} |"
        )
    lines.append(
        f"| sd Sycophancy_Mean | {df.Sycophancy_Mean.std():.4f} "
        f"| {sample.Sycophancy_Mean.std():.4f} | (n/a) |"
    )
    for cond in ["Control", "Protocol", "Simple"]:
        lines.append(
            f"| share {cond} | {(df.condition == cond).mean():.4f} "
            f"| {(sample.condition == cond).mean():.4f} "
            f"| {np.average((sample.condition == cond).astype(float), weights=w):.4f} |"
        )
    lines += ["", "## Allocation by stratum", "", "| stratum | population | drawn | weight |", "|---|---|---|---|"]
    for stratum in sorted(sizes.index):
        k = int(alloc[stratum])
        if k:
            lines.append(f"| {stratum} | {sizes[stratum]} | {k} | {sizes[stratum]/k:.2f} |")

    (OUT / "allocation.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"wrote {path}  sha256 {digest[:16]}...")
    print(f"n={len(sample)}  strata={sample.stratum.nunique()}")
    print(f"unweighted mean syc {sample.Sycophancy_Mean.mean():.4f} "
          f"| weighted {np.average(sample.Sycophancy_Mean, weights=w):.4f} "
          f"| corpus {df.Sycophancy_Mean.mean():.4f}")


if __name__ == "__main__":
    main()
