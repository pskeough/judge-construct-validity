"""Analyse the judge panel.

Five questions, in the order they matter for the paper:

1. ANCHOR COLLAPSE. Does every judge collapse the 1-5 scale onto the 1/3/5 anchors the rubric
   defines? If so the "5-point Likert" is operationally 3-point and the paper must say so.

2. PER-JUDGE DECOMPOSITION. For each judge independently: how much of its own severity variance
   does its own refuse-or-comply verdict explain (R^2), and what is the ceiling for any function
   of its own refusal axis (eta^2)? Replicating the Gemini result inside other judges is what
   converts the construct gap from a within-instrument identity into a property of
   refusal-based measurement.

3. THE GAP BETWEEN JUDGES. Judges agree strongly on verdicts and weakly on severity. That is the
   same phenomenon appearing across instruments rather than inside one, and it does not depend on
   the algebraic identity at all.

4. VARIANCE DECOMPOSITION. Split score variance into within-judge (temperature sampling) and
   between-judge (model identity) components. The paper's best-of-3 conflates these.

5. AGREEMENT WITH THE BASE OF RECORD. Does the panel corroborate the original Gemini scores?

The original Gemini judge joins the panel for free: its per-vote scores reconstruct exactly from
(mean, sd) because each axis is three integers on 1-5.
"""
from __future__ import annotations

import itertools
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from judge_rubric import votes_to_verdict  # noqa: E402

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = HERE / "out" / "judge_logs.jsonl"
SAMPLE = HERE / "sample" / "panel_sample.csv"
REPORT = HERE / "out" / "PANEL_RESULTS.md"

AXES = ["sycophancy", "truthfulness", "refusal"]
ORIGINAL = "google/gemini-3-pro-preview (reconstructed)"


def invert_lut() -> dict[tuple[float, float], tuple[int, ...]]:
    lut: dict[tuple[float, float], list[tuple[int, ...]]] = defaultdict(list)
    for combo in itertools.combinations_with_replacement(range(1, 6), 3):
        m = sum(combo) / 3
        v = sum((x - m) ** 2 for x in combo) / 2
        lut[(round(m, 4), round(math.sqrt(v), 4))].append(combo)
    return {k: v[0] for k, v in lut.items() if len(v) == 1}


def r2(x: np.ndarray, y: np.ndarray) -> float:
    x = np.asarray(x, float)
    if x.std() == 0:
        return float("nan")
    X = np.column_stack([np.ones_like(x), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(1 - ((y - X @ beta) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def eta2(groups: np.ndarray, y: np.ndarray) -> float:
    frame = pd.DataFrame({"g": np.round(groups, 4), "y": y})
    means = frame.groupby("g").y.transform("mean")
    return float(1 - ((frame.y - means) ** 2).sum() / ((frame.y - frame.y.mean()) ** 2).sum())


def main() -> None:
    sample = pd.read_csv(SAMPLE).set_index("Response_ID")

    # votes[(response_id, judge)][axis] = [v1, v2, v3]
    votes: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with LOG.open(encoding="utf-8") as fh:
        for line in fh:
            rec = json.loads(line)
            if not rec.get("ok"):
                continue
            slot = votes[(rec["response_id"], rec["judge"])]
            for axis in AXES:
                slot[axis].append(int(rec[axis]))

    # Add the original judge, reconstructed from the corpus.
    lut = invert_lut()
    recon = skipped = 0
    for rid, row in sample.iterrows():
        triples = {}
        for axis, col in zip(AXES, ["Sycophancy", "Truthfulness", "Refusal"]):
            key = (round(float(row[f"{col}_Mean"]), 4),
                   round(0.0 if pd.isna(row[f"{col}_StdDev"]) else float(row[f"{col}_StdDev"]), 4))
            if key not in lut:
                break
            triples[axis] = list(lut[key])
        if len(triples) == 3:
            votes[(rid, ORIGINAL)] = triples
            recon += 1
        else:
            skipped += 1

    judges = sorted({j for _, j in votes})
    out: list[str] = ["# Judge panel results", ""]
    out.append(f"n = {len(sample)} stratified responses, {len(judges)} judges, 3 votes each.")
    out.append(f"Original judge reconstructed on {recon}/{len(sample)} responses "
               f"({skipped} not uniquely invertible).")
    out.append("")

    # ---- 1. anchor collapse -------------------------------------------------
    out += ["## 1. Anchor collapse", "",
            "The rubric defines anchors at 1, 3 and 5 only. Share of individual votes on each "
            "scale point:", "",
            "| judge | axis | 1 | 2 | 3 | 4 | 5 | share on 2 or 4 |", "|---|---|---|---|---|---|---|---|"]
    for judge in judges:
        for axis in AXES:
            allv = [v for (r, j), d in votes.items() if j == judge for v in d[axis]]
            if not allv:
                continue
            c = Counter(allv)
            tot = len(allv)
            pct = [100 * c[k] / tot for k in range(1, 6)]
            out.append(f"| {judge} | {axis} | " + " | ".join(f"{p:.2f}%" for p in pct)
                       + f" | **{pct[1] + pct[3]:.2f}%** |")
    out.append("")

    # ---- per-response aggregates -------------------------------------------
    rows = []
    for (rid, judge), d in votes.items():
        if len(d["refusal"]) == 0:
            continue
        verdict, _ = votes_to_verdict(d["refusal"])
        rows.append({
            "Response_ID": rid, "judge": judge,
            "syc": float(np.mean(d["sycophancy"])),
            "tru": float(np.mean(d["truthfulness"])),
            "ref": float(np.mean(d["refusal"])),
            "verdict": verdict,
            "n_votes": len(d["refusal"]),
            "weight": float(sample.loc[rid, "weight"]),
        })
    per = pd.DataFrame(rows)

    # ---- 2. per-judge decomposition ----------------------------------------
    out += ["## 2. Per-judge decomposition of the construct gap", "",
            "Each judge scored on its OWN axes. `R2` is its own binary verdict against its own "
            "severity; `eta2` is the ceiling for any function of its own refusal axis.", "",
            "| judge | n | R2 (own verdict) | eta2 (own refusal axis) | thresholding loss | construct residual |",
            "|---|---|---|---|---|---|"]
    for judge in judges:
        g = per[per.judge == judge]
        if len(g) < 50:
            continue
        flag = (g.verdict == "AGREED").astype(float).values
        y = g.syc.values
        a, b = r2(flag, y), eta2(g.ref.values, y)
        out.append(f"| {judge} | {len(g)} | {a:.4f} | {b:.4f} | {b - a:.4f} | {1 - b:.4f} |")
    out.append("")

    # ---- 3. the gap between judges -----------------------------------------
    out += ["## 3. The same gap appears BETWEEN judges", "",
            "Judges agree on the binary verdict far more than they agree on severity. This is the "
            "construct gap across instruments, and it does not rest on the within-judge "
            "algebraic identity.", "",
            "| judge A | judge B | n | verdict agreement | severity rho |", "|---|---|---|---|---|"]
    wide = per.pivot(index="Response_ID", columns="judge")
    pair_rows = []
    for a, b in itertools.combinations(judges, 2):
        try:
            va, vb = wide[("verdict", a)], wide[("verdict", b)]
            sa, sb = wide[("syc", a)], wide[("syc", b)]
        except KeyError:
            continue
        mask = va.notna() & vb.notna() & sa.notna() & sb.notna()
        if mask.sum() < 50:
            continue
        agree = float((va[mask] == vb[mask]).mean())
        rho = float(spearmanr(sa[mask], sb[mask]).statistic)
        pair_rows.append((agree, rho))
        out.append(f"| {a} | {b} | {int(mask.sum())} | {100 * agree:.2f}% | {rho:.3f} |")
    if pair_rows:
        ma = float(np.mean([p[0] for p in pair_rows]))
        mr = float(np.mean([p[1] for p in pair_rows]))
        out += ["", f"**Mean across all {len(pair_rows)} judge pairs: verdict agreement "
                    f"{100 * ma:.2f}%, severity correlation {mr:.3f}.**", ""]

    # ---- 4. variance decomposition -----------------------------------------
    out += ["## 4. Where judge disagreement lives", "",
            "Variance of individual sycophancy votes, split into the component from resampling the "
            "same judge at temperature 0.7 and the component from swapping judge.", "",
            "| component | variance | share |", "|---|---|---|"]
    within, between = [], []
    for rid in sample.index:
        per_judge_means = []
        for judge in judges:
            d = votes.get((rid, judge))
            if not d or len(d["sycophancy"]) < 2:
                continue
            v = d["sycophancy"]
            within.append(float(np.var(v, ddof=1)))
            per_judge_means.append(float(np.mean(v)))
        if len(per_judge_means) >= 2:
            between.append(float(np.var(per_judge_means, ddof=1)))
    w, b = float(np.mean(within)), float(np.mean(between))
    tot = w + b
    out.append(f"| within-judge (temperature resampling) | {w:.4f} | {100 * w / tot:.1f}% |")
    out.append(f"| between-judge (model identity) | {b:.4f} | {100 * b / tot:.1f}% |")
    out += ["", f"Between-judge variance is **{b / w:.1f}x** the within-judge variance. "
                "A best-of-3 consensus from a single judge controls the smaller component.", ""]

    # ---- 5. agreement with the base of record ------------------------------
    out += ["## 5. Panel versus the original judge", "",
            "| panel judge | n | verdict agreement with original | severity rho | mean bias (panel - original) |",
            "|---|---|---|---|---|"]
    for judge in judges:
        if judge == ORIGINAL:
            continue
        try:
            mask = (wide[("verdict", judge)].notna() & wide[("verdict", ORIGINAL)].notna())
        except KeyError:
            continue
        if mask.sum() < 50:
            continue
        agree = float((wide[("verdict", judge)][mask] == wide[("verdict", ORIGINAL)][mask]).mean())
        rho = float(spearmanr(wide[("syc", judge)][mask], wide[("syc", ORIGINAL)][mask]).statistic)
        bias = float((wide[("syc", judge)][mask] - wide[("syc", ORIGINAL)][mask]).mean())
        out.append(f"| {judge} | {int(mask.sum())} | {100 * agree:.2f}% | {rho:.3f} | {bias:+.3f} |")
    out.append("")

    # ---- weighted corpus-level severity means ------------------------------
    out += ["## 6. Weighted corpus-level mean sycophancy", "",
            "Sample weights reverse the stratification, so these estimate the full 8,830-response "
            "corpus rather than the sample.", "",
            "| judge | unweighted | weighted (corpus estimate) |", "|---|---|---|"]
    for judge in judges:
        g = per[per.judge == judge]
        if len(g) < 50:
            continue
        out.append(f"| {judge} | {g.syc.mean():.4f} "
                   f"| {np.average(g.syc, weights=g.weight):.4f} |")
    out.append("")

    REPORT.write_text("\n".join(out) + "\n", encoding="utf-8")
    print("\n".join(out))
    print(f"\nwritten to {REPORT}")


if __name__ == "__main__":
    main()
