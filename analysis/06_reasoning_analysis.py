"""Three analyses the panel enables that we have not run.

A. THE REFUSAL AXIS DOES NOT APPLY. The judges' own refusal_analysis text often says the prompt was
   not a harmful request at all. If that is systematic, it is the qualitative mechanism behind the
   ~65-point construct residual, stated by the judges rather than inferred by us.

B. VERBOSITY BIAS. Flagged by the methods reviewer and never tested: does response length predict
   the sycophancy score independently of the response's content?

C. WHAT MAKES A RESPONSE HARD TO JUDGE. With four judges per response we can ask which responses
   they disagree about, and whether disagreement is predictable.
"""
from __future__ import annotations

import gzip

import json
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOG = ROOT / "data" / "panel_judge_votes.jsonl.gz"
SAMPLE = ROOT / "data" / "panel_sample.csv.gz"

NOT_HARMFUL = re.compile(
    r"(not (?:a )?(?:harmful|unethical|dangerous|illegal)|no harmful|nothing harmful|"
    r"does not (?:request|ask for|involve) (?:anything )?(?:harmful|unethical)|"
    r"not asking for (?:harmful|unethical)|benign|no ethical concern|"
    r"there is no harmful|refusal (?:axis|score|metric) (?:does not|is not) (?:apply|applicable)|"
    r"not applicable|n/?a )", re.I)


def main() -> None:
    sample = pd.read_csv(SAMPLE).set_index("Response_ID")
    rows = []
    with gzip.open(LOG, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            a = r.get("analysis") or {}
            if not isinstance(a, dict):
                a = {}
            rows.append({
                "rid": r["response_id"],
                "judge": r["judge"].split("/")[-1],
                "syc": r["sycophancy"], "ref": r["refusal"],
                "ref_text": str(a.get("refusal_analysis", "")),
                "syc_text": str(a.get("sycophancy_analysis", "")),
                "words": r.get("tokens_out", 0),
            })
    v = pd.DataFrame(rows)
    v = v.join(sample[["Category", "condition", "Assistant_Response"]], on="rid")
    v["resp_chars"] = v.Assistant_Response.astype(str).str.len()

    print("=" * 82)
    print("A. How often does the judge say the refusal axis has nothing to grade?")
    print("=" * 82)
    v["not_harmful"] = v.ref_text.str.contains(NOT_HARMFUL, na=False)
    print(f"\noverall: {100*v.not_harmful.mean():.1f}% of {len(v):,} votes\n")
    print("by judge:")
    print((100 * v.groupby("judge").not_harmful.mean()).round(1).to_string())
    print("\nby prompt category (sorted):")
    bycat = (100 * v.groupby("Category").not_harmful.mean()).sort_values(ascending=False)
    print(bycat.round(1).to_string())
    print("\ncross-tab against the refusal score given:")
    print(pd.crosstab(v.ref, v.not_harmful, normalize="index").mul(100).round(1).to_string())
    hi = v[(v.syc >= 3) & v.not_harmful]
    print(f"\nvotes scoring sycophancy >= 3 WHERE the judge says nothing harmful was asked: "
          f"{len(hi):,} ({100*len(hi)/len(v):.1f}% of all votes)")
    print("   These are exactly the responses the verdict cannot see.")

    print()
    print("=" * 82)
    print("B. Verbosity bias: does response length predict the sycophancy score?")
    print("=" * 82)
    per = v.groupby("rid").agg(syc=("syc", "mean"), chars=("resp_chars", "first"),
                               cat=("Category", "first"))
    rho = spearmanr(per.chars, per.syc)
    print(f"\noverall Spearman(response length, mean sycophancy) = {rho.statistic:+.3f} "
          f"(p={rho.pvalue:.2e}, n={len(per)})")
    print("\nwithin category, to remove the category confound:")
    for c, g in per.groupby("cat"):
        if len(g) > 40:
            r = spearmanr(g.chars, g.syc)
            print(f"   {c:26s} n={len(g):4d}  rho={r.statistic:+.3f}  p={r.pvalue:.3f}")

    print()
    print("=" * 82)
    print("C. What makes a response hard to judge?")
    print("=" * 82)
    w = v.pivot_table(index="rid", columns="judge", values="syc", aggfunc="mean")
    w = w.dropna()
    spread = w.max(axis=1) - w.min(axis=1)
    meta = sample.loc[spread.index]
    print(f"\nn={len(spread)} responses scored by all judges")
    print(f"between-judge spread in mean sycophancy: median {spread.median():.2f}, "
          f"p90 {spread.quantile(0.9):.2f}, max {spread.max():.2f}")
    print(f"responses where judges differ by >= 2 points: {(spread>=2).sum()} "
          f"({100*(spread>=2).mean():.1f}%)")
    print("\nmean spread by category:")
    print(meta.assign(sp=spread).groupby("Category").sp.mean().sort_values(
        ascending=False).round(3).to_string())
    print("\nmean spread by severity bucket (the paper's own bands):")
    print(meta.assign(sp=spread).groupby("severity").sp.mean().round(3).to_string())
    print(f"\nSpearman(response length, judge spread) = "
          f"{spearmanr(meta.Assistant_Response.astype(str).str.len(), spread).statistic:+.3f}")


if __name__ == "__main__":
    main()
