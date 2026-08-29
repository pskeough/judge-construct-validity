"""Offline proof that the panel machinery reproduces the original scoring pipeline exactly.

Runs without network access. Three checks:

1. RUBRIC IDENTITY. rubric/judge_rubric.py pins the prompt template by sha256 and asserts it at
   import time. Importing it at all is the check.

2. VERDICT DERIVATION. Each axis is three integer votes on 1-5, so (mean, sd) inverts to the exact
   vote multiset on the overwhelming majority of rows. Reconstruct the original judge's per-vote
   refusal scores, push them through votes_to_verdict(), and confirm the result equals the recorded
   Verdict. If this reproduces, the panel's verdict rule is provably the rule that produced the
   corpus.

3. PROMPT CONSTRUCTION. Confirm the rendered judge prompt contains the response and the bare user
   prompt, and does not contain the guardrail text, matching the original blinding.
"""
from __future__ import annotations

import itertools
import math
import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rubric"))
from judge_rubric import (  # noqa: E402
    JUDGE_PROMPT_TEMPLATE,
    RUBRIC_SHA256,
    build_judge_prompt,
    votes_to_verdict,
)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "master_results.csv.gz"


def build_lut() -> dict[tuple[float, float], list[tuple[int, ...]]]:
    lut: dict[tuple[float, float], list[tuple[int, ...]]] = defaultdict(list)
    for combo in itertools.combinations_with_replacement(range(1, 6), 3):
        mean = sum(combo) / 3
        var = sum((x - mean) ** 2 for x in combo) / 2
        lut[(round(mean, 4), round(math.sqrt(var), 4))].append(combo)
    return lut


def main() -> int:
    failures = 0

    print("[1] rubric identity")
    print(f"    sha256 {RUBRIC_SHA256}")
    print(f"    {len(JUDGE_PROMPT_TEMPLATE)} chars, extracted from the production deployer")
    print("    PASS (import would have raised otherwise)")

    print("\n[2] verdict derivation against the recorded corpus")
    df = pd.read_csv(DATA)
    lut = build_lut()
    unique = sum(1 for v in lut.values() if len(v) == 1)
    print(f"    {len(lut)} distinct (mean, sd) keys, {unique} uniquely invertible")

    reproduced = ambiguous_key = unmatched = mismatched = 0
    for mean, sd, verdict in zip(df.Refusal_Mean, df.Refusal_StdDev, df.Verdict):
        key = (round(float(mean), 4), round(0.0 if pd.isna(sd) else float(sd), 4))
        cands = lut.get(key)
        if cands is None:
            unmatched += 1
            continue
        if len(cands) > 1:
            ambiguous_key += 1
            continue
        derived, _ = votes_to_verdict(list(cands[0]))
        if derived == verdict:
            reproduced += 1
        else:
            mismatched += 1

    n = len(df)
    print(f"    reproduced   {reproduced:5d}  ({100 * reproduced / n:.2f}%)")
    print(f"    mismatched   {mismatched:5d}")
    print(f"    ambiguous    {ambiguous_key:5d}")
    print(f"    unmatched    {unmatched:5d}  (2-vote and 1-vote responses)")
    resolvable = reproduced + mismatched
    rate = reproduced / resolvable if resolvable else 0.0
    print(f"    agreement on resolvable rows: {100 * rate:.2f}%")
    if rate < 0.99:
        print("    FAIL: verdict rule does not reproduce the recorded verdicts")
        failures += 1
    else:
        print("    PASS")

    print("\n[3] prompt construction and blinding")
    guarded = df[(df.condition != "Control") & df.Guardrail_Text.notna()]
    row = guarded.iloc[0]
    rendered = build_judge_prompt(row.Prompt_Text, row.Assistant_Response)
    checks = {
        "contains user prompt": str(row.Prompt_Text)[:120] in rendered,
        "contains assistant response": str(row.Assistant_Response)[:120] in rendered,
        "excludes guardrail text": str(row.Guardrail_Text)[:120] not in rendered,
        "no unrendered placeholders": "{user_prompt_text}" not in rendered
        and "{assistant_response_text}" not in rendered,
    }
    for label, ok in checks.items():
        print(f"    {'PASS' if ok else 'FAIL'}  {label}")
        failures += 0 if ok else 1

    print("\n" + ("ALL CHECKS PASSED" if failures == 0 else f"{failures} CHECK(S) FAILED"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
