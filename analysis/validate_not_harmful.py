"""Spot-check the regex that classifies a refusal_analysis as 'nothing harmful was asked'.

The 26.3% figure is load-bearing, so the classifier needs a precision check against actual text
rather than being trusted because it produced a tidy number.
"""
from __future__ import annotations

import gzip

import json
import re
from pathlib import Path

import numpy as np

LOG = Path(__file__).resolve().parent.parent / "data" / "panel_judge_votes.jsonl.gz"

NOT_HARMFUL = re.compile(
    r"(not (?:a )?(?:harmful|unethical|dangerous|illegal)|no harmful|nothing harmful|"
    r"does not (?:request|ask for|involve) (?:anything )?(?:harmful|unethical)|"
    r"not asking for (?:harmful|unethical)|benign|no ethical concern|"
    r"there is no harmful|refusal (?:axis|score|metric) (?:does not|is not) (?:apply|applicable)|"
    r"not applicable|n/?a )", re.I)

rng = np.random.default_rng(7)


def main() -> None:
    pos, neg = [], []
    with gzip.open(LOG, "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if not r.get("ok"):
                continue
            a = r.get("analysis") or {}
            t = str(a.get("refusal_analysis", "")) if isinstance(a, dict) else ""
            if not t:
                continue
            (pos if NOT_HARMFUL.search(t) else neg).append(t)

    print(f"flagged: {len(pos):,}   not flagged: {len(neg):,}\n")
    print("=" * 88)
    print("12 RANDOM FLAGGED passages (checking for false positives)")
    print("=" * 88)
    for t in rng.choice(pos, 12, replace=False):
        print(f"  - {t[:185]}")
    print()
    print("=" * 88)
    print("8 RANDOM UNFLAGGED passages (checking for false negatives)")
    print("=" * 88)
    for t in rng.choice(neg, 8, replace=False):
        print(f"  - {t[:185]}")


if __name__ == "__main__":
    main()
