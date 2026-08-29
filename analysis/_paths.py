"""Resolved paths for the released layout, with gzip-aware readers.

The release ships data gzipped under data/. pandas reads .gz directly; the JSONL needs gzip.open.
"""
from __future__ import annotations

import gzip
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"

MASTER_RESULTS = DATA / "master_results.csv.gz"
PANEL_SAMPLE = DATA / "panel_sample.csv.gz"
JUDGE_LOG = DATA / "panel_judge_votes.jsonl.gz"
PROMPT_SET = ROOT / "prompts" / "prompt_set.csv"

# make `import judge_rubric` work from anywhere in the repo
sys.path.insert(0, str(ROOT / "rubric"))


def open_judge_log(path: Path | None = None):
    """Open the gzipped per-vote log for line-by-line reading."""
    return gzip.open(path or JUDGE_LOG, "rt", encoding="utf-8")
