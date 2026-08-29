"""Numbers needed to (a) merge Tables 3 and 4 and (b) write the missing panel subsection."""
from __future__ import annotations

import itertools
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
AXES = ["sycophancy", "truthfulness", "refusal"]
ORIGINAL = "gemini-3-pro-preview (reconstructed)"


def level(s: float) -> str:
    if s < 2.0:
        return "L1 clean"
    if s < 3.0:
        return "L2 mild"
    if s < 4.0:
        return "L3 moderate"
    return "L4-5 severe"


def main() -> None:
    sample = pd.read_csv(ROOT / "data" / "panel_sample.csv.gz").set_index("Response_ID")
    votes: dict[tuple[str, str], dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
    with gzip.open(ROOT / "data" / "panel_judge_votes.jsonl.gz", "rt", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if r.get("ok"):
                for a in AXES:
                    votes[(r["response_id"], r["judge"].split("/")[-1])][a].append(int(r[a]))

    lut = defaultdict(list)
    for c in itertools.combinations_with_replacement(range(1, 6), 3):
        m = sum(c) / 3
        v = sum((x - m) ** 2 for x in c) / 2
        lut[(round(m, 4), round(math.sqrt(v), 4))].append(c)
    uniq = {k: v[0] for k, v in lut.items() if len(v) == 1}
    for rid, row in sample.iterrows():
        t = {}
        for a, col in zip(AXES, ["Sycophancy", "Truthfulness", "Refusal"]):
            key = (round(float(row[f"{col}_Mean"]), 4),
                   round(0.0 if pd.isna(row[f"{col}_StdDev"]) else float(row[f"{col}_StdDev"]), 4))
            if key not in uniq:
                break
            t[a] = list(uniq[key])
        if len(t) == 3:
            votes[(rid, ORIGINAL)] = t

    judges = sorted({j for _, j in votes})

    print("=== (a) judge spread on the LEVEL bands, so Tables 3 and 4 can merge ===")
    w = pd.DataFrame({j: {rid: np.mean(votes[(rid, j)]["sycophancy"])
                          for rid in sample.index if (rid, j) in votes} for j in judges}).dropna()
    spread = w.max(axis=1) - w.min(axis=1)
    lv = sample.loc[spread.index, "Sycophancy_Mean"].map(level)
    tbl = pd.DataFrame({"spread": spread, "level": lv}).groupby("level").agg(
        n=("spread", "size"), mean_spread=("spread", "mean"))
    print(tbl.round(3).to_string())

    print("\n=== (b) each judge reproduces the gap on its OWN axes ===")
    print(f"{'judge':30s} {'n':>5s} {'R2 own verdict':>15s} {'eta2 own refusal':>17s}")
    for j in judges:
        rows = [(np.mean(votes[(rid, j)]["sycophancy"]), np.mean(votes[(rid, j)]["refusal"]))
                for rid in sample.index if (rid, j) in votes and votes[(rid, j)]["refusal"]]
        y = np.array([a for a, _ in rows]); r = np.array([b for _, b in rows])
        flag = (r > 3).astype(float)
        def r2(x):
            X = np.column_stack([np.ones_like(x), x])
            b, *_ = np.linalg.lstsq(X, y, rcond=None)
            return 1 - ((y - X @ b) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        d = pd.DataFrame({"g": np.round(r, 4), "y": y})
        eta = 1 - ((d.y - d.groupby("g").y.transform("mean")) ** 2).sum() / ((d.y - d.y.mean()) ** 2).sum()
        print(f"{j:30s} {len(y):5d} {r2(flag):15.4f} {eta:17.4f}")

    print("\n=== (c) anchor collapse: share of votes on the unanchored points 2 and 4 ===")
    for j in judges:
        for axis in ["sycophancy", "refusal"]:
            v = [x for (rid, jj), dd in votes.items() if jj == j for x in dd[axis]]
            c = Counter(v)
            print(f"  {j:30s} {axis:12s} {100*(c[2]+c[4])/max(len(v),1):5.2f}%")

    print("\n=== (d) panel scale ===")
    ok = sum(1 for (rid, j) in votes if j != ORIGINAL)
    print(f"  responses {len(sample)}, judges {len(judges)} ({len(judges)-1} newly run), "
          f"votes {sum(len(d['sycophancy']) for (rid,j),d in votes.items() if j != ORIGINAL):,}")


if __name__ == "__main__":
    main()
