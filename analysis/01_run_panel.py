"""Run the multi-judge panel over the stratified sample.

Every judge scores every sampled response on the SAME rubric the base of record was scored with
(see judge_rubric.py, which hashes the original to prevent drift), with the same temperature (0.7),
the same vote count (3), and the same blinding: the judge sees Prompt_Text only, never the
guardrail, exactly as sycophancy_deployer_openrouter.py:329 does.

Every individual vote is written to judge_logs.jsonl with its raw reasoning. That file has never
existed for this project; its absence has been the single largest limitation in every audit so far.

The run is resumable and idempotent. Completed (response_id, judge, vote_index) triples are skipped
on restart, so an interrupted run costs nothing to resume.

Usage:
    python panel_2026_08/01_run_panel.py --smoke            # 3 responses x all judges, prints cost
    python panel_2026_08/01_run_panel.py --limit 50         # partial run
    python panel_2026_08/01_run_panel.py                    # full run
    python panel_2026_08/01_run_panel.py --judges deepseek/deepseek-v4-flash-0731
"""
from __future__ import annotations

import gzip

import argparse
import asyncio
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "rubric"))
from judge_rubric import (  # noqa: E402
    JUDGE_TEMPERATURE,
    N_VOTES,
    RUBRIC_SHA256,
    build_judge_prompt,
    votes_to_verdict,
)

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SAMPLE = ROOT / "data" / "panel_sample.csv.gz"
SAMPLE_HASH = HERE / "sample" / "panel_sample.sha256"
OUTDIR = HERE / "out"
LOG = OUTDIR / "judge_logs.jsonl"

API = "https://openrouter.ai/api/v1/chat/completions"

# id -> (input $/M, output $/M). Prices confirmed from the OpenRouter models API 2026-08-12.
# Dated snapshots are preferred where they exist: an undated alias can be repointed by the
# provider mid-run, which would silently break comparability.
JUDGES: dict[str, tuple[float, float]] = {
    "deepseek/deepseek-v4-flash-0731": (0.08, 0.18),
    "moonshotai/kimi-k2.5": (0.57, 2.85),
    "google/gemini-3.5-flash-lite": (0.30, 2.50),
    "z-ai/glm-5.2": (0.50, 3.15),
    "qwen/qwen3.5-flash-02-23": (0.065, 0.26),
}

# Concurrency is PER JUDGE, not global. A single shared semaphore lets one slow provider occupy
# every slot: DeepSeek V4 Flash emits ~1,200 output tokens per vote and takes far longer than
# Gemini Flash Lite at ~235, so under a shared limit the fast judges sit idle waiting on the slow
# one. Per-judge semaphores let each provider run at its own ceiling.
PER_JUDGE_CONCURRENCY = 32
MAX_ATTEMPTS = 4

# httpx defaults to max_keepalive_connections=20, which throttles anything above that regardless
# of the semaphore. Size the pool to the total concurrency we actually intend to use.
def _limits(n_judges: int) -> httpx.Limits:
    total = PER_JUDGE_CONCURRENCY * max(n_judges, 1)
    return httpx.Limits(max_connections=total + 16, max_keepalive_connections=total + 16)


class Ledger:
    """Tracks spend from the usage figures the API actually returns, not from estimates."""

    def __init__(self) -> None:
        self.rows: list[dict] = []

    def add(self, judge: str, pin: int, pout: int) -> None:
        i, o = JUDGES.get(judge, (0.0, 0.0))
        self.rows.append(
            {"judge": judge, "in": pin, "out": pout, "usd": pin / 1e6 * i + pout / 1e6 * o}
        )

    def total(self) -> float:
        return sum(r["usd"] for r in self.rows)

    def over(self, budget: float | None) -> bool:
        return budget is not None and self.total() >= budget

    def summary(self) -> pd.DataFrame:
        if not self.rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(self.rows)
            .groupby("judge")
            .agg(calls=("usd", "size"), tok_in=("in", "sum"), tok_out=("out", "sum"),
                 usd=("usd", "sum"))
            .round(4)
        )


AXES = ("sycophancy", "truthfulness", "refusal")


def extract_scores(parsed: dict) -> dict[str, int]:
    """Pull the three axis scores out of a judge reply.

    The rubric asks for {"scores": {...}}, and most models comply, but not all. Observed variants:
    a flat top-level object, and scores nested one level deeper. Accept any shape that yields three
    integers on the three named axes; reject anything ambiguous rather than guessing, because a
    silently mis-parsed score is worse than a retry.
    """
    candidates = [parsed]
    if isinstance(parsed.get("scores"), dict):
        candidates.insert(0, parsed["scores"])
    for value in parsed.values():
        if isinstance(value, dict):
            candidates.append(value)
            if isinstance(value.get("scores"), dict):
                candidates.insert(0, value["scores"])

    for cand in candidates:
        if not isinstance(cand, dict):
            continue
        if all(axis in cand for axis in AXES):
            out = {}
            for axis in AXES:
                raw = cand[axis]
                if isinstance(raw, dict):  # e.g. {"score": 3, "reason": "..."}
                    raw = raw.get("score", raw.get("value"))
                if raw is None:
                    break
                score = int(round(float(raw)))
                if not 1 <= score <= 5:
                    break
                out[axis] = score
            if len(out) == 3:
                return out
    raise ValueError(f"no usable scores in reply with keys {list(parsed)[:6]}")


def extract_content(payload: dict) -> str:
    """Get the assistant text. Some models return null content and put the body in `reasoning`."""
    msg = (payload.get("choices") or [{}])[0].get("message") or {}
    for field in ("content", "reasoning"):
        val = msg.get(field)
        if isinstance(val, str) and val.strip():
            return val
    raise ValueError("empty content and empty reasoning")


def load_done() -> set[tuple[str, str, int]]:
    done: set[tuple[str, str, int]] = set()
    if LOG.exists():
        with gzip.open(LOG, "rt", encoding="utf-8") as fh:
            for line in fh:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("ok"):
                    done.add((rec["response_id"], rec["judge"], rec["vote_index"]))
    return done


async def one_vote(client: httpx.AsyncClient, sem: asyncio.Semaphore, key: str,
                   judge: str, row: pd.Series, vote_index: int,
                   ledger: Ledger, lock: asyncio.Lock,
                   budget: float | None = None) -> dict | None:
    # Hard stop. Checked before every call, so an unexpected token blowup cannot run away.
    if ledger.over(budget):
        return None
    prompt = build_judge_prompt(row["Prompt_Text"], row["Assistant_Response"])
    body = {
        "model": judge,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": JUDGE_TEMPERATURE,
        "response_format": {"type": "json_object"},
    }
    headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}

    async with sem:
        for attempt in range(1, MAX_ATTEMPTS + 1):
            try:
                r = await client.post(API, json=body, headers=headers, timeout=180.0)
                if r.status_code == 429 or r.status_code >= 500:
                    await asyncio.sleep(min(2 ** attempt, 30))
                    continue
                r.raise_for_status()
                payload = r.json()
                content = extract_content(payload)
                usage = payload.get("usage", {}) or {}
                # Some models wrap JSON in prose or a fenced block; take the outermost object.
                start, end = content.find("{"), content.rfind("}")
                parsed = json.loads(content[start:end + 1] if start >= 0 < end else content)
                scores = extract_scores(parsed)
                rec = {
                    "response_id": row["Response_ID"],
                    "judge": judge,
                    "vote_index": vote_index,
                    "ok": True,
                    "sycophancy": scores["sycophancy"],
                    "truthfulness": scores["truthfulness"],
                    "refusal": scores["refusal"],
                    "analysis": parsed.get("analysis", {}),
                    "tokens_in": usage.get("prompt_tokens", 0),
                    "tokens_out": usage.get("completion_tokens", 0),
                    "attempt": attempt,
                    "rubric_sha256": RUBRIC_SHA256,
                }
                async with lock:
                    ledger.add(judge, rec["tokens_in"], rec["tokens_out"])
                return rec
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
                if attempt == MAX_ATTEMPTS:
                    return {"response_id": row["Response_ID"], "judge": judge,
                            "vote_index": vote_index, "ok": False,
                            "error": f"parse: {type(exc).__name__}: {exc}"}
                await asyncio.sleep(1.5 * attempt)
            except Exception as exc:  # network/HTTP
                if attempt == MAX_ATTEMPTS:
                    return {"response_id": row["Response_ID"], "judge": judge,
                            "vote_index": vote_index, "ok": False,
                            "error": f"{type(exc).__name__}: {exc}"}
                await asyncio.sleep(min(2 ** attempt, 30))
    return None


async def run(sample: pd.DataFrame, judges: list[str], key: str,
              budget: float | None = None) -> Ledger:
    OUTDIR.mkdir(parents=True, exist_ok=True)
    done = load_done()
    if done:
        print(f"resuming: {len(done)} votes already recorded")

    tasks_spec = [
        (judge, row, v)
        for judge in judges
        for _, row in sample.iterrows()
        for v in range(N_VOTES)
        if (row["Response_ID"], judge, v) not in done
    ]
    print(f"{len(tasks_spec)} votes to cast "
          f"({len(sample)} responses x {len(judges)} judges x {N_VOTES} votes)")
    if not tasks_spec:
        return Ledger()

    ledger, lock = Ledger(), asyncio.Lock()
    sems = {j: asyncio.Semaphore(PER_JUDGE_CONCURRENCY) for j in judges}
    started = time.time()
    completed = failed = 0
    print(f"concurrency: {PER_JUDGE_CONCURRENCY}/judge x {len(judges)} judges "
          f"= {PER_JUDGE_CONCURRENCY * len(judges)} in flight")

    async with httpx.AsyncClient(limits=_limits(len(judges)),
                                 timeout=httpx.Timeout(180.0, connect=30.0)) as client:
        with LOG.open("a", encoding="utf-8") as fh:
            pending = [one_vote(client, sems[j], key, j, r, v, ledger, lock, budget)
                       for j, r, v in tasks_spec]
            skipped = 0
            for coro in asyncio.as_completed(pending):
                rec = await coro
                if rec is None:
                    skipped += 1
                    continue
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fh.flush()
                completed += 1
                if not rec.get("ok"):
                    failed += 1
                if completed % 25 == 0 or completed == len(tasks_spec):
                    rate = completed / max(time.time() - started, 1e-9)
                    eta = (len(tasks_spec) - completed) / max(rate, 1e-9)
                    print(f"  {completed}/{len(tasks_spec)} | fail {failed} | "
                          f"${ledger.total():.3f} | {rate:.1f}/s | eta {eta/60:.1f}m")
            if skipped:
                print(f"  !! {skipped} votes SKIPPED: budget ${budget:.2f} reached. "
                      f"Re-run with a higher --budget to resume; nothing is lost.")
    return ledger


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="3 responses per judge, then stop")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--judges", nargs="*", default=list(JUDGES))
    ap.add_argument("--budget", type=float, default=17.0,
                    help="hard USD ceiling for this invocation; run aborts cleanly when reached")
    args = ap.parse_args()

    load_dotenv(ROOT / ".env")
    key = os.getenv("OPENROUTER_API_KEY")
    if not key:
        sys.exit("OPENROUTER_API_KEY not set in .env")

    expected = SAMPLE_HASH.read_text(encoding="utf-8").strip()
    actual = hashlib.sha256(SAMPLE.read_bytes()).hexdigest()
    if expected != actual:
        sys.exit(f"sample file changed since it was built\n  expected {expected}\n  actual   {actual}")

    sample = pd.read_csv(SAMPLE)
    if args.smoke:
        sample = sample.head(3)
    elif args.limit:
        sample = sample.head(args.limit)

    unknown = [j for j in args.judges if j not in JUDGES]
    if unknown:
        sys.exit(f"unknown judges (no price recorded): {unknown}")

    print(f"rubric sha256 {RUBRIC_SHA256[:16]}... verified against the original deployer")
    print(f"sample sha256 {actual[:16]}...  n={len(sample)}")
    print(f"judges: {', '.join(args.judges)}")

    print(f"budget ceiling: ${args.budget:.2f}")
    ledger = asyncio.run(run(sample, args.judges, key, args.budget))

    print("\nspend (from API-reported usage):")
    summary = ledger.summary()
    if not summary.empty:
        print(summary.to_string())
        print(f"\nTOTAL ${ledger.total():.4f}")
        per_resp = ledger.total() / max(len(sample), 1)
        print(f"per response across {len(args.judges)} judges: ${per_resp:.5f}")
        print(f"projected full run (n=1200): ${per_resp * 1200:.2f}")


if __name__ == "__main__":
    main()
