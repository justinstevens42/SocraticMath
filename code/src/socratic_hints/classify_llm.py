"""LLM-based classification of Socratic hints and Quarfoot problem types.

Mirrors the regex pipeline (:mod:`classify` / :mod:`classify_problem`) but asks
an LLM to do both classifications in one shot per problem:

* each Socratic question -> one of the four hint types (f / r / b / e), and
* the whole problem -> one of Quarfoot's nine problem types.

Problems are sent in batches to amortize prompt overhead. Every problem's
result is written to its own JSON file under ``classifications_llms/`` as soon
as its batch returns, so an interrupted run resumes where it left off.
"""

from __future__ import annotations

import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from .parse import HintRecord
from .problem_types import PROBLEM_TYPE_DESCRIPTIONS, PROBLEM_TYPES
from .taxonomy import FEW_SHOT_EXAMPLE, HINT_DESCRIPTIONS, HINT_TYPES

import os

BATCH_SIZE = int(os.environ.get("SOCRATIC_LLM_BATCH_SIZE", "10"))
MAX_WORKERS = int(os.environ.get("SOCRATIC_LLM_WORKERS", "12"))
MAX_SOLUTION_CHARS = 1800
MAX_PROBLEM_CHARS = 2500

_JSON_RE = re.compile(r"\[.*\]", re.S)


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

def build_system_prompt() -> str:
    lines = [
        "You are an expert in mathematics education and intelligent tutoring "
        "systems. For each mathematics problem you are given, you perform two "
        "classification tasks.",
        "",
        "TASK 1 - label every Socratic question in the problem's hint chain "
        "with exactly one of four hint types:",
    ]
    for code, desc in HINT_DESCRIPTIONS.items():
        lines.append(f"  {code} = {desc}")
    lines += [
        "",
        "Rules for hint labels:",
        "  - Output one label per question, in order, using only the letters f, r, b, e.",
        "  - Judge each question in the context of the whole chain and the problem.",
        "  - Chains typically progress feature/principle -> bottom-out -> extension, "
        "but always label by content, not position alone.",
        "",
        "Worked example (human gold labels):",
        f"  Problem: {FEW_SHOT_EXAMPLE['problem']}",
    ]
    for i, (q, lab) in enumerate(
        zip(FEW_SHOT_EXAMPLE["questions"], FEW_SHOT_EXAMPLE["labels"]), start=1
    ):
        lines.append(f"  {i}. [{lab}] {q}")
    lines += [
        "",
        "TASK 2 - classify the whole problem into exactly one of David "
        "Quarfoot's nine pedagogical problem types (musical metaphor, ordered "
        "roughly least -> most mathematical substance):",
    ]
    for code in PROBLEM_TYPES:
        lines.append(f"  {code} = {PROBLEM_TYPE_DESCRIPTIONS[code]}")
    lines += [
        "",
        "Judge the problem type from the problem statement, its difficulty "
        "level, and the worked solution (number of ideas involved, cognitive "
        "vs. affective payoff, and how open the solution space is).",
        "",
        "OUTPUT FORMAT: respond with ONLY a JSON array, no prose and no "
        "markdown fences. One object per problem, in the same order as given:",
        '  [{"id": "<problem id>", "problem_type": "<one of '
        + "/".join(PROBLEM_TYPES)
        + '>", "labels": ["f", "r", ...]}]',
        "The labels array must contain exactly one letter per numbered "
        "question, in order.",
    ]
    return "\n".join(lines)


def _clip(text: str, limit: int) -> str:
    text = (text or "").strip()
    return text if len(text) <= limit else text[:limit] + " [...truncated]"


def record_key(record: HintRecord) -> str:
    """Unique id sent to the model (problem ids repeat across domains)."""
    return f"{record.domain}/{record.problem_id}"


def render_problem(record: HintRecord) -> str:
    numbered = "\n".join(
        f"  {i}. {q}" for i, q in enumerate(record.questions, start=1)
    )
    return (
        f"### Problem id: {record_key(record)} "
        f"(level: {record.level or 'unknown'})\n"
        f"Problem: {_clip(record.problem, MAX_PROBLEM_CHARS)}\n"
        f"Solution: {_clip(record.solution, MAX_SOLUTION_CHARS)}\n"
        f"Socratic questions ({len(record.questions)}):\n{numbered}"
    )


def build_user_prompt(records: list[HintRecord]) -> str:
    body = "\n\n".join(render_problem(r) for r in records)
    return (
        f"Classify the following {len(records)} problem(s). Remember: output "
        "ONLY the JSON array.\n\n" + body
    )


# ---------------------------------------------------------------------------
# Response parsing / validation
# ---------------------------------------------------------------------------

@dataclass
class LLMResult:
    problem_type: str
    labels: list[str]


def parse_response(text: str, records: list[HintRecord]) -> dict[str, LLMResult]:
    """Parse and validate the model's JSON. Raises ValueError on any problem."""
    match = _JSON_RE.search(text)
    if not match:
        raise ValueError(f"no JSON array in response: {text[:200]!r}")
    items = json.loads(match.group(0))
    by_id = {record_key(r): r for r in records}
    out: dict[str, LLMResult] = {}
    for item in items:
        pid = item.get("id")
        if pid not in by_id:
            raise ValueError(f"unknown problem id {pid!r}")
        ptype = item.get("problem_type")
        labels = item.get("labels")
        if ptype not in PROBLEM_TYPES:
            raise ValueError(f"{pid}: bad problem_type {ptype!r}")
        if (
            not isinstance(labels, list)
            or len(labels) != len(by_id[pid].questions)
            or any(l not in HINT_TYPES for l in labels)
        ):
            raise ValueError(f"{pid}: bad labels {labels!r}")
        out[pid] = LLMResult(problem_type=ptype, labels=[str(l) for l in labels])
    missing = set(by_id) - set(out)
    if missing:
        raise ValueError(f"missing results for {sorted(missing)}")
    return out


# ---------------------------------------------------------------------------
# Batched, parallel, resumable driver
# ---------------------------------------------------------------------------

def classify_records_llm(
    backend,
    records: list[HintRecord],
    on_result,
    *,
    batch_size: int = BATCH_SIZE,
    max_workers: int = MAX_WORKERS,
    log=print,
) -> list[str]:
    """Classify ``records``; call ``on_result(record, LLMResult)`` per problem.

    Returns the list of problem ids that failed after all retries.
    """
    system = build_system_prompt()
    batches = [
        records[i:i + batch_size] for i in range(0, len(records), batch_size)
    ]
    failed: list[str] = []
    done_count = 0
    consecutive_bad = 0
    lock = threading.Lock()
    # Circuit breaker: when the API/usage limit trips, every call starts
    # failing. Stop the run instead of churning through thousands of
    # doomed calls -- unclassified problems simply stay pending for the
    # next (resumed) run.
    stop = threading.Event()

    def run_batch(batch: list[HintRecord]) -> None:
        nonlocal done_count, consecutive_bad
        if stop.is_set():
            return
        results: dict[str, LLMResult] = {}
        try:
            results = _attempt(backend, system, batch)
        except Exception:
            # Batch failed twice; retry each problem individually.
            for record in batch:
                if stop.is_set():
                    break
                try:
                    results.update(_attempt(backend, system, [record]))
                except Exception as exc:
                    with lock:
                        failed.append(record_key(record))
                        log(f"  [fail] {record_key(record)}: {exc}")
        for record in batch:
            if record_key(record) in results:
                on_result(record, results[record_key(record)])
        with lock:
            done_count += len(batch)
            if results:
                consecutive_bad = 0
            else:
                consecutive_bad += 1
                if consecutive_bad >= 3 and not stop.is_set():
                    stop.set()
                    log(
                        "  [abort] 3 consecutive batches failed entirely -- "
                        "likely usage limit; stopping run (resume later)."
                    )
            if done_count % 100 < len(batch):
                log(f"  ... {done_count}/{len(records)} problems classified")

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(run_batch, b) for b in batches]
        for f in as_completed(futures):
            f.result()
    return failed


def _attempt(backend, system: str, batch: list[HintRecord]) -> dict[str, LLMResult]:
    """One batch with a single retry on transport or validation errors."""
    last: Exception | None = None
    for _ in range(2):
        try:
            raw = backend.complete(system, build_user_prompt(batch))
            return parse_response(raw, batch)
        except Exception as exc:  # noqa: BLE001 - retry then surface
            last = exc
    raise last  # type: ignore[misc]
