"""Loading hint JSON files and splitting the Socratic question chains."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

# Splits a numbered list like "1. ... \n2. ... \n10. ..." on the leading
# "<number>." markers, tolerant of leading whitespace and blank lines.
_ENUM_SPLIT = re.compile(r"(?m)^\s*\d+[.)]\s+")


@dataclass
class HintRecord:
    domain: str
    problem_id: str          # e.g. "hint_0"
    path: Path
    problem: str
    level: str
    type: str
    solution: str
    questions: list[str]


def split_socratic_questions(raw: str) -> list[str]:
    """Split the ``socratic_questions`` field into individual questions."""
    if not raw:
        return []
    parts = _ENUM_SPLIT.split(raw)
    # The first chunk is whatever preceded "1." (usually empty).
    questions = [p.strip() for p in parts if p.strip()]
    return questions


def load_hint_file(path: Path, domain: str) -> HintRecord | None:
    """Parse a single ``hint_<n>.json`` file into a :class:`HintRecord`."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return None
    questions = split_socratic_questions(data.get("socratic_questions", ""))
    if not questions:
        return None
    return HintRecord(
        domain=domain,
        problem_id=path.stem,
        path=path,
        problem=data.get("problem", ""),
        level=data.get("level", ""),
        type=data.get("type", ""),
        solution=data.get("solution", ""),
        questions=questions,
    )


def iter_domain_records(domain_dir: Path, domain: str):
    """Yield every parseable :class:`HintRecord` in a domain directory."""
    for path in sorted(domain_dir.glob("hint_*.json")):
        record = load_hint_file(path, domain)
        if record is not None:
            yield record
