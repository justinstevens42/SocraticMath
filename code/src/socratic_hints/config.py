"""Project paths and shared configuration.

The repository root is located by walking up from this file until a directory
containing ``hint_algebra`` is found, so the package works regardless of where
it is invoked from.
"""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: Path | None = None) -> Path:
    """Return the repository root (the dir that holds the ``hint_*`` folders)."""
    here = (start or Path(__file__)).resolve()
    for parent in [here, *here.parents]:
        if (parent / "hint_algebra").is_dir():
            return parent
    # Fallback: assume <repo>/code/src/socratic_hints/config.py layout.
    return Path(__file__).resolve().parents[3]


REPO_ROOT: Path = find_repo_root()
CODE_DIR: Path = REPO_ROOT / "code"
OUTPUT_DIR: Path = CODE_DIR / "outputs"
CLASSIFICATIONS_DIR: Path = CODE_DIR / "classifications"

# Master CSV of every classified question (shareable, per the proposal).
CLASSIFIED_CSV: Path = OUTPUT_DIR / "classified_hints.csv"

# LLM-backed variants of the pipeline (same layout, separate folders so the
# regex and LLM classifications can be compared side by side).
OUTPUT_LLM_DIR: Path = CODE_DIR / "outputs_llms"
CLASSIFICATIONS_LLM_DIR: Path = CODE_DIR / "classifications_llms"
CLASSIFIED_LLM_CSV: Path = OUTPUT_LLM_DIR / "classified_hints.csv"


def use_llm_paths() -> None:
    """Point the shared pipeline at the ``*_llms`` directories."""
    global OUTPUT_DIR, CLASSIFICATIONS_DIR, CLASSIFIED_CSV
    OUTPUT_DIR = OUTPUT_LLM_DIR
    CLASSIFICATIONS_DIR = CLASSIFICATIONS_LLM_DIR
    CLASSIFIED_CSV = CLASSIFIED_LLM_CSV


def domain_dirs() -> list[Path]:
    """All ``hint_*`` domain directories, sorted by name."""
    return sorted(p for p in REPO_ROOT.glob("hint_*") if p.is_dir())


def domain_name(path: Path) -> str:
    """``hint_counting_and_probability`` -> ``counting_and_probability``."""
    return path.name[len("hint_"):]
