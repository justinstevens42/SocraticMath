"""Hand-labeled gold set: Cursor's own per-question judgments.

These chains were read and labeled directly (the proposal's worked example,
``algebra/hint_0``, uses the paper's published ground truth). The gold set is
used by ``socratic-hints evaluate`` to measure how well the automatic rubric
reproduces Cursor's judgments. Labels use the f/r/b/e codes from
:mod:`taxonomy`.
"""

from __future__ import annotations

# (domain, problem_id) -> list of labels aligned with the parsed question order.
GOLD_LABELS: dict[tuple[str, str], list[str]] = {
    # From the proposal (published human labels).
    ("algebra", "hint_0"): ["f", "r", "f", "b", "b", "b", "e", "e"],
    # 9^3 + 3*9^2 + 3*9 + 1 as (9+1)^3.
    ("counting_and_probability", "hint_1"): ["f", "f", "r", "r", "b", "b", "b", "b"],
    # AAA_4 = 33_b smallest sum.
    ("number_theory", "hint_10"): ["b", "b", "b", "e"],
    # Dilation of a square.
    ("geometry", "hint_1"): ["f", "r", "f", "b", "b", "b"],
    # Reflection matrix -> direction vector.
    ("precalculus", "hint_1001"): ["f", "b", "r", "r", "b", "b", "r", "e"],
    # Phone-number probability.
    ("prealgebra", "hint_0"): ["b", "b", "b", "r", "r", "b", "r"],
    # Pascal's triangle ratio sums.
    ("intermediate_algebra", "hint_1"): ["f", "r", "b", "b", "b", "b", "b", "b", "b", "b"],
}
