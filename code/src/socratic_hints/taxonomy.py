"""The four Socratic hint types and the labeling rubric used by Cursor.

The first three types come from Goldin, Koedinger & Aleven (2013), "Hints: you
can't have just one." For Socratic chains we add a fourth type, *extension*,
which pushes the student's thinking beyond the current problem.
"""

from __future__ import annotations

from enum import Enum


class HintType(str, Enum):
    FEATURE_POINTING = "f"
    PRINCIPLE_STATING = "r"
    BOTTOM_OUT = "b"
    EXTENSION = "e"


# Canonical ordering used for every matrix/vector in the project.
HINT_TYPES: list[str] = [t.value for t in HintType]  # ["f", "r", "b", "e"]

HINT_NAMES: dict[str, str] = {
    "f": "feature-pointing",
    "r": "principle-stating",
    "b": "bottom-out",
    "e": "extension",
}

HINT_DESCRIPTIONS: dict[str, str] = {
    "f": (
        "Feature-pointing: draws the student's attention to a specific given, "
        "quantity, or salient aspect of THIS problem (e.g. 'Notice that these "
        "two angles are vertical angles', 'What is the center of dilation in "
        "this problem?')."
    ),
    "r": (
        "Principle-stating: supplies or asks the student to recall a general "
        "domain principle, rule, definition, theorem, or formula that is not "
        "specific to this problem's numbers (e.g. 'Vertical angles are equal in "
        "measure', 'What is the general formula for expanding (x+y)^3?')."
    ),
    "b": (
        "Bottom-out: directly advances the solution by asking the student to "
        "carry out a concrete step on THIS problem -- set up, substitute, "
        "solve, compute, or simplify (e.g. 'Add these two known angles to find "
        "the missing one', 'What are the values of x and y?')."
    ),
    "e": (
        "Extension: pushes thinking beyond the current problem -- real-life "
        "applications, significance, generalizations, or other examples (e.g. "
        "'Can you think of any other examples of piecewise functions?')."
    ),
}

# Few-shot example with human ground-truth labels, taken verbatim from the
# proposal (algebra hint_0). Used both in the LLM system prompt and as a gold
# anchor for evaluation.
FEW_SHOT_EXAMPLE = {
    "problem": (
        "Let f(x) = ax+3 if x>2; x-5 if -2<=x<=2; 2x-b if x<-2. "
        "Find a+b if the piecewise function is continuous."
    ),
    "questions": [
        "Can you explain what a piecewise function is and how it is represented algebraically?",
        "How would you determine if the given piecewise function is continuous at the points where different functions are used?",
        "What conditions need to be satisfied for a piecewise function to be continuous?",
        "Can you identify the value of a and b that would make the piecewise function continuous?",
        "How would you approach solving for a and b in order to make the function continuous?",
        "Are there any restrictions or constraints on the values of a and b that need to be considered for the function to be continuous?",
        "Can you explain the significance of a continuous function in real-life applications or mathematical concepts?",
        "Can you think of any other examples of piecewise functions or situations where continuity is important?",
    ],
    "labels": ["f", "r", "f", "b", "b", "b", "e", "e"],
}


def build_system_prompt() -> str:
    """The strict system prompt Cursor uses to label a Socratic chain.

    Returned as text so it can be (a) sent to an LLM via ``classify.py`` or
    (b) inspected/audited. It encodes the rubric and the few-shot anchor.
    """
    lines = [
        "You are an expert in mathematics education and intelligent tutoring "
        "systems. You label each Socratic question in a hint chain with exactly "
        "one of four hint types.",
        "",
        "Hint types:",
    ]
    for code, desc in HINT_DESCRIPTIONS.items():
        lines.append(f"  {code} = {desc}")
    lines += [
        "",
        "Rules:",
        "  - Output one label per question, in order, using only the letters f, r, b, e.",
        "  - Judge each question in the context of the whole chain and the problem.",
        "  - Chains typically progress feature/principle -> bottom-out -> extension, "
        "but always label by content, not position alone.",
        "",
        "Few-shot example:",
        f"  Problem: {FEW_SHOT_EXAMPLE['problem']}",
    ]
    for i, (q, lab) in enumerate(
        zip(FEW_SHOT_EXAMPLE["questions"], FEW_SHOT_EXAMPLE["labels"]), start=1
    ):
        lines.append(f"  {i}. [{lab}] {q}")
    lines += [
        "",
        "Now label the provided chain. Respond with a JSON array of labels only, "
        'e.g. ["f", "r", "b", "e"].',
    ]
    return "\n".join(lines)
