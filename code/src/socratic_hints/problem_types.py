"""Quarfoot's pedagogical problem taxonomy (nine types).

Where the four *hint* types (see :mod:`taxonomy`) classify individual Socratic
questions, these nine *problem* types classify whole problems by their
pedagogical character -- how much mathematical substance they carry, whether
their value is cognitive or affective, and how open their solution space is.

The taxonomy (musical metaphor) is ordered here roughly from least to most
mathematical substance:

    First Notes  < Accidentals < Chords < Etudes < Improvisations
                 < Interpretations < First Pieces < Showpieces < Masterpieces

The codes below are the canonical short identifiers used for every matrix,
CSV column, and output filename in the project.
"""

from __future__ import annotations

from enum import Enum


class ProblemType(str, Enum):
    FIRST_NOTES = "FN"
    ACCIDENTALS = "AC"
    CHORDS = "CH"
    ETUDES = "ET"
    IMPROVISATIONS = "IM"
    INTERPRETATIONS = "IN"
    FIRST_PIECES = "FP"
    SHOWPIECES = "SP"
    MASTERPIECES = "MP"


# Canonical ordering (least -> most substance) used everywhere.
PROBLEM_TYPES: list[str] = [t.value for t in ProblemType]

PROBLEM_TYPE_NAMES: dict[str, str] = {
    "FN": "First Notes",
    "AC": "Accidentals",
    "CH": "Chords",
    "ET": "Etudes",
    "IM": "Improvisations",
    "IN": "Interpretations",
    "FP": "First Pieces",
    "SP": "Showpieces",
    "MP": "Masterpieces",
}

PROBLEM_TYPE_DESCRIPTIONS: dict[str, str] = {
    "FN": (
        "First Notes: one-step, single-path, algorithmic -- atomic skills like "
        "definitions, facts, and procedures. Least substance; value is fluency."
    ),
    "AC": (
        "Accidentals: look routine but hide a twist, usually targeting a "
        "specific misconception. Short and lean on trickery; a well-placed "
        "'gotcha' reshapes a mental model."
    ),
    "CH": (
        "Chords: built around a single idea but with more steps than a First "
        "Note; more affective than cognitive. Deepen one concept without yet "
        "combining ideas."
    ),
    "ET": (
        "Etudes: complete, cognitive, drilling one technique -- repetitive-skill "
        "or word problems with a real-world hook but no genuine real-world "
        "depth. Narrow focus builds a specific recurring tool."
    ),
    "IM": (
        "Improvisations: a mathematical playground -- intriguing and accessible "
        "but with constrained solution spaces, so first attempts often fail and "
        "you experiment. Value is creativity and play."
    ),
    "IN": (
        "Interpretations: modest substance but wide-open solution spaces with "
        "qualitatively different valid paths. The point is comparing approaches "
        "and forming judgments -- mathematical taste."
    ),
    "FP": (
        "First Pieces: first problems requiring multiple ideas plus a "
        "macro-level plan and real choices -- deciding how to proceed, not just "
        "executing. The bridge from technique to problem-solving."
    ),
    "SP": (
        "Showpieces: very high substance, cognitive over affective, hinging on "
        "a flash of insight and sometimes contrived. Pyrotechnic and hard, but "
        "not emotionally unified."
    ),
    "MP": (
        "Masterpieces: very high substance and high affect, drawing on many "
        "skills toward a generalizable truth that transcends the specific "
        "numbers. Culminating, capstone experiences."
    ),
}
