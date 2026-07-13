"""Classify whole problems into Quarfoot's nine pedagogical types.

This is a transparent, deterministic heuristic -- no LLM required -- so all
~4,000 problems can be labeled in seconds and the reasoning is auditable.

Design
------
Quarfoot's taxonomy is organized along a few interpretable axes:

* **Substance** -- how much mathematical depth / how many ideas a problem
  carries. Proxied by difficulty ``level``, solution length, the number of
  reasoning steps, and the length of the Socratic chain.
* **Cognitive vs. affective** -- whether a problem's payoff is a clever
  insight (cognitive) or a durable, meaningful experience (affective).
  Proxied by the share of *extension* hints (``e``) in the Socratic chain and
  by generalization/proof language.
* **Solution-space openness** -- one path vs. many qualitatively different
  paths. Proxied by explicit alternative-approach markers ("alternatively",
  "another method") in the worked solution.

Each problem gets a numeric feature vector (:func:`extract_features`). Every
type then receives a score: a smooth preference for a target substance level
(so problems flow along the substance axis) plus additive bonuses when the
type's *signature* signals fire (:func:`score_types`). The argmax is the
label. Weights were tuned on the dataset so the distribution spreads sensibly
across all nine types (fluency drills common, masterpieces/accidentals rare).

Because the taxonomy is qualitative, these labels are heuristic proxies of
Quarfoot's axes, not ground truth -- but they are reproducible and inspectable.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass

from .parse import HintRecord
from .problem_types import PROBLEM_TYPES

# ---------------------------------------------------------------------------
# Regex signal banks (compiled once)
# ---------------------------------------------------------------------------

_LEVEL_RE = re.compile(r"level\s*(\d)", re.I)

# Reasoning-step markers in a worked solution (proxy for number of steps).
_STEP_RE = re.compile(
    r"\b(therefore|thus|hence|then|next|first|second|third|fourth|finally|"
    r"now|so that|substitut|we get|we have|we find|it follows|this (gives|"
    r"implies|means)|so,)\b",
    re.I,
)

# Explicit multiple / qualitatively different solution paths -> open space.
_ALT_RE = re.compile(
    r"(alternativel|alternatel|another (way|approach|method|solution)|"
    r"a second (method|approach|solution)|(we|you) (can|could) also|"
    r"or,? we (can|could)|yet another|a different (way|approach)|"
    r"second solution|one (way|approach) is|another is to)",
    re.I,
)

# Generalization / proof language -> transcends the specific numbers.
_PROOF_RE = re.compile(
    r"\b(prove|proof|show that|demonstrate that|for all\b|for every\b|"
    r"for any\b|for each\b|in general\b|generaliz|it follows that|"
    r"in every case|for all real|for all positive|holds? for)",
    re.I,
)

# Real-world hook (word problems).
_WORD_RE = re.compile(
    r"\b(dollars?|cents?|\bmiles?\b|kilomet|\bmeters?\b|\bhours?\b|minutes?|"
    r"seconds?|\bdays?\b|weeks?|months?|years old|age\b|apples?|oranges?|"
    r"marbles?|coins?|\bcards?\b|students?|people|children|\bteam\b|\bgame\b|"
    r"\brace\b|\bspeed\b|distance|\bmoney\b|\bcost\b|\bprice\b|\bpaid\b|"
    r"profit|percent|discount|recipe|\bpaint\b|garden|\bfence\b|\bclock\b|"
    r"calendar|\bdice\b|\bdie\b|\bball\b|box of|bag of|\bheight\b|\bweight\b|"
    r"gallons?|liters?|\bfeet\b|inches|\bpounds?\b|degrees? (celsius|fahren))",
    re.I,
)

# Enumerate / experiment -> playground with a constrained but explorable space.
_ENUM_RE = re.compile(
    r"(how many (ways|different|distinct|possible|integers|values|solutions|"
    r"pairs|triples|ordered)|number of ways|in how many|find all|list all|"
    r"all possible|possible values|how many ways|smallest (positive )?"
    r"(integer|value|number)|largest (possible )?(value|integer|number)|"
    r"greatest (possible )?(value|integer|number))",
    re.I,
)

# "Gotcha" / misconception language explained in the solution -> Accidental.
_TWIST_RE = re.compile(
    r"(be careful|careful[:,]|however|but wait|common (mistake|error|"
    r"misconception)|watch out|don'?t forget|(remember|note|notice) (to|that|"
    r"however)|the key (is|insight)|surprisingly|\btrick(y|)\b|it is tempting|"
    r"(you |one )?might (think|assume|expect)|seems? (like|to)|appears? to|"
    r"a common|at first glance|counterintuitiv|not (equal|the same)|"
    r"actually,|in fact,)",
    re.I,
)

# Flash-of-insight / contrived-construction language -> Showpiece.
_INSIGHT_RE = re.compile(
    r"(consider the|construct|auxiliary|introduce|let us define|the (key |clever "
    r")?(trick|insight|observation)|by symmetry|\bWLOG\b|without loss|"
    r"add and subtract|multiply (both sides|numerator)|complete the square|"
    r"telescop|clever|remarkabl|elegant|the substitution|notice that)",
    re.I,
)

# Asymptote figure -> geometric diagram / construction.
_ASY_RE = re.compile(r"\[asy\]", re.I)


def level_num(level: str) -> int:
    """Parse ``"Level 3"`` -> ``3``; default to 3 when missing/unknown."""
    m = _LEVEL_RE.search(level or "")
    return int(m.group(1)) if m else 3


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

@dataclass
class ProblemFeatures:
    level: int
    n_steps: int
    sol_len: int
    n_questions: int
    n_methods: int          # alternative-approach markers
    n_distinct_labels: int  # distinct hint types used in the chain
    affect: float           # share of extension ('e') hints in [0, 1]
    is_word: bool
    is_enum: bool
    is_proof: bool
    has_twist: bool
    has_insight: bool
    has_figure: bool
    substance: float        # composite in [0, 1]


def extract_features(record: HintRecord, labels: list[str] | None = None) -> ProblemFeatures:
    labels = labels or []
    problem = record.problem or ""
    solution = record.solution or ""

    level = level_num(record.level)
    n_step_markers = len(_STEP_RE.findall(solution))
    n_sentences = len(re.findall(r"[.!?](?:\s|$)", solution))
    n_steps = max(n_step_markers, n_sentences)
    sol_len = len(solution)
    n_questions = len(record.questions)
    n_methods = len(_ALT_RE.findall(solution))
    n_distinct_labels = len(set(labels))
    affect = (labels.count("e") / len(labels)) if labels else 0.0

    is_word = bool(_WORD_RE.search(problem))
    is_enum = bool(_ENUM_RE.search(problem))
    is_proof = bool(_PROOF_RE.search(problem) or _PROOF_RE.search(solution))
    has_twist = bool(_TWIST_RE.search(solution))
    has_insight = bool(_INSIGHT_RE.search(solution))
    has_figure = bool(_ASY_RE.search(problem))

    # Composite substance in [0, 1]: difficulty dominates, then depth signals.
    level_norm = (level - 1) / 4.0
    steps_norm = min(n_steps / 10.0, 1.0)
    sollen_norm = min(sol_len / 1200.0, 1.0)
    q_norm = min(n_questions / 10.0, 1.0)
    substance = (
        0.45 * level_norm
        + 0.25 * steps_norm
        + 0.15 * sollen_norm
        + 0.15 * q_norm
    )

    return ProblemFeatures(
        level=level,
        n_steps=n_steps,
        sol_len=sol_len,
        n_questions=n_questions,
        n_methods=n_methods,
        n_distinct_labels=n_distinct_labels,
        affect=affect,
        is_word=is_word,
        is_enum=is_enum,
        is_proof=is_proof,
        has_twist=has_twist,
        has_insight=has_insight,
        has_figure=has_figure,
        substance=substance,
    )


# ---------------------------------------------------------------------------
# Scoring model
# ---------------------------------------------------------------------------

# Target substance level for each type (least -> most). A Gaussian preference
# around each target makes problems flow smoothly along the substance axis;
# signature bonuses below then pull specific problems to specific types.
_TARGET_SUBSTANCE: dict[str, float] = {
    "FN": 0.10,
    "AC": 0.22,
    "CH": 0.34,
    "ET": 0.42,
    "IM": 0.52,
    "IN": 0.58,
    "FP": 0.66,
    "SP": 0.86,
    "MP": 0.90,
}

_SIGMA = 0.20  # width of the substance preference


def _substance_pref(substance: float, target: float) -> float:
    return math.exp(-((substance - target) ** 2) / (2 * _SIGMA ** 2))


def score_types(f: ProblemFeatures) -> dict[str, float]:
    """Return a fit score per problem type (higher = better fit)."""
    S = f.substance
    scores = {t: _substance_pref(S, _TARGET_SUBSTANCE[t]) for t in PROBLEM_TYPES}

    single_concept = f.n_distinct_labels <= 2 and f.n_methods == 0 and not f.is_proof
    multi_idea = f.n_distinct_labels >= 3 and f.n_steps >= 4

    # First Notes -- atomic, one-step, algorithmic.
    if f.n_steps <= 2:
        scores["FN"] += 0.35
    if f.level <= 2:
        scores["FN"] += 0.15
    if f.is_word or f.n_methods or f.is_proof:
        scores["FN"] -= 0.30

    # Accidentals -- short, routine-looking, but a misconception twist. Rare by
    # design: only fires meaningfully when a twist is actually detected.
    if f.has_twist:
        scores["AC"] += 0.55
        if f.level <= 3 and f.sol_len < 500:
            scores["AC"] += 0.25
    else:
        scores["AC"] -= 0.20

    # Chords -- one idea, a handful of steps, more affective than cognitive.
    if 3 <= f.n_steps <= 6 and single_concept and not f.is_word:
        scores["CH"] += 0.30
    if f.affect > 0.0:
        scores["CH"] += 0.10

    # Etudes -- drilling one technique; word problems with a real-world hook.
    if f.is_word:
        scores["ET"] += 0.55
    if single_concept and 2 <= f.n_steps <= 7:
        scores["ET"] += 0.20

    # Improvisations -- accessible playground; enumerate / experiment.
    if f.is_enum and 2 <= f.level <= 4:
        scores["IM"] += 0.50
    if f.is_enum:
        scores["IM"] += 0.15

    # Interpretations -- modest substance, several qualitatively different paths.
    if f.n_methods >= 1:
        scores["IN"] += 0.45 + 0.25 * min(f.n_methods, 3)
        if not f.is_proof:
            scores["IN"] += 0.15

    # First Pieces -- multiple ideas + a macro plan + real choices.
    if multi_idea and f.n_methods == 0 and not f.is_proof:
        scores["FP"] += 0.35
    if f.n_questions >= 7 and multi_idea:
        scores["FP"] += 0.15

    # Showpieces -- very hard, cognitive, insight-driven, low affect.
    if f.has_insight and S >= 0.6:
        scores["SP"] += 0.35
    if f.has_figure and f.level >= 4:
        scores["SP"] += 0.20
    if S >= 0.6 and f.affect < 0.12:
        scores["SP"] += 0.15

    # Masterpieces -- very hard, generalizable, high affect.
    if f.is_proof and S >= 0.55:
        scores["MP"] += 0.40
    if f.is_proof and f.affect > 0.12:
        scores["MP"] += 0.25
    if f.sol_len >= 900 and S >= 0.6:
        scores["MP"] += 0.15

    return scores


def classify_problem(record: HintRecord, labels: list[str] | None = None) -> str:
    """Return the single best-fit Quarfoot problem-type code for a problem."""
    f = extract_features(record, labels)
    scores = score_types(f)
    # Tie-break by canonical order (earlier / lower-substance type wins).
    return max(scores, key=lambda t: (scores[t], -PROBLEM_TYPES.index(t)))
