"""Classify Socratic questions into the four hint types.

Two backends are provided:

* :func:`classify_chain` -- a transparent, deterministic rubric that encodes the
  same labeling guidance given to Cursor (keyword/structural signals + a weak
  positional prior). This is the default so the whole pipeline is reproducible
  offline with no API keys.
* :func:`classify_chain_llm` -- an optional hook that sends the chain to an LLM
  using the strict system prompt from :mod:`taxonomy`. Wire up your own client.

The rubric was authored by reading the dataset and the proposal's worked
example; a hand-labeled gold set in :mod:`gold` is used to measure agreement.
"""

from __future__ import annotations

import re

from .taxonomy import HINT_TYPES

# --- Keyword / phrase signals per hint type -------------------------------
# Each entry is (compiled regex, weight). Higher weight = stronger evidence.

_EXTENSION = [
    (r"real[\s-]?life|real[\s-]?world|everyday life", 3.0),
    (r"applications?\b|applied in|apply this|used in (real|other|everyday|the real)", 2.0),
    (r"other examples?|another example|other problems?|similar problems?", 3.0),
    (r"other (situations?|scenarios?|contexts?|cases?|methods?|ways?|approaches?)", 1.5),
    (r"are there (any )?other|any other (possible )?(value|solution|method|way|case|approach)"
     r"|other possible (value|solution|way)", 1.8),
    (r"think of (any |some )?other|can you think of", 2.5),
    (r"significance of|importance of (knowing|understanding|learning|this concept|these)", 1.8),
    (r"beyond (this|the) problem|in general life|generalize|generalization", 2.0),
    (r"connect(ed|ion)? to|relate(s|d)? to (other|real)", 1.5),
    (r"history|invented|discovered", 1.5),
]

_PRINCIPLE = [
    (r"\bgeneral formula\b|formula for (expanding|finding|the area|the volume|a |an )", 2.5),
    (r"\bdefinition\b|\bdefine\b|what does .* mean", 2.0),
    (r"concept of|the concept|explain the (concept|idea|principle)", 2.5),
    (r"\btheorem\b|\bthe rule\b|\bproperty of\b|\bproperties of\b|\bthe principle\b", 2.5),
    (r"what is a\b|what are\b(?!.*values)|what is an\b", 1.0),
    (r"in general\b|generally,", 1.5),
    (r"recall (the|that|what)|remember (the|that)", 1.5),
    (r"relationship between|how (is|are) .* related|how does .* relate", 1.5),
    (r"explain what .* is\b", 1.2),
    (r"what conditions|conditions (are |that are )?(necessary|needed|required|must)"
     r"|conditions (need|needs) to be (satisfied|met)", 1.8),
    (r"why (do|does|is|are) (we|you|it)|why is it important", 1.5),
]

_BOTTOM_OUT = [
    (r"\bsolve for\b|how (would|do|can|will) (you|we) solve|solve the", 3.0),
    (r"\bcalculate\b|\bcompute\b|\bevaluate\b", 2.5),
    (r"how (can|do|would|will) .* (be )?(rewritten|rewrite|simplif|express)", 2.5),
    (r"\bsimplif(y|ied|ies)\b|can you simplify", 2.5),
    (r"express(ed)? .* in terms of|write .* in the form", 2.0),
    (r"\bsubstitut|\bplug in|\bplug the", 2.5),
    (r"(values?|value) of\b.*\?|find the values? of|determine the values? of"
     r"|identify the values? of", 2.5),
    (r"set .* equal|setting .* equal", 2.0),
    (r"\bfinal (value|answer|result|step)\b|what is the answer", 2.5),
    (r"how (many|much)\b|what is the (value|result|sum|product|total|number) of", 1.8),
    (r"(values?|value) (for|of)\b", 1.5),
    (r"how (did|would|do|can|will) (you|we) (find|determine|approach|get|obtain|arrive|rewrite|derive|solve|calculate|compute)", 1.8),
    (r"explain how (you|we|to) (rewr|solv|determin|f[io]nd|found|obtain|calculat|simplif|deriv|comput|arriv|set)", 2.0),
    (r"\bnext step\b|what is the next|proceed to", 2.0),
    (r"add|subtract|multiply|divide|factor(ize)?|expand the", 1.2),
]

_FEATURE = [
    (r"\bnotice\b|\bobserve\b|do you (notice|see|observe)", 2.5),
    (r"identify (any |the )?(pattern|feature|given|key|relevant|important)", 2.5),
    (r"in (this|the given) problem|in the (problem|figure|diagram|expression|equation)", 1.5),
    (r"what (is|are) (the )?(given|provided)\b", 2.0),
    (r"what information|what (do|are) (we|you) (given|told)|what is being asked", 2.0),
    (r"what (is|are) (we|you) (asked|trying) to|the (goal|objective|question is)", 1.5),
    (r"key (feature|information|detail|element|component)s?", 2.0),
    (r"what (is|are) the (given expression|expression equal)", 1.5),
    (r"can you identify\b", 1.5),
]

_SIGNALS = {
    "e": [(re.compile(p, re.I), w) for p, w in _EXTENSION],
    "r": [(re.compile(p, re.I), w) for p, w in _PRINCIPLE],
    "b": [(re.compile(p, re.I), w) for p, w in _BOTTOM_OUT],
    "f": [(re.compile(p, re.I), w) for p, w in _FEATURE],
}


def _keyword_scores(text: str) -> dict[str, float]:
    scores = {t: 0.0 for t in HINT_TYPES}
    for label, signals in _SIGNALS.items():
        for pattern, weight in signals:
            if pattern.search(text):
                scores[label] += weight
    return scores


def _positional_prior(index: int, total: int) -> dict[str, float]:
    """Weak prior reflecting the typical f/r -> b -> e arc of a chain."""
    if total <= 1:
        p = 0.0
    else:
        p = index / (total - 1)  # 0.0 (first) .. 1.0 (last)
    return {
        "f": max(0.0, 1.0 - 1.8 * p),          # front-loaded
        "r": max(0.0, 0.8 - abs(p - 0.25) * 1.6),  # early-middle
        "b": max(0.0, 1.0 - abs(p - 0.6) * 2.2),   # middle
        # Extension is reliably keyword-identifiable; a positional prior here
        # tends to mislabel keyword-free tail questions, so we keep it at 0.
        "e": 0.0,
    }


# Keyword evidence dominates; the positional prior only breaks ties.
_PRIOR_WEIGHT = 0.6


def classify_question(text: str, index: int, total: int) -> str:
    kw = _keyword_scores(text)
    prior = _positional_prior(index, total)
    combined = {t: kw[t] + _PRIOR_WEIGHT * prior[t] for t in HINT_TYPES}
    best = max(combined, key=lambda t: (combined[t], -HINT_TYPES.index(t)))
    return best


def classify_chain(questions: list[str]) -> list[str]:
    """Label every question in a chain with the rubric backend."""
    total = len(questions)
    return [classify_question(q, i, total) for i, q in enumerate(questions)]


# --- Optional LLM backend --------------------------------------------------

def classify_chain_llm(problem: str, questions: list[str], client=None) -> list[str]:
    """Label a chain with an LLM using the strict system prompt.

    ``client`` must expose ``complete(system: str, user: str) -> str`` returning
    a JSON array of labels. Left unwired by default; the rubric backend is used
    for the runnable pipeline.
    """
    import json

    from .taxonomy import build_system_prompt

    if client is None:
        raise RuntimeError(
            "No LLM client provided. Use classify_chain() for the offline rubric, "
            "or pass a client implementing complete(system, user) -> str."
        )
    system = build_system_prompt()
    numbered = "\n".join(f"{i}. {q}" for i, q in enumerate(questions, start=1))
    user = f"Problem: {problem}\n\nChain:\n{numbered}"
    raw = client.complete(system, user)
    labels = json.loads(raw)
    if len(labels) != len(questions) or any(l not in HINT_TYPES for l in labels):
        raise ValueError(f"LLM returned invalid labels: {raw!r}")
    return labels
