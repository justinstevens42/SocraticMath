# Socratic Hint-Type Transition Analysis

Implementation of the proposal *"Analyzing Mathematical Hint Probabilities using
Bayesian Machine Learning"* (Justin Stevens).

Given the repository's chains of Socratic questions (`hint_<domain>/hint_<n>.json`),
this package:

1. **Parses** the `socratic_questions` field of every hint file into individual
   questions.
2. **Classifies** each question into one of four pedagogical hint types using a
   transparent rubric authored from the proposal's labeling guidance (with an
   optional LLM backend).
3. **Learns** a 4×4 hint-type transition matrix per domain with a
   **Dirichlet–Multinomial** Bayesian model, under two priors.
4. **Compares** domains via **row-wise KL divergence**, producing a
   divergence/similarity matrix and heatmaps.

## Hint types

| code | type | description |
|------|------|-------------|
| `f` | feature-pointing | draws attention to a given/feature of *this* problem |
| `r` | principle-stating | states a general rule, definition, theorem, or formula |
| `b` | bottom-out | a concrete solving step on *this* problem |
| `e` | extension | pushes thinking beyond the current problem |

The transition matrix has the form from the proposal (rows = "from", cols = "to",
order `f, r, b, e`):

```
        ->f   ->r   ->b   ->e
from f [ p_ff p_fr p_fb p_fe ]
from r [ p_rf p_rr p_rb p_re ]
from b [ p_bf p_br p_bb p_be ]
from e [ p_ef p_er p_eb p_ee ]
```

## Setup

Uses [`uv`](https://docs.astral.sh/uv/).

```bash
cd code
uv sync
```

## Usage

```bash
uv run socratic-hints classify     # parse + classify -> per-file JSON + master CSV
uv run socratic-hints analyze      # transition matrices + KL + similarity + plots
uv run socratic-hints evaluate     # rubric vs Cursor's hand-labeled gold set
uv run socratic-hints all          # all of the above
uv run socratic-hints analyze --prior-strength 2.0   # scale the Dirichlet priors
```

## Bayesian model

Each row of a domain's transition matrix is a categorical distribution over the
next hint type. With a Dirichlet prior `alpha` per row and observed multinomial
transition counts `n`, the posterior over each row is Dirichlet, and the reported
matrix is the posterior mean:

```
M[i, j] = (alpha[i, j] + n[i, j]) / sum_k (alpha[i, k] + n[i, k])
```

Two priors are provided (see `transitions.py`):

- **symmetric** — uniform pseudo-counts (every transition equally likely a priori).
- **pedagogical** — asymmetric, favoring the expected Socratic arc
  (`f → r`, `b → b`, `b → e`) and down-weighting implausible jumps (`f → e`).

## Evaluation

`evaluate` compares the automatic rubric against a small gold set hand-labeled by
Cursor (`gold.py`), including the proposal's worked example (`algebra/hint_0`,
which uses the paper's published human labels). It reports per-question agreement
and a confusion matrix.

## Outputs (`code/outputs/`)

- `classified_hints.csv` — every question with its domain, problem id, and label.
- `transition_<prior>_<domain>.csv` / `.png` — per-domain transition matrices.
- `divergence_<prior>.csv` / `.png` — mean row-wise symmetric KL between domains.
- `divergence_directed_<prior>.csv` — directed (asymmetric) KL.
- `similarity_<prior>.csv` / `.png` — `exp(-KL)` similarity heatmap.
- `analysis_summary.json` — machine-readable summary (matrices, KL, top pairs).

Per-file classifications are written to `code/classifications/hint_<domain>/`.

## Layout

```
code/
  src/socratic_hints/
    taxonomy.py     # the 4 hint types, rubric, LLM system prompt
    parse.py        # load hint JSON, split the question chain
    classify.py     # rubric classifier (+ optional LLM hook)
    gold.py         # Cursor's hand-labeled gold set
    transitions.py  # Dirichlet-Multinomial transition matrices
    kl.py           # row-wise KL divergence + similarity
    cli.py          # classify / analyze / evaluate / all
```
