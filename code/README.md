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
5. **Classifies** each whole problem into one of Quarfoot's nine **pedagogical
   problem types** (a heuristic, no LLM required) and produces the analogous
   KL divergence matrix *between problem types* instead of subjects.
6. **Tests** whether each domain's transition structure differs
   *significantly* from the other domains, using a problem-level
   permutation test (`significance.py`).

## Hint types

| code | type | description |
|------|------|-------------|
| `f` | feature-pointing | draws attention to a given/feature of *this* problem |
| `r` | principle-stating | states a general rule, definition, theorem, or formula |
| `b` | bottom-out | a concrete solving step on *this* problem |
| `e` | extension | pushes thinking beyond the current problem |

## Problem types (Quarfoot taxonomy)

Where the four *hint* types classify individual Socratic questions, these nine
*problem* types classify whole problems by pedagogical character. Each problem
is assigned exactly one type by a transparent heuristic (`classify_problem.py`)
that scores interpretable features — difficulty `level`, worked-solution length
and step count, chain length, alternative-approach markers, word-problem /
enumeration / proof / "gotcha" / insight language, and the extension-hint share
— along Quarfoot's axes (**substance**, **cognitive vs. affective**,
**solution-space openness**). Ordered least → most substance:

| code | type | one-line character |
|------|------|--------------------|
| `FN` | First Notes | one-step, algorithmic fluency drills |
| `AC` | Accidentals | routine-looking but hide a misconception twist |
| `CH` | Chords | one idea, several steps, more affective |
| `ET` | Etudes | drill one technique; word problems with a real-world hook |
| `IM` | Improvisations | accessible playground; enumerate/experiment |
| `IN` | Interpretations | modest substance, several qualitatively different paths |
| `FP` | First Pieces | multiple ideas + a macro plan + real choices |
| `SP` | Showpieces | very hard, cognitive, insight-driven, contrived |
| `MP` | Masterpieces | very hard, generalizable, high affect |

These labels are heuristic proxies of a qualitative taxonomy, not ground truth,
but they are reproducible and fully inspectable. `analyze-types` then learns a
4×4 hint-type transition matrix **per problem type** and computes the row-wise
KL divergence matrix between the nine types.

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
uv run socratic-hints classify       # parse + classify (hint types + problem type) -> per-file JSON + master CSV
uv run socratic-hints analyze        # per-domain transition matrices + KL + similarity + plots
uv run socratic-hints analyze-types  # per-problem-type transition matrices + KL + similarity + plots
uv run socratic-hints test           # one-vs-rest permutation significance tests
uv run socratic-hints evaluate       # rubric vs Cursor's hand-labeled gold set
uv run socratic-hints all            # all of the above
uv run socratic-hints analyze --prior-strength 2.0   # scale the Dirichlet priors
uv run socratic-hints test --target geometry --n-perm 20000 --seed 1
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

## Significance testing

KL divergence says *how far apart* two domains' transition matrices are, but
not whether the gap is larger than chance. `test` answers that with a
**problem-level permutation test**: for each domain vs. all other domains
pooled, whole problems (hint chains) are randomly reassigned between the two
groups 10,000 times and the observed statistics are compared against the
resulting null distribution. Resampling whole chains — rather than individual
transitions — respects the correlation between transitions within a chain,
which a plain chi-square over pooled counts would ignore.

Two test statistics are computed per comparison (see `significance.py`):

- **G** — the likelihood-ratio statistic for homogeneity of the two groups'
  4×4 transition-count tables, also broken down per "from" row to localize
  *which* transitions differ.
- **symKL** — the repository's own comparison metric: symmetric mean row-wise
  KL between the two groups' posterior-mean matrices (symmetric prior).

p-values are permutation p-values, `(1 + #{perm ≥ obs}) / (1 + n_perm)`, so
the smallest reportable value at the default `--n-perm 10000` is ~1e-4;
Holm-adjusted versions control the family-wise error rate across the seven
one-vs-rest tests. `--target` (default `counting_and_probability`) selects
the domain that gets the detailed per-row report. `test-llm` runs the same
tests on the LLM classifications (reads and writes under `outputs_llms/`).

## Evaluation

`evaluate` compares the automatic rubric against a small gold set hand-labeled by
Cursor (`gold.py`), including the proposal's worked example (`algebra/hint_0`,
which uses the paper's published human labels). It reports per-question agreement
and a confusion matrix.

## Outputs (`code/outputs/`)

- `classified_hints.csv` — every question with its domain, problem id, hint
  label, and its problem's Quarfoot `problem_type`.
- `transition_<prior>_<domain>.csv` / `.png` — per-domain transition matrices.
- `divergence_<prior>.csv` / `.png` — mean row-wise symmetric KL between domains.
- `divergence_directed_<prior>.csv` — directed (asymmetric) KL.
- `similarity_<prior>.csv` / `.png` — `exp(-KL)` similarity heatmap.
- `analysis_summary.json` — machine-readable summary (matrices, KL, top pairs).
- `significance_one_vs_rest.csv` — per-domain one-vs-rest permutation test
  results (G, symKL, raw + Holm-adjusted p-values).
- `significance_summary.json` — the full test results, including the per-row
  G breakdown and both groups' transition-count tables per domain.

Problem-type outputs (`analyze-types`) mirror the above with a `types_` infix
and problem-type codes in place of domains:

- `transition_types_<prior>_<TYPE>.csv` / `.png` — per-problem-type matrices.
- `divergence_types_<prior>.csv` / `.png` — row-wise symmetric KL **between the
  nine problem types**.
- `divergence_directed_types_<prior>.csv`, `similarity_types_<prior>.csv` / `.png`.
- `problem_type_analysis_summary.json` — machine-readable summary.

Per-file classifications (now including `problem_type` / `problem_type_name`)
are written to `code/classifications/hint_<domain>/`.

## Layout

```
code/
  src/socratic_hints/
    taxonomy.py         # the 4 hint types, rubric, LLM system prompt
    problem_types.py    # the 9 Quarfoot problem types + descriptions
    parse.py            # load hint JSON, split the question chain
    classify.py         # rubric hint-type classifier (+ optional LLM hook)
    classify_problem.py # heuristic whole-problem type classifier
    gold.py             # Cursor's hand-labeled gold set
    transitions.py      # Dirichlet-Multinomial transition matrices
    kl.py               # row-wise KL divergence + similarity
    significance.py     # problem-level one-vs-rest permutation tests
    cli.py              # classify / analyze / analyze-types / test / evaluate / all
```
