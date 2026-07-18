# TMATH

## Description
This repository hosts the Socratic Math Hints Dataset, a curated collection of hints for various mathematical problems. The hints are presented in a Socratic dialogue format, promoting an inquisitive learning approach. It also contains an analysis package (`code/`) that classifies the hints into pedagogical types and studies how hint types transition within a problem.

<img width="1299" alt="dataset" src="https://github.com/user-attachments/assets/e0c9fc4e-b9df-4264-a7a1-a10e8e6ff9f6">

## Repository structure

```
hint_<domain>/      the dataset: one JSON file per problem, per math domain
code/               the analysis package (see code/README.md for full details)
```

The dataset is organized into directories, each corresponding to a specific area of mathematics:

- `hint_algebra/` - Hints for algebra problems
- `hint_counting_and_probability/` - Hints for counting and probability problems
- `hint_geometry/` - Hints for geometry problems
- `hint_intermediate_algebra/` - Hints for intermediate algebra problems
- `hint_number_theory/` - Hints for number theory problems
- `hint_prealgebra/` - Hints for pre-algebra problems
- `hint_precalculus/` - Hints for precalculus problems

## Using the dataset

Each `hint_<n>.json` file describes one problem and its chain of Socratic hints:

| field | contents |
|-------|----------|
| `problem` | the problem statement (LaTeX) |
| `level` | difficulty, e.g. `"Level 5"` |
| `type` | math domain, e.g. `"Algebra"` |
| `solution` | a worked solution (LaTeX) |
| `socratic_questions` | the numbered chain of Socratic hint questions |

Example (Python):

```python
import json
from pathlib import Path

for path in Path("hint_algebra").glob("hint_*.json"):
    hint = json.loads(path.read_text())
    print(hint["level"], hint["problem"][:60])
```

## Running the analysis

The `code/` package parses every hint chain, classifies each question into one of four pedagogical hint types (and each problem into one of nine problem types), fits Bayesian transition matrices, and compares domains via KL divergence. It uses [`uv`](https://docs.astral.sh/uv/) for dependency management.

```bash
cd code
uv sync
uv run socratic-hints all        # regex-rubric pipeline -> code/outputs/
uv run socratic-hints all-llm    # Claude-based pipeline -> code/outputs_llms/  (needs ANTHROPIC_API_KEY)
```

See [`code/README.md`](code/README.md) for the model, the individual commands, and a description of every output file.
