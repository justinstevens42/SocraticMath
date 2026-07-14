"""Command-line entry point: classify -> analyze -> evaluate.

Usage (from anywhere, via uv):
    uv run socratic-hints classify      # label every chain, write files + CSV
    uv run socratic-hints analyze       # learn matrices, KL, similarity, plots
    uv run socratic-hints evaluate      # rubric vs Cursor's hand-labeled gold
    uv run socratic-hints all           # classify + analyze + evaluate
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from . import config
from .classify import classify_chain
from .classify_problem import classify_problem
from .gold import GOLD_LABELS
from .kl import (
    divergence_matrix,
    most_similar_pairs,
    similarity_matrix,
)
from .parse import iter_domain_records, load_hint_file
from .problem_types import PROBLEM_TYPE_NAMES, PROBLEM_TYPES
from .taxonomy import HINT_NAMES, HINT_TYPES
from .transitions import (
    learn_domain,
    pedagogical_prior,
    symmetric_prior,
)

PRIORS = {
    "symmetric": symmetric_prior,
    "pedagogical": pedagogical_prior,
}


# --------------------------------------------------------------------------
# classify
# --------------------------------------------------------------------------

def cmd_classify(args: argparse.Namespace) -> None:
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CLASSIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    per_domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ptype_counts: dict[str, int] = defaultdict(int)
    n_files = 0

    for domain_dir in config.domain_dirs():
        domain = config.domain_name(domain_dir)
        out_domain_dir = config.CLASSIFICATIONS_DIR / domain_dir.name
        out_domain_dir.mkdir(parents=True, exist_ok=True)

        for record in iter_domain_records(domain_dir, domain):
            labels = classify_chain(record.questions)
            problem_type = classify_problem(record, labels)
            problem_type_name = PROBLEM_TYPE_NAMES[problem_type]
            ptype_counts[problem_type] += 1
            n_files += 1
            classified = [
                {
                    "index": i + 1,
                    "question": q,
                    "label": lab,
                    "label_name": HINT_NAMES[lab],
                }
                for i, (q, lab) in enumerate(zip(record.questions, labels))
            ]
            # Per-file classification file (mirrors the source layout).
            out_path = out_domain_dir / f"{record.problem_id}.json"
            out_path.write_text(
                json.dumps(
                    {
                        "problem_id": record.problem_id,
                        "domain": domain,
                        "type": record.type,
                        "level": record.level,
                        "problem_type": problem_type,
                        "problem_type_name": problem_type_name,
                        "problem": record.problem,
                        "n_questions": len(record.questions),
                        "labels": labels,
                        "classified_questions": classified,
                    },
                    indent=2,
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            for c in classified:
                per_domain_counts[domain][c["label"]] += 1
                rows.append(
                    {
                        "domain": domain,
                        "problem_id": record.problem_id,
                        "type": record.type,
                        "level": record.level,
                        "problem_type": problem_type,
                        "problem_type_name": problem_type_name,
                        "n_questions": len(record.questions),
                        "q_index": c["index"],
                        "label": c["label"],
                        "label_name": c["label_name"],
                        "question": c["question"].replace("\n", " ").strip(),
                    }
                )

    fieldnames = [
        "domain", "problem_id", "type", "level",
        "problem_type", "problem_type_name", "n_questions",
        "q_index", "label", "label_name", "question",
    ]
    with config.CLASSIFIED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Classified {len(rows)} questions across {n_files} problems.")
    print(f"Per-file classifications: {config.CLASSIFICATIONS_DIR}")
    print(f"Master CSV: {config.CLASSIFIED_CSV}")
    print("\nLabel distribution by domain (f/r/b/e):")
    for domain in sorted(per_domain_counts):
        counts = per_domain_counts[domain]
        total = sum(counts.values())
        dist = "  ".join(
            f"{t}={counts.get(t, 0):>5} ({counts.get(t, 0) / total:5.1%})"
            for t in HINT_TYPES
        )
        print(f"  {domain:<28} n={total:>5}  {dist}")

    ptype_total = sum(ptype_counts.values()) or 1
    print("\nProblem-type distribution (Quarfoot taxonomy):")
    for t in PROBLEM_TYPES:
        n = ptype_counts.get(t, 0)
        print(
            f"  {t}  {PROBLEM_TYPE_NAMES[t]:<16} "
            f"n={n:>5} ({n / ptype_total:5.1%})"
        )


# --------------------------------------------------------------------------
# classify-llm
# --------------------------------------------------------------------------

def cmd_classify_llm(args: argparse.Namespace) -> None:
    """LLM-based counterpart of ``classify``: same outputs, ``*_llms`` dirs.

    Resumable: problems that already have a classification file are skipped,
    so re-running after an interruption only classifies what's missing.
    """
    from .classify_llm import classify_records_llm
    from .llm_backend import get_backend

    config.use_llm_paths()
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    config.CLASSIFICATIONS_DIR.mkdir(parents=True, exist_ok=True)

    pending: list = []
    n_existing = 0
    for domain_dir in config.domain_dirs():
        domain = config.domain_name(domain_dir)
        out_domain_dir = config.CLASSIFICATIONS_DIR / domain_dir.name
        out_domain_dir.mkdir(parents=True, exist_ok=True)
        for record in iter_domain_records(domain_dir, domain):
            if (out_domain_dir / f"{record.problem_id}.json").exists():
                n_existing += 1
            else:
                pending.append(record)

    limit = getattr(args, "limit", None)
    if limit:
        pending = pending[:limit]

    backend = get_backend(model=args.model) if args.model else get_backend()
    print(
        f"LLM classify: {len(pending)} problems to do "
        f"({n_existing} already classified), model={backend.model}, "
        f"backend={type(backend).__name__}"
    )

    def on_result(record, result) -> None:
        out_path = (
            config.CLASSIFICATIONS_DIR / f"hint_{record.domain}"
            / f"{record.problem_id}.json"
        )
        classified = [
            {
                "index": i + 1,
                "question": q,
                "label": lab,
                "label_name": HINT_NAMES[lab],
            }
            for i, (q, lab) in enumerate(zip(record.questions, result.labels))
        ]
        out_path.write_text(
            json.dumps(
                {
                    "problem_id": record.problem_id,
                    "domain": record.domain,
                    "type": record.type,
                    "level": record.level,
                    "problem_type": result.problem_type,
                    "problem_type_name": PROBLEM_TYPE_NAMES[result.problem_type],
                    "problem": record.problem,
                    "n_questions": len(record.questions),
                    "labels": result.labels,
                    "classified_questions": classified,
                    "classifier": "llm",
                },
                indent=2,
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    failed = classify_records_llm(backend, pending, on_result)
    if failed:
        print(f"\n{len(failed)} problems failed (re-run classify-llm to retry them):")
        for key in failed[:20]:
            print(f"  {key}")

    _write_llm_master_csv()


def _write_llm_master_csv() -> None:
    """Rebuild the master CSV from the per-file LLM classifications."""
    rows: list[dict] = []
    per_domain_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    ptype_counts: dict[str, int] = defaultdict(int)
    n_files = 0
    for domain_dir in sorted(config.CLASSIFICATIONS_DIR.glob("hint_*")):
        for path in sorted(domain_dir.glob("hint_*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            n_files += 1
            ptype_counts[data["problem_type"]] += 1
            for c in data["classified_questions"]:
                per_domain_counts[data["domain"]][c["label"]] += 1
                rows.append(
                    {
                        "domain": data["domain"],
                        "problem_id": data["problem_id"],
                        "type": data["type"],
                        "level": data["level"],
                        "problem_type": data["problem_type"],
                        "problem_type_name": data["problem_type_name"],
                        "n_questions": data["n_questions"],
                        "q_index": c["index"],
                        "label": c["label"],
                        "label_name": c["label_name"],
                        "question": c["question"].replace("\n", " ").strip(),
                    }
                )

    fieldnames = [
        "domain", "problem_id", "type", "level",
        "problem_type", "problem_type_name", "n_questions",
        "q_index", "label", "label_name", "question",
    ]
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with config.CLASSIFIED_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nClassified {len(rows)} questions across {n_files} problems (LLM).")
    print(f"Per-file classifications: {config.CLASSIFICATIONS_DIR}")
    print(f"Master CSV: {config.CLASSIFIED_CSV}")
    if per_domain_counts:
        print("\nLabel distribution by domain (f/r/b/e):")
        for domain in sorted(per_domain_counts):
            counts = per_domain_counts[domain]
            total = sum(counts.values())
            dist = "  ".join(
                f"{t}={counts.get(t, 0):>5} ({counts.get(t, 0) / total:5.1%})"
                for t in HINT_TYPES
            )
            print(f"  {domain:<28} n={total:>5}  {dist}")
    ptype_total = sum(ptype_counts.values()) or 1
    print("\nProblem-type distribution (Quarfoot taxonomy):")
    for t in PROBLEM_TYPES:
        n = ptype_counts.get(t, 0)
        print(f"  {t}  {PROBLEM_TYPE_NAMES[t]:<16} n={n:>5} ({n / ptype_total:5.1%})")


def cmd_evaluate_llm(args: argparse.Namespace) -> None:
    """Compare the stored LLM labels to the hand-labeled gold set."""
    config.use_llm_paths()
    total = 0
    correct = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    print("Evaluating LLM classifier against the hand-labeled gold set:\n")
    for (domain, pid), gold in GOLD_LABELS.items():
        path = config.CLASSIFICATIONS_DIR / f"hint_{domain}" / f"{pid}.json"
        if not path.exists():
            print(f"  [skip] {domain}/{pid}: no LLM classification file")
            continue
        pred = json.loads(path.read_text(encoding="utf-8"))["labels"]
        if len(pred) != len(gold):
            print(f"  [skip] {domain}/{pid}: length mismatch (gold={len(gold)}, pred={len(pred)})")
            continue
        agree = sum(1 for g, p in zip(gold, pred) if g == p)
        total += len(gold)
        correct += agree
        for g, p in zip(gold, pred):
            confusion[(g, p)] += 1
        print(f"  {domain}/{pid}: {agree}/{len(gold)}  gold={gold}  pred={pred}")

    if total:
        print(f"\nOverall per-question agreement: {correct}/{total} = {correct / total:.1%}")
        print("\nConfusion (gold -> pred):")
        header = "       " + "  ".join(f"{t:>4}" for t in HINT_TYPES)
        print(header)
        for g in HINT_TYPES:
            row = "  ".join(f"{confusion.get((g, p), 0):>4}" for p in HINT_TYPES)
            print(f"  {g} -> {row}")


# --------------------------------------------------------------------------
# analyze
# --------------------------------------------------------------------------

def _sequences_by(group_col: str) -> dict[str, list[list[str]]]:
    """Load hint-label chains from the master CSV, bucketed by ``group_col``.

    Each problem's questions are reassembled (in ``q_index`` order) into one
    label sequence, then chains are grouped by the requested column (e.g.
    ``"domain"`` or ``"problem_type"``).
    """
    if not config.CLASSIFIED_CSV.exists():
        raise SystemExit(
            f"Missing {config.CLASSIFIED_CSV}. Run 'socratic-hints classify' first."
        )
    grouped: dict[tuple[str, str], list[tuple[int, str]]] = defaultdict(list)
    group_of: dict[tuple[str, str], str] = {}
    with config.CLASSIFIED_CSV.open(encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if group_col not in (reader.fieldnames or []):
            raise SystemExit(
                f"Column '{group_col}' not in {config.CLASSIFIED_CSV}. "
                "Re-run 'socratic-hints classify' to regenerate it."
            )
        for r in reader:
            key = (r["domain"], r["problem_id"])
            grouped[key].append((int(r["q_index"]), r["label"]))
            group_of[key] = r[group_col]

    sequences: dict[str, list[list[str]]] = defaultdict(list)
    for key, items in grouped.items():
        items.sort(key=lambda t: t[0])
        sequences[group_of[key]].append([lab for _, lab in items])
    return sequences


def _load_sequences_from_csv() -> dict[str, list[list[str]]]:
    return _sequences_by("domain")


def _save_matrix_csv(path: Path, matrix: np.ndarray, labels: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for name, row in zip(labels, matrix):
            writer.writerow([name] + [f"{v:.6f}" for v in row])


def _plot_heatmap(path: Path, matrix: np.ndarray, labels_x, labels_y, title, fmt="{:.2f}", cmap="viridis"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(labels_x) + 2, 1.0 * len(labels_y) + 1.5))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(labels_x)))
    ax.set_xticklabels(labels_x, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels_y)))
    ax.set_yticklabels(labels_y, fontsize=8)
    ax.set_title(title, fontsize=10)
    vmid = (matrix.max() + matrix.min()) / 2 if matrix.size else 0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j, i, fmt.format(matrix[i, j]),
                ha="center", va="center", fontsize=7,
                color="white" if matrix[i, j] < vmid else "black",
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _run_grouped_analysis(
    sequences: dict[str, list[list[str]]],
    prior_strength: float,
    *,
    file_prefix: str,
    summary_key: str,
    summary_filename: str,
    group_noun: str,
    display_labels: dict[str, str] | None = None,
    label_order: list[str] | None = None,
) -> None:
    """Learn per-group transition matrices, then KL/similarity between groups.

    Reused for both the domain grouping (``file_prefix=""``) and the Quarfoot
    problem-type grouping (``file_prefix="types_"``). ``label_order`` fixes the
    axis/row order (e.g. least->most substance); otherwise groups are sorted.
    """
    config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    groups = [g for g in (label_order or sorted(sequences)) if g in sequences]
    disp = [(display_labels or {}).get(g, g) for g in groups]
    summary: dict = {summary_key: groups, "hint_types": HINT_TYPES, "priors": {}}

    for prior_name, prior_fn in PRIORS.items():
        prior = prior_fn(strength=prior_strength)
        posteriors = {g: learn_domain(g, sequences[g], prior) for g in groups}
        matrices = {g: posteriors[g].transition_matrix for g in groups}

        matrices_json: dict = {}
        for g in groups:
            _save_matrix_csv(
                config.OUTPUT_DIR / f"transition_{file_prefix}{prior_name}_{g}.csv",
                matrices[g], HINT_TYPES,
            )
            _plot_heatmap(
                config.OUTPUT_DIR / f"transition_{file_prefix}{prior_name}_{g}.png",
                matrices[g], HINT_TYPES, HINT_TYPES,
                f"{(display_labels or {}).get(g, g)}\ntransition matrix ({prior_name} prior)",
                fmt="{:.2f}", cmap="viridis",
            )
            matrices_json[g] = {
                "transition_matrix": matrices[g].tolist(),
                "counts": posteriors[g].counts.tolist(),
                "n_sequences": posteriors[g].n_sequences,
                "n_questions": posteriors[g].n_questions,
            }

        # KL divergence + similarity (symmetrized for the heatmap).
        div = divergence_matrix(groups, matrices, symmetric=True)
        div_dir = divergence_matrix(groups, matrices, symmetric=False)
        sim = similarity_matrix(div)
        _save_matrix_csv(config.OUTPUT_DIR / f"divergence_{file_prefix}{prior_name}.csv", div, groups)
        _save_matrix_csv(config.OUTPUT_DIR / f"divergence_directed_{file_prefix}{prior_name}.csv", div_dir, groups)
        _save_matrix_csv(config.OUTPUT_DIR / f"similarity_{file_prefix}{prior_name}.csv", sim, groups)
        _plot_heatmap(
            config.OUTPUT_DIR / f"divergence_{file_prefix}{prior_name}.png",
            div, disp, disp,
            f"Mean row-wise symmetric KL divergence ({prior_name} prior)\nlower = more similar",
            fmt="{:.3f}", cmap="magma_r",
        )
        _plot_heatmap(
            config.OUTPUT_DIR / f"similarity_{file_prefix}{prior_name}.png",
            sim, disp, disp,
            f"Similarity exp(-KL) ({prior_name} prior)\nhigher = more similar",
            fmt="{:.2f}", cmap="viridis",
        )

        pairs = most_similar_pairs(groups, div)
        summary["priors"][prior_name] = {
            "matrices": matrices_json,
            "divergence": div.tolist(),
            "similarity": sim.tolist(),
            "most_similar": pairs[:3],
            "most_different": pairs[-3:][::-1],
        }

        print(f"\n=== Prior: {prior_name} (strength={prior_strength}) ===")
        print(f"Most similar {group_noun} pairs (lowest KL):")
        for a, b, v in pairs[:3]:
            print(f"  {a:<26} <-> {b:<26} KL={v:.4f}")
        print(f"Most different {group_noun} pairs (highest KL):")
        for a, b, v in pairs[-3:][::-1]:
            print(f"  {a:<26} <-> {b:<26} KL={v:.4f}")

    (config.OUTPUT_DIR / summary_filename).write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(f"\nWrote matrices, divergence/similarity CSVs, heatmaps to {config.OUTPUT_DIR}")


def cmd_analyze(args: argparse.Namespace) -> None:
    sequences = _sequences_by("domain")
    _run_grouped_analysis(
        sequences,
        args.prior_strength,
        file_prefix="",
        summary_key="domains",
        summary_filename="analysis_summary.json",
        group_noun="domain",
    )


def cmd_analyze_types(args: argparse.Namespace) -> None:
    sequences = _sequences_by("problem_type")
    _run_grouped_analysis(
        sequences,
        args.prior_strength,
        file_prefix="types_",
        summary_key="problem_types",
        summary_filename="problem_type_analysis_summary.json",
        group_noun="problem-type",
        display_labels=PROBLEM_TYPE_NAMES,
        label_order=PROBLEM_TYPES,
    )


# --------------------------------------------------------------------------
# problem-type histograms
# --------------------------------------------------------------------------

def cmd_plot_type_hist(args: argparse.Namespace) -> None:
    """Histogram of Quarfoot problem types per domain, plus an overall one.

    Writes ``problem_type_hist_<domain>.png`` for every domain,
    ``problem_type_hist_overall.png``, a combined small-multiples figure, and
    ``problem_type_distribution.csv`` (domains x types, counts).
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not config.CLASSIFIED_CSV.exists():
        raise SystemExit(f"Missing {config.CLASSIFIED_CSV}. Run classify first.")

    # One row per problem (the CSV has one row per question).
    ptype_of: dict[tuple[str, str], str] = {}
    with config.CLASSIFIED_CSV.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ptype_of[(r["domain"], r["problem_id"])] = r["problem_type"]

    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    overall: dict[str, int] = defaultdict(int)
    for (domain, _pid), t in ptype_of.items():
        counts[domain][t] += 1
        overall[t] += 1
    domains = sorted(counts)

    with (config.OUTPUT_DIR / "problem_type_distribution.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.writer(f)
        writer.writerow(["domain"] + PROBLEM_TYPES + ["total"])
        for d in domains + ["overall"]:
            row = overall if d == "overall" else counts[d]
            writer.writerow([d] + [row.get(t, 0) for t in PROBLEM_TYPES]
                            + [sum(row.values())])

    BAR = "#2a78d6"  # single series -> one hue; identity is carried by the axis
    x = range(len(PROBLEM_TYPES))

    def draw(ax, dist: dict[str, int], title: str, ymax: float) -> None:
        total = sum(dist.values()) or 1
        pct = [100.0 * dist.get(t, 0) / total for t in PROBLEM_TYPES]
        ax.bar(x, pct, width=0.62, color=BAR, zorder=3)
        for xi, (p, t) in enumerate(zip(pct, PROBLEM_TYPES)):
            ax.text(xi, p + ymax * 0.015, str(dist.get(t, 0)),
                    ha="center", va="bottom", fontsize=7, color="#3d4351")
        ax.set_title(f"{title} (n={total})", fontsize=10)
        ax.set_xticks(list(x))
        ax.set_xticklabels(PROBLEM_TYPES, fontsize=8)
        ax.set_ylim(0, ymax)
        ax.set_ylabel("% of problems", fontsize=8)
        ax.tick_params(labelsize=8)
        ax.grid(axis="y", color="#e3e5ea", linewidth=0.8, zorder=0)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)

    all_dists = [counts[d] for d in domains] + [overall]
    ymax = 1.15 * max(
        100.0 * v / (sum(dist.values()) or 1)
        for dist in all_dists for v in dist.values()
    )

    for d in domains:
        fig, ax = plt.subplots(figsize=(6.4, 3.6))
        draw(ax, counts[d], d, ymax)
        fig.tight_layout()
        fig.savefig(config.OUTPUT_DIR / f"problem_type_hist_{d}.png", dpi=150)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    draw(ax, overall, "all domains", ymax)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "problem_type_hist_overall.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharey=True)
    for ax, (title, dist) in zip(
        axes.flat, [(d, counts[d]) for d in domains] + [("all domains", overall)]
    ):
        draw(ax, dist, title, ymax)
    fig.suptitle("Quarfoot problem-type distribution by domain", fontsize=12)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "problem_type_hist_combined.png", dpi=150)
    plt.close(fig)

    n_pngs = len(domains) + 2
    n_pngs += _plot_type_comparison(plt, overall)

    print(f"Wrote {n_pngs} histogram PNGs and "
          f"problem_type_distribution.csv to {config.OUTPUT_DIR}")


def _plot_type_comparison(plt, this_overall: dict[str, int]) -> int:
    """Grouped bars: this pipeline's type counts vs the other pipeline's.

    Compares the LLM run against the regex run (or vice versa) when the other
    master CSV exists. Returns the number of PNGs written (0 or 1).
    """
    is_llm = config.CLASSIFIED_CSV == config.CLASSIFIED_LLM_CSV
    other_csv = (
        config.CODE_DIR / "outputs" / "classified_hints.csv" if is_llm
        else config.CLASSIFIED_LLM_CSV
    )
    if not other_csv.exists():
        return 0

    ptype_of: dict[tuple[str, str], str] = {}
    with other_csv.open(encoding="utf-8") as f:
        for r in csv.DictReader(f):
            ptype_of[(r["domain"], r["problem_id"])] = r["problem_type"]
    other_overall: dict[str, int] = defaultdict(int)
    for t in ptype_of.values():
        other_overall[t] += 1

    llm_counts = this_overall if is_llm else other_overall
    rx_counts = other_overall if is_llm else this_overall
    n_llm, n_rx = sum(llm_counts.values()), sum(rx_counts.values())

    # Two-series categorical pair (palette slots 1 & 3; validator-passing.
    # The yellow's contrast WARN is relieved by the direct value labels).
    C_LLM, C_RX = "#2a78d6", "#eda100"
    x = list(range(len(PROBLEM_TYPES)))
    w = 0.36

    fig, ax = plt.subplots(figsize=(12, 5.5))
    llm_vals = [llm_counts.get(t, 0) for t in PROBLEM_TYPES]
    rx_vals = [rx_counts.get(t, 0) for t in PROBLEM_TYPES]
    ax.bar([xi - w / 2 for xi in x], llm_vals,
           width=w - 0.02, color=C_LLM, label=f"Claude (n = {n_llm})", zorder=3)
    ax.bar([xi + w / 2 for xi in x], rx_vals,
           width=w - 0.02, color=C_RX, label=f"RegEx (n = {n_rx})", zorder=3)
    ymax = max(llm_vals + rx_vals)
    for xi, (lv, rv) in enumerate(zip(llm_vals, rx_vals)):
        ax.text(xi - w / 2, lv + ymax * 0.01, str(lv), ha="center",
                va="bottom", fontsize=8, color="#3d4351")
        ax.text(xi + w / 2, rv + ymax * 0.01, str(rv), ha="center",
                va="bottom", fontsize=8, color="#3d4351")
    ax.set_xticks(x)
    ax.set_xticklabels(
        [f"{t}\n{PROBLEM_TYPE_NAMES[t]}" for t in PROBLEM_TYPES], fontsize=9
    )
    ax.set_xlabel("Problem type (least → most substance)", fontsize=10)
    ax.set_ylabel("Number of problems", fontsize=10)
    ax.set_title("Claude vs RegEx problem-type counts — all domains", fontsize=12)
    ax.legend(frameon=False, fontsize=10)
    ax.grid(axis="y", color="#e3e5ea", linewidth=0.8, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(config.OUTPUT_DIR / "problem_type_hist_llm_vs_regex.png", dpi=150)
    plt.close(fig)
    return 1


# --------------------------------------------------------------------------
# evaluate
# --------------------------------------------------------------------------

def cmd_evaluate(args: argparse.Namespace) -> None:
    total = 0
    correct = 0
    confusion: dict[tuple[str, str], int] = defaultdict(int)
    print("Evaluating rubric classifier against Cursor's hand-labeled gold set:\n")
    for (domain, pid), gold in GOLD_LABELS.items():
        path = config.REPO_ROOT / f"hint_{domain}" / f"{pid}.json"
        record = load_hint_file(path, domain)
        if record is None:
            print(f"  [skip] {domain}/{pid}: file not found or unparseable")
            continue
        pred = classify_chain(record.questions)
        if len(pred) != len(gold):
            print(f"  [skip] {domain}/{pid}: length mismatch (gold={len(gold)}, parsed={len(pred)})")
            continue
        agree = sum(1 for g, p in zip(gold, pred) if g == p)
        total += len(gold)
        correct += agree
        for g, p in zip(gold, pred):
            confusion[(g, p)] += 1
        print(f"  {domain}/{pid}: {agree}/{len(gold)}  gold={gold}  pred={pred}")

    if total:
        print(f"\nOverall per-question agreement: {correct}/{total} = {correct / total:.1%}")
        print("\nConfusion (gold -> pred):")
        header = "       " + "  ".join(f"{t:>4}" for t in HINT_TYPES)
        print(header)
        for g in HINT_TYPES:
            row = "  ".join(f"{confusion.get((g, p), 0):>4}" for p in HINT_TYPES)
            print(f"  {g} -> {row}")


# --------------------------------------------------------------------------
# entrypoint
# --------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Socratic hint-type transition analysis.")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("classify", help="Classify all hint chains; write files + CSV.")

    p_an = sub.add_parser("analyze", help="Learn per-domain transition matrices, KL, similarity.")
    p_an.add_argument("--prior-strength", type=float, default=1.0,
                      help="Scales both Dirichlet priors (default 1.0).")

    p_ty = sub.add_parser(
        "analyze-types",
        help="Group by Quarfoot problem type; KL divergence between types.",
    )
    p_ty.add_argument("--prior-strength", type=float, default=1.0,
                      help="Scales both Dirichlet priors (default 1.0).")

    sub.add_parser("evaluate", help="Compare rubric to the hand-labeled gold set.")

    p_all = sub.add_parser("all", help="Run classify + analyze + analyze-types + evaluate.")
    p_all.add_argument("--prior-strength", type=float, default=1.0)

    # ---- LLM variants (write to classifications_llms / outputs_llms) ----
    p_cl = sub.add_parser(
        "classify-llm",
        help="Classify hint chains + Quarfoot problem types with an LLM.",
    )
    p_cl.add_argument("--model", default=None,
                      help="Model id (default: claude-opus-4-8 or $SOCRATIC_LLM_MODEL).")
    p_cl.add_argument("--limit", type=int, default=None,
                      help="Only classify the first N pending problems (smoke test).")

    p_anl = sub.add_parser("analyze-llm", help="Analyze the LLM classifications.")
    p_anl.add_argument("--prior-strength", type=float, default=1.0)

    p_tyl = sub.add_parser("analyze-types-llm",
                           help="Problem-type analysis on the LLM classifications.")
    p_tyl.add_argument("--prior-strength", type=float, default=1.0)

    sub.add_parser("evaluate-llm", help="Compare LLM labels to the gold set.")

    sub.add_parser("plot-types", help="Problem-type histograms (regex labels).")
    sub.add_parser("plot-types-llm", help="Problem-type histograms (LLM labels).")

    p_alll = sub.add_parser(
        "all-llm", help="Run classify-llm + analyze-llm + analyze-types-llm + evaluate-llm."
    )
    p_alll.add_argument("--prior-strength", type=float, default=1.0)
    p_alll.add_argument("--model", default=None)
    p_alll.add_argument("--limit", type=int, default=None)

    args = parser.parse_args(argv)

    if args.command == "classify":
        cmd_classify(args)
    elif args.command == "analyze":
        cmd_analyze(args)
    elif args.command == "analyze-types":
        cmd_analyze_types(args)
    elif args.command == "evaluate":
        cmd_evaluate(args)
    elif args.command == "all":
        cmd_classify(args)
        cmd_analyze(args)
        cmd_analyze_types(args)
        cmd_evaluate(args)
    elif args.command == "classify-llm":
        cmd_classify_llm(args)
    elif args.command == "analyze-llm":
        config.use_llm_paths()
        cmd_analyze(args)
    elif args.command == "analyze-types-llm":
        config.use_llm_paths()
        cmd_analyze_types(args)
    elif args.command == "evaluate-llm":
        cmd_evaluate_llm(args)
    elif args.command == "plot-types":
        cmd_plot_type_hist(args)
    elif args.command == "plot-types-llm":
        config.use_llm_paths()
        cmd_plot_type_hist(args)
    elif args.command == "all-llm":
        cmd_classify_llm(args)
        cmd_analyze(args)
        cmd_analyze_types(args)
        cmd_evaluate_llm(args)


if __name__ == "__main__":
    main()
