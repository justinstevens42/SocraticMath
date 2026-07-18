"""Problem-level permutation tests for transition-matrix differences.

Answers "is this domain's hint-type transition structure *significantly*
different from the rest?" The unit of resampling is the problem (a whole
hint chain), so the test respects the correlation between transitions within
a chain — a plain chi-square over the pooled transitions would treat them as
independent and overstate significance.

For each group (domain) vs. the complement (all other domains pooled), two
test statistics are computed on the 4x4 transition-count tables:

* ``G`` — the likelihood-ratio statistic for homogeneity of the two groups'
  tables (a count-weighted analogue of KL divergence). Also reported per
  "from" row, to localize which transitions drive a difference.
* ``symKL`` — the repository's own comparison metric: symmetric mean
  row-wise KL divergence between the two groups' Dirichlet posterior-mean
  matrices (symmetric prior; see ``transitions.py``).

The null distribution is built by randomly reassigning whole problems to the
two groups (keeping group sizes fixed) and recomputing both statistics.
p-values use the standard permutation estimator

    p = (1 + #{permuted statistic >= observed}) / (1 + n_perm)

so the smallest reportable p-value is ``1 / (1 + n_perm)``. Holm-adjusted
p-values control the family-wise error rate across the one-vs-rest family.
"""

from __future__ import annotations

import numpy as np

from .taxonomy import HINT_TYPES

_INDEX = {t: i for i, t in enumerate(HINT_TYPES)}
K = len(HINT_TYPES)


def per_problem_counts(sequences: list[list[str]]) -> np.ndarray:
    """Per-chain transition counts: (n_problems, K*K), 4x4 row-major."""
    counts = np.zeros((len(sequences), K * K), dtype=float)
    for p, seq in enumerate(sequences):
        for a, b in zip(seq, seq[1:]):
            counts[p, _INDEX[a] * K + _INDEX[b]] += 1.0
    return counts


def _pooled_row_probs(total: np.ndarray) -> np.ndarray:
    """Row-conditional probabilities of the pooled (K, K) count table."""
    row_sums = total.sum(axis=1, keepdims=True)
    return np.where(row_sums > 0, total / np.where(row_sums > 0, row_sums, 1.0), 1.0 / K)


def _g_half(n: np.ndarray, p_pool: np.ndarray) -> np.ndarray:
    """One group's G contribution per row: (m, K, K) counts -> (m, K)."""
    row_tot = n.sum(axis=2, keepdims=True)
    expected = row_tot * p_pool[None, :, :]
    with np.errstate(divide="ignore", invalid="ignore"):
        term = n * np.log(n / expected)
    return np.where(n > 0, term, 0.0).sum(axis=2)


def g_statistic_rows(a_flat: np.ndarray, total_flat: np.ndarray) -> np.ndarray:
    """Per-'from'-row G for group A vs. its complement, vectorized over perms.

    ``a_flat``: (m, K*K) group-A counts for m permutations; ``total_flat``:
    (K*K,) pooled counts (fixed under permutation). Returns (m, K); summing
    over the row axis gives the full homogeneity G (asymptotically chi-square
    with K*(K-1) df per fully populated table).
    """
    a = a_flat.reshape(-1, K, K)
    total = total_flat.reshape(K, K)
    b = total[None, :, :] - a
    p_pool = _pooled_row_probs(total)
    return 2.0 * (_g_half(a, p_pool) + _g_half(b, p_pool))


def symmetric_kl_statistic(
    a_flat: np.ndarray, total_flat: np.ndarray, prior_strength: float = 1.0
) -> np.ndarray:
    """Symmetric mean row-wise KL between the two groups' posterior means.

    Matches ``kl.symmetric_mean_row_kl`` applied to the posterior-mean
    matrices of ``transitions.learn_domain`` under the symmetric prior.
    Vectorized over permutations: (m, K*K) group-A counts -> (m,).
    """
    a = a_flat.reshape(-1, K, K)
    b = total_flat.reshape(K, K)[None, :, :] - a
    ma = a + prior_strength
    ma = ma / ma.sum(axis=2, keepdims=True)
    mb = b + prior_strength
    mb = mb / mb.sum(axis=2, keepdims=True)
    kl_ab = (ma * np.log(ma / mb)).sum(axis=2).mean(axis=1)
    kl_ba = (mb * np.log(mb / ma)).sum(axis=2).mean(axis=1)
    return 0.5 * (kl_ab + kl_ba)


def holm(pvals: list[float]) -> list[float]:
    """Holm step-down adjustment (family-wise error control)."""
    order = np.argsort(pvals)
    m = len(pvals)
    adjusted = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, (m - rank) * pvals[i])
        adjusted[i] = min(1.0, running)
    return adjusted.tolist()


def _one_vs_rest(
    counts: np.ndarray,
    n_target: int,
    *,
    n_perm: int,
    rng: np.random.Generator,
    prior_strength: float,
    chunk: int = 1_000,
) -> dict:
    """Permutation test for the first ``n_target`` problems of ``counts``.

    ``counts`` is the stacked (n_problems, K*K) per-problem count matrix with
    the target group's problems first. Permutations are processed in chunks
    to bound memory.
    """
    total = counts.sum(axis=0)
    obs_a = counts[:n_target].sum(axis=0, keepdims=True)
    obs_g_rows = g_statistic_rows(obs_a, total)[0]
    obs_g = float(obs_g_rows.sum())
    obs_kl = float(symmetric_kl_statistic(obs_a, total, prior_strength)[0])

    n = counts.shape[0]
    ge_g = ge_kl = 0
    ge_g_rows = np.zeros(K)
    done = 0
    while done < n_perm:
        m = min(chunk, n_perm - done)
        keys = rng.random((m, n))
        pick = np.argpartition(keys, n_target - 1, axis=1)[:, :n_target]
        perm_a = counts[pick].sum(axis=1)
        perm_g_rows = g_statistic_rows(perm_a, total)
        ge_g += int((perm_g_rows.sum(axis=1) >= obs_g).sum())
        ge_g_rows += (perm_g_rows >= obs_g_rows[None, :]).sum(axis=0)
        ge_kl += int(
            (symmetric_kl_statistic(perm_a, total, prior_strength) >= obs_kl).sum()
        )
        done += m

    n_target_trans = int(obs_a.sum())
    return {
        "n_transitions": n_target_trans,
        "n_rest_transitions": int(total.sum()) - n_target_trans,
        "g": obs_g,
        "p_g": (1 + ge_g) / (1 + n_perm),
        "g_rows": obs_g_rows.tolist(),
        "p_g_rows": ((1 + ge_g_rows) / (1 + n_perm)).tolist(),
        "sym_kl": obs_kl,
        "p_kl": (1 + ge_kl) / (1 + n_perm),
        "counts": obs_a.reshape(K, K).tolist(),
        "rest_counts": (total - obs_a[0]).reshape(K, K).tolist(),
    }


def one_vs_rest_tests(
    sequences: dict[str, list[list[str]]],
    *,
    n_perm: int = 10_000,
    seed: int = 0,
    prior_strength: float = 1.0,
) -> dict[str, dict]:
    """Run the one-vs-rest permutation test for every group.

    Returns ``{group: result}`` where each result holds the observed
    statistics, raw and Holm-adjusted permutation p-values, the per-row G
    breakdown, and both groups' transition-count tables (JSON-ready).
    """
    groups = sorted(sequences)
    per_group = {g: per_problem_counts(sequences[g]) for g in groups}
    rng = np.random.default_rng(seed)

    results: dict[str, dict] = {}
    for g in groups:
        stacked = np.vstack([per_group[g]] + [per_group[x] for x in groups if x != g])
        res = _one_vs_rest(
            stacked, len(per_group[g]),
            n_perm=n_perm, rng=rng, prior_strength=prior_strength,
        )
        res["n_problems"] = len(per_group[g])
        res["n_rest_problems"] = stacked.shape[0] - len(per_group[g])
        results[g] = res

    for raw_key, adj_key in (("p_g", "p_g_holm"), ("p_kl", "p_kl_holm")):
        adjusted = holm([results[g][raw_key] for g in groups])
        for g, p in zip(groups, adjusted):
            results[g][adj_key] = p
    return results
