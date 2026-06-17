"""Row-wise KL divergence between domain transition matrices.

The proposal evaluates similarity by computing the KL divergence row-by-row
between two domains' transition matrices and averaging across rows. KL is
asymmetric, so we also provide a symmetrized version (the average of the two
directions), which yields a proper distance-like matrix suitable for a
similarity heatmap.
"""

from __future__ import annotations

import numpy as np


def _row_kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q) for two probability vectors (nats)."""
    mask = p > 0
    return float(np.sum(p[mask] * np.log(p[mask] / q[mask])))


def mean_row_kl(matrix_p: np.ndarray, matrix_q: np.ndarray) -> float:
    """Average over rows of KL(P_row || Q_row). Directed (P relative to Q)."""
    rows = matrix_p.shape[0]
    return float(np.mean([_row_kl(matrix_p[i], matrix_q[i]) for i in range(rows)]))


def symmetric_mean_row_kl(matrix_p: np.ndarray, matrix_q: np.ndarray) -> float:
    """Symmetrized average row KL: 0.5*(mean_row_kl(P,Q)+mean_row_kl(Q,P))."""
    return 0.5 * (mean_row_kl(matrix_p, matrix_q) + mean_row_kl(matrix_q, matrix_p))


def divergence_matrix(
    domains: list[str],
    matrices: dict[str, np.ndarray],
    symmetric: bool = True,
) -> np.ndarray:
    """Pairwise divergence matrix between every domain (lower = more similar)."""
    n = len(domains)
    out = np.zeros((n, n), dtype=float)
    for i, di in enumerate(domains):
        for j, dj in enumerate(domains):
            if i == j:
                continue
            if symmetric:
                out[i, j] = symmetric_mean_row_kl(matrices[di], matrices[dj])
            else:
                out[i, j] = mean_row_kl(matrices[di], matrices[dj])
    return out


def similarity_matrix(divergence: np.ndarray) -> np.ndarray:
    """Map divergence to a (0, 1] similarity via exp(-divergence)."""
    return np.exp(-divergence)


def most_similar_pairs(domains: list[str], divergence: np.ndarray):
    """Return (domain_a, domain_b, divergence) sorted from most to least similar."""
    pairs = []
    n = len(domains)
    for i in range(n):
        for j in range(i + 1, n):
            pairs.append((domains[i], domains[j], float(divergence[i, j])))
    pairs.sort(key=lambda t: t[2])
    return pairs
