"""Learn 4x4 hint-type transition matrices with a Dirichlet-Multinomial model.

Each row of the transition matrix is a categorical distribution over the next
hint type. We place a Dirichlet prior on each row and observe multinomial
transition counts, so the posterior over each row is again Dirichlet. The
reported transition matrix is the Dirichlet posterior mean:

    M[i, j] = (alpha[i, j] + n[i, j]) / sum_k (alpha[i, k] + n[i, k])

where ``n[i, j]`` counts observed transitions i -> j and ``alpha`` is the prior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .taxonomy import HINT_TYPES

_INDEX = {t: i for i, t in enumerate(HINT_TYPES)}
N = len(HINT_TYPES)


def count_transitions(sequences: list[list[str]]) -> np.ndarray:
    """Count consecutive i -> j transitions across all label sequences."""
    counts = np.zeros((N, N), dtype=float)
    for seq in sequences:
        for a, b in zip(seq, seq[1:]):
            counts[_INDEX[a], _INDEX[b]] += 1.0
    return counts


def symmetric_prior(strength: float = 1.0) -> np.ndarray:
    """Uniform Dirichlet prior: every transition equally likely a priori."""
    return np.full((N, N), strength, dtype=float)


def pedagogical_prior(strength: float = 1.0) -> np.ndarray:
    """Asymmetric, pedagogically informed prior.

    Encodes the expected Socratic arc: feature/principle hints early, bottom-out
    in the middle, extension at the end. Favored transitions (e.g. f -> r,
    b -> b, b -> e) get higher pseudo-counts; implausible ones (e.g. f -> e,
    e -> f) get lower pseudo-counts. ``strength`` scales the whole prior.

    Rows/cols follow HINT_TYPES order: [f, r, b, e].
    """
    #            ->f   ->r   ->b   ->e
    weights = np.array(
        [
            [1.0, 3.0, 2.0, 0.5],   # from feature-pointing
            [1.0, 1.5, 3.0, 0.5],   # from principle-stating
            [0.5, 1.0, 3.0, 2.0],   # from bottom-out
            [0.5, 0.5, 1.0, 3.0],   # from extension
        ],
        dtype=float,
    )
    return weights * strength


@dataclass
class DomainPosterior:
    domain: str
    counts: np.ndarray          # observed transition counts (N x N)
    prior: np.ndarray           # Dirichlet prior (N x N)
    n_sequences: int
    n_questions: int

    @property
    def alpha_post(self) -> np.ndarray:
        """Posterior Dirichlet concentration per row."""
        return self.prior + self.counts

    @property
    def transition_matrix(self) -> np.ndarray:
        """Posterior-mean transition matrix (rows sum to 1)."""
        a = self.alpha_post
        return a / a.sum(axis=1, keepdims=True)

    def row_variance(self) -> np.ndarray:
        """Posterior variance of each entry (uncertainty of the estimate)."""
        a = self.alpha_post
        a0 = a.sum(axis=1, keepdims=True)
        mean = a / a0
        return mean * (1.0 - mean) / (a0 + 1.0)


def learn_domain(domain: str, sequences: list[list[str]], prior: np.ndarray) -> DomainPosterior:
    counts = count_transitions(sequences)
    return DomainPosterior(
        domain=domain,
        counts=counts,
        prior=prior,
        n_sequences=len(sequences),
        n_questions=sum(len(s) for s in sequences),
    )
