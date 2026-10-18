"""Sinkhorn re-ranking: use that every candidate video appears exactly once per test set.

Each subject's tst split has 102 recordings of the 102 candidate videos, one each. Ranking
every recording on its own ignores that, so one "attractive" video can sit at the top for
many recordings. Sinkhorn normalisation (Sinkhorn 1964; Cuturi 2013) turns the fused score
table into P = exp(score / tau) and rescales rows and columns in turn until every recording
and every video sums to 1 (a soft one-to-one assignment); each recording is then ranked by
its row of P. Label-free: it uses only the shape of the test set (the same assumption as
the per-candidate normalisation in src/xsubj/submit.py).

tau sets how strongly the constraint acts (small = close to a hard one-to-one assignment).
For mean-rank fusion the score is minus the weighted mean rank (0..101), so tau is in rank
units; LOO picks tau = 2 (docs/megafusion_rebuild.md, section 8).
"""

from __future__ import annotations

import numpy as np

CANDIDATES = np.arange(1001, 1103)


def mean_rank_matrix(rankings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """(n, 102) weighted mean rank (0 = top) of every candidate, columns 1001..1102."""
    w = np.asarray(weights, float) / np.sum(weights)
    n, k = rankings[0].shape
    avg = np.zeros((n, len(CANDIDATES)))
    for r, wi in zip(rankings, w):
        R = np.empty((n, len(CANDIDATES)))
        R[np.arange(n)[:, None], np.searchsorted(CANDIDATES, r)] = np.arange(k)
        avg += wi * R
    return avg


def sinkhorn(score: np.ndarray, tau: float, n_iter: int = 200) -> np.ndarray:
    """Doubly normalised exp(score / tau); higher score = better match."""
    P = np.exp((score - score.max()) / tau)
    for _ in range(n_iter):
        P /= P.sum(1, keepdims=True)
        P /= P.sum(0, keepdims=True)
    return P


def rank(score: np.ndarray, tau: float | None = None) -> np.ndarray:
    """(n, 102) candidate ids, best first; with tau, after Sinkhorn normalisation."""
    s = score if tau is None else np.log(sinkhorn(score, tau) + 1e-300)
    return CANDIDATES[np.argsort(-s, axis=1, kind="stable")]
