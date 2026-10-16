"""Retrieval metrics: is the true video ranked first among k candidates?"""
from __future__ import annotations

import numpy as np


def _ranks(sim: np.ndarray, true_col: np.ndarray, k, rng) -> np.ndarray:
    """Rank of the true video among itself + (k-1) random distractors from the pool (0 = best)."""
    n, P = sim.shape
    s_true = sim[np.arange(n), true_col]
    if k == "all" or k >= P:
        return (sim > s_true[:, None]).sum(1)
    r = rng.random((n, P))
    r[np.arange(n), true_col] = np.inf                       # never pick the true video as distractor
    distractors = np.argsort(r, axis=1)[:, : k - 1]
    return (np.take_along_axis(sim, distractors, 1) > s_true[:, None]).sum(1)


def retrieval_metrics(pred: np.ndarray, y: np.ndarray, bank_emb: np.ndarray, ks, seed: int = 0) -> dict:
    """pred: (n, d) unit EEG embeddings; y: (n,) bank indices. Candidates = unique videos in y."""
    pool = np.unique(y)
    col = np.searchsorted(pool, y)
    sim = pred @ bank_emb[pool].T
    rng = np.random.default_rng(seed)
    out = {"n_trials": int(len(y)), "n_videos": int(len(pool))}
    for k in ks:
        kk = len(pool) if k == "all" else min(int(k), len(pool))
        ranks = _ranks(sim, col, k, rng)
        out[f"top1@{k}"] = float((ranks == 0).mean())
        if kk > 5:
            out[f"top5@{k}"] = float((ranks < 5).mean())
        out[f"chance1@{k}"] = 1.0 / kk
    return out


def average_repeats(pred: np.ndarray, y: np.ndarray, subject: np.ndarray):
    """Average embeddings of repeated presentations of the same video to the same subject."""
    keys = np.array([f"{s}|{v}" for s, v in zip(subject, y)])
    uniq, inv = np.unique(keys, return_inverse=True)
    avg = np.zeros((len(uniq), pred.shape[1]), dtype=pred.dtype)
    np.add.at(avg, inv, pred)
    avg /= np.linalg.norm(avg, axis=1, keepdims=True)
    y_avg = np.zeros(len(uniq), dtype=y.dtype)
    y_avg[inv] = y
    subj_avg = np.empty(len(uniq), dtype=subject.dtype)
    subj_avg[inv] = subject
    return avg, y_avg, subj_avg


def permutation_test(pred: np.ndarray, y: np.ndarray, bank_emb: np.ndarray, n_perm: int, seed: int = 0) -> dict:
    """Top-1 among all candidates vs. the null of shuffled trial->video labels."""
    pool = np.unique(y)
    sim = pred @ bank_emb[pool].T
    best = sim.argmax(1)
    col = np.searchsorted(pool, y)
    obs = float((best == col).mean())
    rng = np.random.default_rng(seed)
    null = np.array([(best == rng.permutation(col)).mean() for _ in range(n_perm)])
    return {"top1@all": obs, "null_mean": float(null.mean()),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}
