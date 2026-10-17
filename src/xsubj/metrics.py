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


def similarity(pred: np.ndarray, cand_emb: np.ndarray, col_norm: bool = False) -> np.ndarray:
    """Cosine similarity (recordings x candidates). col_norm z-scores each candidate's column over
    the recordings, so videos that look similar to every EEG response ("hubs") stop dominating."""
    sim = pred @ cand_emb.T
    if col_norm:
        sim = (sim - sim.mean(0)) / (sim.std(0) + 1e-8)
    return sim


def csls_similarity(pred: np.ndarray, cand_emb: np.ndarray, k: int = 10) -> np.ndarray:
    """Cross-domain similarity local scaling for EEG queries and video candidates."""
    if k < 1:
        raise ValueError("CSLS neighborhood size k must be positive")
    sim = pred @ cand_emb.T
    k_query = min(k, sim.shape[1])
    k_candidate = min(k, sim.shape[0])
    query_density = np.partition(sim, -k_query, axis=1)[:, -k_query:].mean(axis=1)
    candidate_density = np.partition(sim, -k_candidate, axis=0)[-k_candidate:, :].mean(axis=0)
    return 2 * sim - query_density[:, None] - candidate_density[None, :]


def weighted_csls_similarity(
    predictions: dict[str, np.ndarray],
    candidate_embeddings: dict[str, np.ndarray],
    weights: dict[str, float],
    k: int = 10,
) -> np.ndarray:
    """Compute CSLS per embedding space, then fuse the weighted score matrices."""
    names = tuple(predictions)
    if not names or set(names) != set(candidate_embeddings):
        raise ValueError(
            "Prediction and candidate embedding heads must have the same "
            "non-empty names")
    if set(weights) != set(names):
        raise ValueError("Fusion weights must be provided for every prediction head")
    if any(not np.isfinite(weight) or weight < 0 for weight in weights.values()):
        raise ValueError("Fusion weights must be finite and non-negative")
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("At least one fusion weight must be greater than zero")

    fused = None
    for name in names:
        scores = csls_similarity(
            predictions[name], candidate_embeddings[name], k)
        if fused is None:
            fused = np.zeros_like(scores)
        elif scores.shape != fused.shape:
            raise ValueError(
                f"CSLS score shape for head {name!r} does not match other heads")
        fused += (weights[name] / weight_total) * scores
    assert fused is not None
    return fused


def retrieval_metrics(pred: np.ndarray, y: np.ndarray, bank_emb: np.ndarray, ks, seed: int = 0,
                      col_norm: bool = False, csls_k: int | None = None) -> dict:
    """pred: (n, d) unit EEG embeddings; y: (n,) bank indices. Candidates = unique videos in y."""
    pool = np.unique(y)
    col = np.searchsorted(pool, y)
    sim = (csls_similarity(pred, bank_emb[pool], csls_k)
           if csls_k is not None else similarity(pred, bank_emb[pool], col_norm))
    rng = np.random.default_rng(seed)
    out = {"n_trials": int(len(y)), "n_videos": int(len(pool))}
    for k in ks:
        kk = len(pool) if k == "all" else min(int(k), len(pool))
        ranks = _ranks(sim, col, k, rng)
        out[f"top1@{k}"] = float((ranks == 0).mean())
        if kk > 5:
            out[f"top5@{k}"] = float((ranks < 5).mean())
        out[f"chance1@{k}"] = 1.0 / kk
        if k == "all":
            out["mean_rank@all"] = float(ranks.mean() + 1)    # 1 = best, chance (n+1)/2
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


def permutation_test(pred: np.ndarray, y: np.ndarray, bank_emb: np.ndarray, n_perm: int, seed: int = 0,
                     col_norm: bool = False, csls_k: int | None = None) -> dict:
    """Top-1 among all candidates vs. the null of shuffled trial->video labels."""
    pool = np.unique(y)
    sim = (csls_similarity(pred, bank_emb[pool], csls_k)
           if csls_k is not None else similarity(pred, bank_emb[pool], col_norm))
    best = sim.argmax(1)
    col = np.searchsorted(pool, y)
    obs = float((best == col).mean())
    rng = np.random.default_rng(seed)
    null = np.array([(best == rng.permutation(col)).mean() for _ in range(n_perm)])
    return {"top1@all": obs, "null_mean": float(null.mean()),
            "p_value": float((1 + (null >= obs).sum()) / (1 + n_perm))}
