"""Similarity scores next to every member's ranking, for score-level fusion.

Every member ranks the 102 candidates by a similarity table S (n_recordings, 102), columns in
candidate order 1001..1102, and saves only the ranking (ids, best first). Mean-rank fusion
then loses how sure each member was. The member scripts now also save S, with the SAME row
order as their ranking file, as <ranking stem>.scores.npy beside it; argsort(-S) gives back
the ranking. Used by src.loo_score_fusion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def scores_path(ranking_path) -> Path:
    p = Path(ranking_path)
    return p.with_name(p.stem + ".scores.npy")


def save_scores(ranking_path, S) -> None:
    S = np.asarray(S, np.float64)
    assert S.ndim == 2 and S.shape[1] == 102, S.shape
    np.save(scores_path(ranking_path), S)


def load_scores(ranking_path) -> np.ndarray:
    return np.load(scores_path(ranking_path))
