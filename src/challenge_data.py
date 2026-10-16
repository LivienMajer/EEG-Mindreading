"""Load the challenge data in its exact documented format.

Layout (from the official challenge PDF):
    stimuli_train/  0001.mp4 .. 1000.mp4   (1000 train videos)
    stimuli_test/   1001.mp4 .. 1102.mp4  (102 test videos)
    trn/sub-XX_trn.npz   dict: video_id (no leading zeros, e.g. "35")
                              -> (17, 1500) float, 17 ch @ 500 Hz, 3 s from onset
    tst/sub-XX_tst.npy   (102, 17, 1500), shuffled order
    tst_stim/sub-XX_tst_stim.npy
                         (102,) video IDs 1001..1102, same order as tst;
                         only provided for sub-01..sub-04.

Rules to respect:
  - Train on trn data of ALL SIX subjects (allowed + encouraged).
  - tst / tst_stim of sub-01..04: validation/model selection ONLY, and
    must NOT be used as reference templates for sub-05/06.
  - Unsupervised normalization (z-scoring) of test recordings allowed.
  - Never train on any test recordings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

SFREQ = 500.0
N_CHANNELS = 17
N_TIMES = 1500
N_CANDIDATES = 102
TEST_VIDEO_IDS = np.arange(1001, 1103)


@dataclass
class SubjectTrain:
    subject: str
    video_ids: np.ndarray        # (n_trials,) int video id (1..1000)
    X: np.ndarray                # (n_trials, 17, 1500) float32


@dataclass
class SubjectTest:
    subject: str
    X: np.ndarray                # (102, 17, 1500) float32, submission order
    y: np.ndarray | None         # (102,) true video ids, None for sub-05/06


def load_train_subject(data_dir: str | Path, subject: str) -> SubjectTrain:
    """Load one subject's training EEG. subject like '01' or 'sub-01'."""
    subject = subject.removeprefix("sub-")
    path = Path(data_dir) / "trn" / f"sub-{subject}_trn.npz"
    d = np.load(path, allow_pickle=True)
    keys = sorted(d.files, key=lambda k: int(k))
    video_ids = np.array([int(k) for k in keys], dtype=int)
    X = np.stack([d[k] for k in keys]).astype(np.float32)
    assert X.shape[1:] == (N_CHANNELS, N_TIMES), X.shape
    return SubjectTrain(subject, video_ids, X)


def load_test_subject(data_dir: str | Path, subject: str,
                      with_labels: bool | None = None) -> SubjectTest:
    """Load one subject's test EEG (and labels if available)."""
    subject = subject.removeprefix("sub-")
    data_dir = Path(data_dir)
    X = np.load(data_dir / "tst" / f"sub-{subject}_tst.npy").astype(np.float32)
    assert X.shape == (N_CANDIDATES, N_CHANNELS, N_TIMES), X.shape

    stim_path = data_dir / "tst_stim" / f"sub-{subject}_tst_stim.npy"
    if with_labels is None:
        with_labels = stim_path.exists()
    y = (np.load(stim_path).astype(int) if with_labels else None)
    return SubjectTest(subject, X, y)


def load_all_train(data_dir: str | Path,
                   subjects: tuple[str, ...] = ("01", "02", "03",
                                                "04", "05", "06")) -> SubjectTrain:
    """Concatenate training data of multiple subjects."""
    parts = [load_train_subject(data_dir, s) for s in subjects]
    video_ids = np.concatenate([p.video_ids for p in parts])
    X = np.concatenate([p.X for p in parts])
    subj = np.concatenate([[p.subject] * len(p.video_ids) for p in parts])
    return SubjectTrain("+".join(p.subject for p in parts), video_ids, X)


def zscore(X: np.ndarray, axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Per-channel z-scoring (allowed unsupervised normalization on test)."""
    mu = X.mean(axis=axis, keepdims=True)
    sd = X.std(axis=axis, keepdims=True)
    return ((X - mu) / (sd + eps)).astype(np.float32)
