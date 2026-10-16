"""Ridge baseline: hand-crafted EEG features -> video embedding, then cosine retrieval.

Features per channel: mean amplitude in short ERP bins after onset, and log band power
(Welch) over the whole window. Every model has to beat this.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def eeg_features(X: np.ndarray, fs: float, tmin: float, r: dict) -> np.ndarray:
    n, c, t = X.shape
    bin_len = int(round(r["erp_bin"] * fs))
    n_bins = int((r["erp_tmax"] - tmin) * fs) // bin_len
    erp = X[..., : n_bins * bin_len].reshape(n, c, n_bins, bin_len).mean(-1)

    f, pxx = signal.welch(X, fs=fs, nperseg=min(t, int(fs)), axis=-1)
    bp = np.stack([pxx[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in r["bands"].values()], -1)
    return np.concatenate([erp.reshape(n, -1), np.log(bp + 1e-12).reshape(n, -1)], 1)


class RidgeDecoder:
    def __init__(self, r: dict, fs: float, tmin: float):
        self.r, self.fs, self.tmin = r, fs, tmin
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=r["alphas"]))

    def fit(self, X: np.ndarray, target_emb: np.ndarray) -> "RidgeDecoder":
        self.model.fit(eeg_features(X, self.fs, self.tmin, self.r), target_emb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict(eeg_features(X, self.fs, self.tmin, self.r))
        return p / np.linalg.norm(p, axis=1, keepdims=True)
