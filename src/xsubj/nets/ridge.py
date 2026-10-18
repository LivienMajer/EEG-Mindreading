"""Ridge decoder: hand-crafted EEG features -> video embedding, then cosine retrieval.

Feature blocks (each switchable in the `ridge` config section):
  erp       mean amplitude per channel in `erp_bin`-second bins over [erp_tmin, erp_tmax)
  bandpower log Welch power per channel and band over the whole window
  riemann   tangent-space spatial covariance per (optional) frequency band: upper triangle of
            logm(C). After Euclidean alignment each subject's mean covariance is ~identity,
            so identity is the reference point and no fitting is needed.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.linear_model import Ridge, RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def erp_bins(X: np.ndarray, fs: float, tmin: float, bin_s: float, t0: float, t1: float) -> np.ndarray:
    i0, i1 = int(round((t0 - tmin) * fs)), int(round((t1 - tmin) * fs))
    bin_len = max(1, int(round(bin_s * fs)))
    n_bins = (i1 - i0) // bin_len
    seg = X[..., i0: i0 + n_bins * bin_len]
    return seg.reshape(*X.shape[:2], n_bins, bin_len).mean(-1).reshape(len(X), -1)


def band_power(X: np.ndarray, fs: float, bands: dict) -> np.ndarray:
    f, pxx = signal.welch(X, fs=fs, nperseg=min(X.shape[-1], int(fs)), axis=-1)
    bp = np.stack([pxx[..., (f >= lo) & (f < hi)].mean(-1) for lo, hi in bands.values()], -1)
    return np.log(bp + 1e-12).reshape(len(X), -1)


def tangent_space(X: np.ndarray, shrink: float = 0.1) -> np.ndarray:
    C = np.einsum("nct,ndt->ncd", X, X) / X.shape[-1]
    tr = np.trace(C, axis1=1, axis2=2)[:, None, None] / C.shape[1]
    C = (1 - shrink) * C + shrink * tr * np.eye(C.shape[1])
    w, V = np.linalg.eigh(C)
    L = np.einsum("ncd,nd,ned->nce", V, np.log(np.maximum(w, 1e-10)), V)   # logm(C)
    iu = np.triu_indices(C.shape[1])
    weight = np.where(iu[0] == iu[1], 1.0, np.sqrt(2))                   # keeps the Frobenius norm
    return L[:, iu[0], iu[1]] * weight


def eeg_features(X: np.ndarray, fs: float, tmin: float, r: dict) -> np.ndarray:
    blocks = []
    if r.get("erp", True):
        blocks.append(erp_bins(X, fs, tmin, r["erp_bin"], r.get("erp_tmin", tmin), r["erp_tmax"]))
    if r.get("bandpower", True):
        blocks.append(band_power(X, fs, r["bands"]))
    if r.get("riemann", False):
        for lo, hi in r.get("riemann_bands") or [(None, None)]:
            Xb = X if lo is None else signal.sosfiltfilt(
                signal.butter(4, [lo, hi], btype="bandpass", fs=fs, output="sos"), X, axis=-1)
            blocks.append(tangent_space(Xb))
    return np.concatenate(blocks, 1)


class RidgeDecoder:
    """StandardScaler + ridge. alpha is chosen by RidgeCV (efficient leave-one-out on the training
    trials), then multiplied by `alpha_scale` (default 1): the trials are single presentations, the
    test recordings look like averages, so the CV choice regularises too strongly for them."""

    def __init__(self, r: dict, fs: float, tmin: float):
        self.r, self.fs, self.tmin = r, fs, tmin
        self.model = make_pipeline(StandardScaler(), RidgeCV(alphas=r["alphas"]))

    def fit(self, X: np.ndarray, target_emb: np.ndarray) -> "RidgeDecoder":
        F = eeg_features(X, self.fs, self.tmin, self.r)
        self.model.fit(F, target_emb)
        scale = self.r.get("alpha_scale", 1.0)
        if scale != 1.0:
            alpha = self.model[-1].alpha_ * scale
            self.model = make_pipeline(StandardScaler(), Ridge(alpha=alpha)).fit(F, target_emb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        p = self.model.predict(eeg_features(X, self.fs, self.tmin, self.r))
        return p / np.linalg.norm(p, axis=1, keepdims=True)
