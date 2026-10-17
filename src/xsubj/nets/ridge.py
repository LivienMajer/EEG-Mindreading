"""Ridge EEG-to-video decoder and its compact EEG feature transforms."""
from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def erp_bins(
    X: np.ndarray,
    fs: float,
    tmin: float,
    bin_width: float,
    start: float,
    stop: float,
) -> np.ndarray:
    """Average EEG into consecutive temporal bins, returning (trials, features)."""
    if bin_width <= 0 or stop <= start:
        raise ValueError("ERP bin width and interval must be positive")
    starts = np.arange(start, stop - 1e-9, bin_width)
    bins = []
    for left in starts:
        i0 = max(0, int(round((left - tmin) * fs)))
        i1 = min(X.shape[-1], int(round((left + bin_width - tmin) * fs)))
        if i1 <= i0:
            raise ValueError(
                f"ERP interval [{left}, {left + bin_width}) is outside "
                f"the available data window [{tmin}, {tmin + X.shape[-1] / fs})")
        bins.append(X[..., i0:i1].mean(axis=-1))
    if not bins:
        raise ValueError("ERP interval did not include any bins")
    return np.concatenate(bins, axis=1).astype(np.float32)


def _bandpower(X: np.ndarray, fs: float, bands: list) -> np.ndarray:
    freqs, psd = signal.welch(
        X, fs=fs, axis=-1, nperseg=min(256, X.shape[-1]))
    features = []
    for low, high in bands:
        mask = (freqs >= low) & (freqs < high)
        if not mask.any():
            raise ValueError(f"No PSD bins in band [{low}, {high}) Hz")
        features.append(np.log(psd[..., mask].mean(axis=-1) + 1e-12))
    return np.stack(features, axis=-1).reshape(len(X), -1).astype(np.float32)


def _covariance_features(X: np.ndarray, fs: float, bands: list | None) -> np.ndarray:
    """Vectorize (optionally filter-bank) log-diagonal-regularized covariances."""
    sources = []
    for band in bands or [[0.5, min(45.0, fs * 0.49)]]:
        sos = signal.butter(4, band, btype="bandpass", fs=fs, output="sos")
        filtered = signal.sosfiltfilt(sos, X, axis=-1)
        cov = np.einsum("nct,ndt->ncd", filtered, filtered) / X.shape[-1]
        scale = np.trace(cov, axis1=1, axis2=2) / X.shape[1]
        cov += (scale[:, None, None] * 1e-6 + 1e-12) * np.eye(X.shape[1])
        w, v = np.linalg.eigh(cov)
        log_cov = np.einsum("nci,ni,ndi->ncd", v, np.log(w), v)
        upper = np.triu_indices(X.shape[1])
        sources.append(log_cov[:, upper[0], upper[1]])
    return np.concatenate(sources, axis=1).astype(np.float32)


class RidgeDecoder:
    """Map compact EEG features to frozen video embeddings using RidgeCV."""

    def __init__(self, cfg: dict, fs: float, tmin: float):
        self.cfg = cfg
        self.fs = fs
        self.tmin = tmin
        self.model = make_pipeline(
            StandardScaler(),
            RidgeCV(alphas=np.asarray(cfg["alphas"], dtype=float)),
        )

    def _feat(self, X: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        parts = []
        if cfg.get("erp", True):
            parts.append(erp_bins(
                X, self.fs, self.tmin, cfg["erp_bin"],
                cfg["erp_tmin"], cfg["erp_tmax"]))
        if cfg.get("bandpower", True):
            parts.append(_bandpower(X, self.fs, cfg["bands"]))
        if cfg.get("riemann", False):
            parts.append(_covariance_features(
                X, self.fs, cfg.get("riemann_bands")))
        if not parts:
            raise ValueError("Ridge config must enable at least one feature family")
        return np.concatenate(parts, axis=1)

    def fit(self, X: np.ndarray, target_emb: np.ndarray) -> "RidgeDecoder":
        if len(X) != len(target_emb):
            raise ValueError("EEG trials and target embeddings must have equal length")
        self.model.fit(self._feat(X), target_emb)
        return self

    def fit_features(
        self, features: np.ndarray, target_emb: np.ndarray
    ) -> "RidgeDecoder":
        if features.ndim != 2 or len(features) != len(target_emb):
            raise ValueError(
                "Precomputed features and target embeddings must be aligned 2D arrays")
        self.model.fit(features, target_emb)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        prediction = self.model.predict(self._feat(X))
        return self._normalize(prediction)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        if features.ndim != 2:
            raise ValueError("Precomputed Ridge features must be a 2D array")
        return self._normalize(self.model.predict(features))

    @staticmethod
    def _normalize(prediction: np.ndarray) -> np.ndarray:
        return (prediction / (np.linalg.norm(prediction, axis=1, keepdims=True)
                              + 1e-8)).astype(np.float32)
