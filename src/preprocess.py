"""EEG preprocessing: filtering, artifact rejection, features."""

from __future__ import annotations

import numpy as np
from scipy import signal


def bandpass(X: np.ndarray, sfreq: float, l_freq: float = 0.1,
             h_freq: float = 40.0, order: int = 4) -> np.ndarray:
    """Zero-phase Butterworth bandpass. X: (n_trials, n_ch, n_times)."""
    nyq = sfreq / 2.0
    sos = signal.butter(order, [l_freq / nyq, min(h_freq, nyq * 0.99)],
                         btype="bandpass", output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)


def notch(X: np.ndarray, sfreq: float, freq: float = 50.0) -> np.ndarray:
    """Remove power-line noise."""
    nyq = sfreq / 2.0
    b, a = signal.iirnotch(freq / nyq, Q=30.0)
    return signal.filtfilt(b, a, X, axis=-1).astype(np.float32)


def reject_artifacts(X: np.ndarray, threshold_uv: float = 150.0) -> np.ndarray:
    """Return boolean mask of trials to KEEP (peak-to-peak below threshold)."""
    ptp = X.max(axis=-1) - X.min(axis=-1)  # (n_trials, n_ch)
    return ptp.max(axis=1) < threshold_uv


def log_bandpower(X: np.ndarray, sfreq: float,
                  bands: dict[str, tuple[float, float]] | None = None) -> np.ndarray:
    """Log band power features.

    Returns (n_trials, n_ch, n_bands). Bands default to the classic ranges.
    """
    if bands is None:
        bands = {"delta": (1, 4), "theta": (4, 8), "alpha": (8, 13),
                 "beta": (13, 30), "gamma": (30, 45)}
    freqs, psd = signal.welch(X, fs=sfreq, axis=-1, nperseg=min(256, X.shape[-1]))
    feats = []
    for lo, hi in bands.values():
        mask = (freqs >= lo) & (freqs < hi)
        feats.append(np.log(psd[..., mask].mean(axis=-1) + 1e-12))
    return np.stack(feats, axis=-1).astype(np.float32)


def crop_window(X: np.ndarray, sfreq: float, t_min: float = 0.0,
                t_max: float = 3.0) -> np.ndarray:
    """Crop each trial to [t_min, t_max] seconds since onset."""
    n_start = int(t_min * sfreq)
    n_end = min(int(t_max * sfreq), X.shape[-1])
    return X[..., n_start:n_end]


def per_video_average(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Average trials per video label. Returns (X_avg, unique_labels)."""
    labels = np.unique(y)
    X_avg = np.stack([X[y == lab].mean(axis=0) for lab in labels])
    return X_avg, labels
