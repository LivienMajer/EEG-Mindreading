"""Plots for the scientific writeup: topographies, spectra, accuracy."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_trial(X: np.ndarray, sfreq: float, ch_names: list[str],
               trial_idx: int = 0, ax=None):
    """Butterfly plot of one trial."""
    t = np.arange(X.shape[-1]) / sfreq
    ax = ax or plt.gca()
    for ch in X[trial_idx]:
        ax.plot(t, ch, lw=0.6, alpha=0.7)
    ax.set(xlabel="Time (s)", ylabel="Amplitude",
           title=f"Trial {trial_idx} ({len(ch_names)} channels)")
    return ax


def plot_bandpower_topography(power_per_ch: np.ndarray, ch_names: list[str],
                              band: str = "alpha", ax=None):
    """Simple scatter topography on a 10-20-ish 2D layout.

    TODO: replace with a real montage (mne.viz.plot_topomap) once channels
    are confirmed as standard 10-20 names.
    """
    ax = ax or plt.gca()
    n = len(ch_names)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False)
    x, y = np.cos(angles), np.sin(angles)
    sc = ax.scatter(x, y, c=power_per_ch, cmap="viridis", s=200)
    for xi, yi, name in zip(x, y, ch_names):
        ax.annotate(name, (xi * 1.15, yi * 1.15), ha="center", va="center",
                    fontsize=7)
    ax.set(title=f"{band} power topography", aspect="equal")
    ax.axis("off")
    plt.colorbar(sc, ax=ax, shrink=0.7)
    return ax


def plot_psd(X: np.ndarray, sfreq: float, ax=None):
    """Average PSD across trials and channels."""
    from scipy import signal
    freqs, psd = signal.welch(X, fs=sfreq, axis=-1, nperseg=256)
    ax = ax or plt.gca()
    ax.loglog(freqs[1:], psd.mean(axis=(0, 1))[1:])
    ax.set(xlabel="Frequency (Hz)", ylabel="Power", title="Average PSD")
    return ax


def plot_decoding_by_time(bandpowers, fold_accs, labels, out="figures"):
    """Bar chart of cross-subject accuracy per frequency band."""
    import pathlib
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.bar(labels, fold_accs)
    ax.set(ylabel="Candidate-match accuracy")
    pathlib.Path(out).mkdir(exist_ok=True)
    fig.savefig(f"{out}/accuracy_bands.png", dpi=150, bbox_inches="tight")
    return fig
