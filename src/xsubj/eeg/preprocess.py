"""Signal preprocessing and per-subject Euclidean alignment.

Order: band-pass -> notch -> average reference -> baseline -> crop -> resample ->
per-subject scaling + clipping -> Euclidean alignment. Every statistic is computed from
one subject's own trials and never uses labels, so the same code runs on test subjects.

Filtering runs on epochs. If the official data ships continuous recordings, filter
those before epoching instead (fewer edge effects with the 0.5 Hz high-pass).
"""
from __future__ import annotations

from dataclasses import replace
from fractions import Fraction

import numpy as np
from scipy import signal

from src.xsubj.eeg.adapter import SubjectData, load_subjects
from src.xsubj.utils import config_hash, resolve


def preprocess_subject(d: SubjectData, p: dict) -> SubjectData:
    X, fs = d.X.astype(np.float64), d.fs
    sos = signal.butter(4, [p["l_freq"], p["h_freq"]], btype="bandpass", fs=fs, output="sos")
    X = signal.sosfiltfilt(sos, X, axis=-1)
    if p.get("notch") and p["notch"] < fs / 2:
        b, a = signal.iirnotch(p["notch"], Q=30, fs=fs)
        X = signal.filtfilt(b, a, X, axis=-1)
    if p["average_reference"]:
        X = X - X.mean(axis=1, keepdims=True)

    t = d.tmin + np.arange(X.shape[-1]) / fs
    if p["baseline"] and (t < 0).any():
        X = X - X[..., t < 0].mean(axis=-1, keepdims=True)
    keep = (t >= p["tmin"] - 1e-9) & (t < p["tmax"] - 1e-9)
    X = X[..., keep]

    if p.get("resample") and p["resample"] != fs:
        frac = Fraction(p["resample"] / fs).limit_denominator(1000)
        X = signal.resample_poly(X, frac.numerator, frac.denominator, axis=-1)
        fs = float(p["resample"])

    X = X / X.std()                                    # one scale per subject (keeps topography)
    if p.get("clip_std"):
        X = np.clip(X, -p["clip_std"], p["clip_std"])
    if p["euclidean_alignment"]:
        X = euclidean_alignment(X)
    return replace(d, X=X.astype(np.float32), fs=fs, tmin=p["tmin"])


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """He & Wu (2019): whiten trials by the subject's mean spatial covariance, R^-1/2 X."""
    R = np.mean([x @ x.T / x.shape[-1] for x in X], axis=0)
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, w.max() * 1e-6)                  # rank deficiency from average reference
    R_isqrt = (V / np.sqrt(w)) @ V.T
    return np.einsum("cd,ndt->nct", R_isqrt, X)


def load_preprocessed(cfg: dict, subjects: list[str] | None = None) -> list[SubjectData]:
    """Load + preprocess all subjects, caching the result per (data, preprocess) config."""
    cache = resolve(cfg["data"]["cache_dir"]) / config_hash(cfg["data"], cfg["preprocess"])
    if cache.exists():
        out = []
        for f in sorted(cache.glob("*.npz")):
            if subjects is not None and f.stem not in subjects:
                continue
            z = np.load(f, allow_pickle=True)
            out.append(SubjectData(f.stem, z["X"], z["video_id"] if z["video_id"].ndim else None,
                                   float(z["fs"]), float(z["tmin"]), list(z["ch_names"])))
        return out

    cache.mkdir(parents=True, exist_ok=True)
    out = []
    for d in load_subjects(cfg):
        d = preprocess_subject(d, cfg["preprocess"])
        np.savez(cache / f"{d.subject}.npz", X=d.X, fs=d.fs, tmin=d.tmin,
                 video_id=d.video_id if d.video_id is not None else np.array(None),
                 ch_names=np.array(d.ch_names))
        if subjects is None or d.subject in subjects:
            out.append(d)
    return out
