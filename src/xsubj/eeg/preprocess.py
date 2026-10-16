"""Signal preprocessing and Euclidean alignment, per subject and split.

Order: band-pass -> notch -> average reference -> baseline -> crop -> resample ->
remove the split's mean response -> scaling + clipping -> Euclidean alignment. Every
statistic comes from one subject's trials of one split and never uses labels (the rules
allow unsupervised normalisation of test recordings).

Why per split: in the challenge data the test recordings have ~0.65x the amplitude of the
training recordings and their mean evoked response is inverted (r = -0.7 to -0.9 with the
training ERP, every subject and channel), while the stimulus-specific signal keeps its
sign. Removing each split's own mean response and scaling per split cancels both.
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

    if p.get("remove_mean_response"):
        X = X - X.mean(axis=0, keepdims=True)          # stimulus-independent evoked response
    X = X / X.std()                                    # one scale per subject+split (keeps topography)
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
    """Load + preprocess all subjects/splits, caching the result per (data, preprocess) config."""
    cache = resolve(cfg["data"]["cache_dir"]) / config_hash(cfg["data"], cfg["preprocess"])
    if not (cache / "done").exists():
        cache.mkdir(parents=True, exist_ok=True)
        for d in load_subjects(cfg):
            d = preprocess_subject(d, cfg["preprocess"])
            np.savez(cache / f"{d.key}.npz", X=d.X, fs=d.fs, tmin=d.tmin, subject=d.subject, split=d.split,
                     video_id=d.video_id if d.video_id is not None else np.array(None),
                     ch_names=np.array(d.ch_names))
        (cache / "done").touch()

    out = []
    for f in sorted(cache.glob("*.npz")):
        z = np.load(f, allow_pickle=True)
        if subjects is not None and str(z["subject"]) not in subjects:
            continue
        out.append(SubjectData(str(z["subject"]), z["X"], z["video_id"] if z["video_id"].ndim else None,
                               float(z["fs"]), float(z["tmin"]), list(z["ch_names"]), str(z["split"])))
    return out
