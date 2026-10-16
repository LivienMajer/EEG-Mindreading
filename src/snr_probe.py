"""Empirical preprocessing comparison via cross-subject EEG identifiability.

Idea: for each video, each subject has exactly ONE 3-s trial. Across subjects,
the same video's trials share the stimulus-locked response. So:

  d' = corr(same video, different subjects) - corr(different videos)

is a model-free, feature-free proxy for how much video identity survives
preprocessing. Run over subject pairs and hundreds of videos -> reliable.

Usage:  python -m src.snr_probe --data-dir <challenge root>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import butter, decimate, sosfiltfilt


def load_subject_trials(root: Path, sub: str, max_trials: int | None = None):
    d = np.load(root / "trn" / f"sub-{sub}_trn.npz")
    ids = sorted(d.files, key=int)
    if max_trials:
        ids = ids[:max_trials]
    v = np.array(ids, dtype=int)
    X = np.stack([d[k] for k in ids]).astype(np.float32)
    return v, X


def preprocess(X, sfreq=500.0, dec=5, band=None, win=None, baseline=None,
               notch=None, avg_ref=False, laplacian=False):
    """band=(lo,hi) Hz applied at native rate; win=(t0,t1) SECONDS; baseline=(tb0,tb1)."""
    if notch:
        from scipy.signal import iirnotch, filtfilt
        b, a = iirnotch(notch / (sfreq / 2), Q=30.0)
        X = filtfilt(b, a, X, axis=-1)
    if avg_ref:  # common average re-referencing
        X = X - X.mean(axis=1, keepdims=True)
    if laplacian:  # crude local Laplacian: channel minus neighbors' mean (1D order)
        X = X - np.roll(X, 1, axis=1) * 0.5 - np.roll(X, -1, axis=1) * 0.5
    if band:
        sos = butter(4, [band[0] / (sfreq / 2), min(band[1], sfreq / 2 * 0.99) / (sfreq / 2)],
                     btype="bandpass", output="sos")
        X = sosfiltfilt(sos, X, axis=-1)
    if dec > 1:
        X = decimate(X, dec, axis=-1, ftype="fir", zero_phase=True)
        sfreq = sfreq / dec
    if baseline:  # subtract mean over [tb0,tb1] s from whole trial
        lo, hi = int(baseline[0] * sfreq), int(baseline[1] * sfreq)
        X = X - X[..., lo:hi].mean(-1, keepdims=True)
    if win:
        lo, hi = int(win[0] * sfreq), int(win[1] * sfreq)
        X = X[..., lo:hi]
    return X.astype(np.float32)


def _flat_norm(X):
    z = X.reshape(len(X), -1)
    z = (z - z.mean(1, keepdims=True)) / (z.std(1, keepdims=True) + 1e-8)
    return z


def identify_pair(Xa, va, Xb, vb, n_neg=20000, rng=None):
    """d' = mean corr(same video pairs) - mean corr(random different pairs)."""
    rng = rng or np.random.default_rng(0)
    za, zb = _flat_norm(Xa), _flat_norm(Xb)
    common, ia, ib = np.intersect1d(va, vb, return_indices=True)

    same_corr = (za[ia] * zb[ib]).sum(1) / za.shape[1]

    n = len(common)
    neg_corr = np.empty(n_neg)
    for j in range(n_neg):
        i, k = rng.integers(n), rng.integers(n)
        while i == k:
            i, k = rng.integers(n), rng.integers(n)
        neg_corr[j] = za[ia[i]] @ zb[ib[k]] / za.shape[1]

    return float(same_corr.mean() - neg_corr.mean()), float(same_corr.mean())


VARIANTS = {
    # fine window tuning
    "raw, win 50-300ms":               dict(dec=5, win=(0.05, 0.30)),
    "raw, win 50-350ms":               dict(dec=5, win=(0.05, 0.35)),
    "raw, win 50-400ms":               dict(dec=5, win=(0.05, 0.40)),
    "raw, win 50-400ms + notch50":     dict(dec=5, win=(0.05, 0.40), notch=50),
    "raw, win 50-400ms, avg ref":      dict(dec=5, win=(0.05, 0.40), avg_ref=True),
    "raw, win 50-400ms, laplacian":    dict(dec=5, win=(0.05, 0.40), laplacian=True),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subjects", nargs="+", default=["01", "02", "03"])
    p.add_argument("--max-trials", type=int, default=None)
    args = p.parse_args()

    root = Path(args.data_dir)
    data = {s: load_subject_trials(root, s, args.max_trials) for s in args.subjects}
    subj_pairs = [(a, b) for i, a in enumerate(args.subjects)
                  for b in args.subjects[i + 1:]]

    print(f"{'variant':<34} {'d-prime':>8} {'same-corr':>10}  (mean over {len(subj_pairs)} subject pairs)")
    rows = []
    for name, cfg in VARIANTS.items():
        ds, cs = [], []
        for a, b in subj_pairs:
            va, Xa = data[a]
            vb, Xb = data[b]
            Pa = preprocess(Xa, **cfg)
            Pb = preprocess(Xb, **cfg)
            d, c = identify_pair(Pa, va, Pb, vb)
            ds.append(d); cs.append(c)
        rows.append((name, float(np.mean(ds)), float(np.mean(cs))))
        print(f"{name:<34} {np.mean(ds):8.4f} {np.mean(cs):10.4f}")

    best = max(rows, key=lambda r: r[1])
    print(f"\nBEST: {best[0]}  (d' = {best[1]:.4f})")


if __name__ == "__main__":
    main()
