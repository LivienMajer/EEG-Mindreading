"""Does it help to adapt Daniel-style ridges to the averaged test recordings?

Test recordings behave like averages of ~2 repetitions (eeg_pipeline/README.md): about twice
the signal-to-noise of the single training trials, and relatively less high-frequency
(non-phase-locked) power. The ridges choose their regularisation by cross-validation on
single training trials, i.e. for noisier input than they get at test time. Two ideas, tested
on the ridge votes of Daniel's pipeline (pooled + per-subject ridge, as src.ridge_member;
the bulk of fusion B's weight):

  1. alpha factor: refit each ridge with its cross-validated alpha times f (f < 1 = less
     regularisation, suited to cleaner input). f is a setting chosen on the dev test sets,
     which the rules allow ("validation and model selection").
  2. spectral matching, label-free and per split: low-pass both splits at a lower cutoff, or
     whiten each split's spectrum by its own mean amplitude spectrum (each split is
     normalised on its own, so test statistics never enter training).

Scored on sub-01..04 (rankings after per-candidate normalisation, as Daniel's submission);
the setting is chosen leave-one-subject-out and compared, paired, with the current ridge.

    python -m src.test_matching --emb models/xsubj/features/clip_vitb32.npz
    python -m src.test_matching --emb work/rsa_features/videomae_l8.npz
"""

from __future__ import annotations

import argparse
import copy

import numpy as np
from scipy import signal, stats
from sklearn.linear_model import Ridge

from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import similarity
from src.xsubj.nets.ridge import RidgeDecoder, eeg_features
from src.xsubj.utils import load_config
from src.xsubj.video.embeddings import VideoBank, _load_file

CANDIDATES = np.arange(1001, 1103)
DEV = ["sub-01", "sub-02", "sub-03", "sub-04"]
FACTORS = (0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
SPECTRA = ("none", "lowpass20", "lowpass12", "lowpass8", "whiten")


def spectral(X: np.ndarray, fs: float, kind: str) -> np.ndarray:
    """Applied to each split separately; statistics come only from that split."""
    if kind == "none":
        return X
    if kind.startswith("lowpass"):
        sos = signal.butter(4, float(kind[7:]), btype="lowpass", fs=fs, output="sos")
        return signal.sosfiltfilt(sos, X, axis=-1).astype(np.float32)
    F = np.fft.rfft(X, axis=-1)
    amp = np.sqrt((np.abs(F) ** 2).mean(axis=0, keepdims=True)) + 1e-8      # (1, ch, freq) of this split
    return np.fft.irfft(F / amp * amp.mean(axis=-1, keepdims=True), n=X.shape[-1], axis=-1).astype(np.float32)


def fit_scaled(r: dict, fs, tmin, X, Y, factor: float) -> RidgeDecoder:
    """RidgeCV as in Daniel's pipeline, then refit with alpha_ * factor."""
    dec = RidgeDecoder(r, fs, tmin).fit(X, Y)
    if factor != 1.0:
        scaler, cv = dec.model[0], dec.model[1]
        ridge = Ridge(alpha=cv.alpha_ * factor).fit(scaler.transform(eeg_features(X, fs, tmin, r)), Y)
        dec.model.steps[1] = ("ridge", ridge)
    return dec


def true_rank(ranked, y):
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--emb", required=True, help="video embedding file (ids + emb)")
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = load_preprocessed(cfg)
    fs, tmin = data[0].fs, data[0].tmin
    bank = VideoBank(*_load_file(args.emb), standardize=True)
    emb = lambda ids: bank.emb[bank.lookup(np.asarray(ids).astype(str))]
    cand = emb(CANDIDATES)

    res = {}                                            # (spectrum, factor) -> {subject: ranks}
    for kind in SPECTRA:
        d2 = [copy.copy(d) for d in data]
        for d in d2:
            d.X = spectral(d.X, fs, kind)
        trn = {d.subject: d for d in d2 if d.split == "trn"}
        tst = {d.subject: d for d in d2 if d.split == "tst" and d.subject in DEV}
        Xp = np.concatenate([d.X for d in trn.values()])
        Yp = emb(np.concatenate([d.video_id for d in trn.values()]))
        for f in FACTORS if kind == "none" else (1.0, 0.1):
            pooled = fit_scaled(cfg["ridge"], fs, tmin, Xp, Yp, f)
            res[(kind, f)] = {}
            for s, d in tst.items():
                own = fit_scaled(cfg["ridge"], fs, tmin, trn[s].X, emb(trn[s].video_id), f)
                sim = similarity(pooled.predict(d.X) + own.predict(d.X), cand, col_norm=True)
                ranked = CANDIDATES[np.argsort(-sim, axis=1, kind="stable")]
                res[(kind, f)][s] = true_rank(ranked, d.video_id.astype(int))
            r = np.concatenate([res[(kind, f)][s] for s in DEV])
            print(f"{kind:9s} alpha x{f:<5}: top-1 {np.mean(r == 1) * 100:5.2f}  top-5 {np.mean(r <= 5) * 100:5.2f}"
                  f"  mean rank {r.mean():.2f}   (pooled alpha {pooled.model[1].alpha if f != 1.0 else pooled.model[1].alpha_:g})",
                  flush=True)

    base = np.concatenate([res[("none", 1.0)][s] for s in DEV])
    held = []
    for h in DEV:
        dev = [s for s in DEV if s != h]
        best = max(res, key=lambda o: (np.mean([np.mean(res[o][s] == 1) for s in dev]),
                                       np.mean([np.mean(res[o][s] <= 5) for s in dev])))
        held.append(res[best][h])
        print(f"hold {h}: chose {best}")
    rn = np.concatenate(held)
    print(f"LOO-chosen setting: top-1 {np.mean(rn == 1) * 100:.2f}  top-5 {np.mean(rn <= 5) * 100:.2f}  mean rank {rn.mean():.2f}"
          f"  (current {np.mean(base == 1) * 100:.2f} / {np.mean(base <= 5) * 100:.2f} / {base.mean():.2f})")
    for k in (1, 5):
        g, l = int(np.sum((rn <= k) & (base > k))), int(np.sum((base <= k) & (rn > k)))
        print(f"  top-{k}: +{g}/-{l}  exact McNemar p = {stats.binomtest(g, g + l).pvalue if g + l else 1:.3f}")
    print(f"  rank better {np.sum(rn < base)} / worse {np.sum(rn > base)}"
          f"  Wilcoxon p = {stats.wilcoxon(rn, base).pvalue if np.any(rn != base) else 1:.2g}")


if __name__ == "__main__":
    main()
