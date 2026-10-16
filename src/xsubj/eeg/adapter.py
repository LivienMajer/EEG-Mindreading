"""The only module that knows the raw data format.

Every loader returns a list of `SubjectData`: epoched EEG (trials x channels x samples) of
one subject and one split, with the video id each trial showed (None where withheld).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from src.xsubj.utils import resolve


@dataclass
class SubjectData:
    subject: str
    X: np.ndarray                    # (n_trials, n_channels, n_samples), float32, volts or µV
    video_id: np.ndarray | None      # (n_trials,) str; None for unlabeled test data
    fs: float
    tmin: float = 0.0                # time of the first sample relative to video onset (s)
    ch_names: list[str] = field(default_factory=list)
    split: str = "trn"               # trn | tst; normalisation statistics are computed per split

    @property
    def key(self) -> str:
        return f"{self.subject}_{self.split}"


def load_subjects(cfg: dict, subjects: list[str] | None = None) -> list[SubjectData]:
    fmt = cfg["data"]["format"]
    if fmt == "synthetic":
        data = _load_synthetic(cfg["data"]["synthetic"])
    elif fmt == "npz":
        data = _load_npz(resolve(cfg["data"]["eeg_dir"]))
    elif fmt == "challenge":
        data = _load_challenge(resolve(cfg["data"]["challenge_dir"]))
    else:
        raise ValueError(f"unknown data.format {fmt!r}; add a loader in src/xsubj/eeg/adapter.py")
    if subjects is not None:
        data = [d for d in data if d.subject in subjects]
    return data


def _load_npz(eeg_dir: Path) -> list[SubjectData]:
    files = sorted(eeg_dir.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no .npz files in {eeg_dir}")
    out = []
    for f in files:
        z = np.load(f, allow_pickle=True)
        out.append(
            SubjectData(
                subject=f.stem,
                X=z["X"].astype(np.float32),
                video_id=z["video_id"].astype(str) if "video_id" in z else None,
                fs=float(z["fs"]),
                tmin=float(z["tmin"]) if "tmin" in z else 0.0,
                ch_names=list(z["ch_names"]) if "ch_names" in z else [],
            )
        )
    return out


def _load_challenge(root: Path) -> list[SubjectData]:
    """Official layout: trn/sub-XX_trn.npz ({video id: 17 x 1500}), tst/sub-XX_tst.npy
    (102 x 17 x 1500, shuffled), tst_stim/sub-XX_tst_stim.npy (withheld for sub-05/06).
    500 Hz, epochs start at video onset. Test trials keep their original order."""
    out = []
    for f in sorted((root / "trn").glob("sub-*_trn.npz")):
        subject = f.stem.removesuffix("_trn")
        z = np.load(f, allow_pickle=False)
        ids = sorted(z.files, key=int)
        out.append(SubjectData(subject, np.stack([z[k] for k in ids]).astype(np.float32),
                               np.array(ids), 500.0, 0.0, split="trn"))
        tst = root / "tst" / f"{subject}_tst.npy"
        if tst.exists():
            stim = root / "tst_stim" / f"{subject}_tst_stim.npy"
            y = np.load(stim, allow_pickle=False).astype(str) if stim.exists() else None
            out.append(SubjectData(subject, np.load(tst, allow_pickle=False).astype(np.float32),
                                   y, 500.0, 0.0, split="tst"))
    if not out:
        raise FileNotFoundError(f"no trn/sub-*_trn.npz under {root}")
    return out


# --------------------------------------------------------------------------- synthetic
# A toy generative model with the same structure as the real problem: each video has an
# embedding, a shared linear map turns it into the time courses of a few cortical
# sources, and each subject sees those sources through its own channel mixing plus
# coloured noise and line noise. With snr=0 there is no signal, so any decoder must sit
# at chance (leakage check).

def synthetic_video_embeddings(syn: dict) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(syn["seed"])
    emb = rng.standard_normal((syn["n_videos"], syn["emb_dim"])).astype(np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    ids = np.array([f"vid_{i:04d}" for i in range(syn["n_videos"])])
    return ids, emb


def _load_synthetic(syn: dict) -> list[SubjectData]:
    ids, emb = synthetic_video_embeddings(syn)
    rng = np.random.default_rng(syn["seed"] + 1)
    fs, n_ch, k = syn["fs"], syn["n_channels"], syn["n_sources"]
    t = np.arange(int(round((syn["tmax"] - syn["tmin"]) * fs))) / fs + syn["tmin"]

    # temporal basis: onset-locked bursts at typical visual latencies + a sustained part
    lat, freq, width = [0.1, 0.17, 0.3, 0.6], [10, 7, 4, 3], [0.04, 0.05, 0.08, 0.15]
    basis = [np.exp(-((t - l) ** 2) / (2 * w**2)) * np.cos(2 * np.pi * f * (t - l))
             for l, f, w in zip(lat, freq, width)]
    basis.append(((t > 0.1) & (t < 2.9)) * np.sin(2 * np.pi * 1.0 * t))
    basis = np.stack(basis)                                    # (n_basis, n_t)
    proj = rng.standard_normal((k * len(basis), emb.shape[1])) / np.sqrt(emb.shape[1])
    coef = (emb @ proj.T).reshape(len(ids), k, len(basis))     # (n_videos, k, n_basis)
    sources = coef @ basis                                      # (n_videos, k, n_t)

    shared_mixing = rng.standard_normal((n_ch, k))    # similar topographies across subjects...
    out = []
    for s in range(syn["n_subjects"]):
        mixing = shared_mixing + syn["subject_variability"] * rng.standard_normal((n_ch, k))  # ...but not identical
        order = rng.permutation(np.repeat(np.arange(len(ids)), syn["reps"]))
        signal = np.einsum("ck,nkt->nct", mixing, sources[order])
        noise = np.cumsum(rng.standard_normal((len(order), n_ch, len(t))), axis=-1)
        noise -= noise.mean(-1, keepdims=True)
        noise = np.einsum("cd,ndt->nct", rng.standard_normal((n_ch, n_ch)) / np.sqrt(n_ch), noise)
        noise /= noise.std()
        noise += 0.3 * np.sin(2 * np.pi * 50 * t + rng.uniform(0, 2 * np.pi, (len(order), n_ch, 1)))
        X = syn["snr"] * signal / signal.std() + noise
        X = X * rng.uniform(5, 20) * 1e-6 + rng.standard_normal((1, n_ch, 1)) * 2e-6  # µV scale + DC offsets
        out.append(SubjectData(f"sub-{s + 1:02d}", X.astype(np.float32), ids[order], float(fs),
                               syn["tmin"], [f"EEG{c:03d}" for c in range(n_ch)]))
    return out
