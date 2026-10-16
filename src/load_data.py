"""Data loading utilities.

ADAPT THIS FILE to the organizers' template scripts / data format once known.
The functions below assume a simple layout and should be adjusted.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class EEGDataset:
    """Container for epoched EEG data.

    Attributes:
        X: (n_trials, n_channels, n_times) float32
        y: (n_trials,) int labels (video id)
        subjects: (n_trials,) int/str subject id
        sfreq: sampling frequency in Hz
        ch_names: list of channel names
        times: (n_times,) time axis relative to video onset
    """

    X: np.ndarray
    y: np.ndarray
    subjects: np.ndarray
    sfreq: float
    ch_names: list[str]
    times: np.ndarray


def load_subject(eeg_file: str | Path, labels_file: str | Path | None = None,
                  sfreq: float = 250.0) -> EEGDataset:
    """Load one subject's epoched EEG.

    Flexible loader: supports .npy / .npz / MNE -epo files (.fif) / .csv.
    TODO: replace with the official loader from the template scripts.
    """
    eeg_file = Path(eeg_file)
    if eeg_file.suffix == ".npz":
        d = np.load(eeg_file)
        X = d["X"].astype(np.float32)
        y = d["y"] if "y" in d else np.zeros(len(X), dtype=int)
        sfreq = float(d["sfreq"]) if "sfreq" in d else sfreq
        ch_names = list(d["ch_names"]) if "ch_names" in d else [
            f"ch{i}" for i in range(X.shape[1])]
    elif eeg_file.suffix == ".npy":
        X = np.load(eeg_file).astype(np.float32)
        y = (np.load(labels_file) if labels_file and Path(labels_file).exists()
             else np.zeros(len(X), dtype=int))
        ch_names = [f"ch{i}" for i in range(X.shape[1])]
    elif eeg_file.suffix == ".fif":
        import mne
        epochs = mne.read_epochs(eeg_file, verbose="ERROR")
        X = epochs.get_data().astype(np.float32)
        y = epochs.events[:, 2]
        sfreq = epochs.info["sfreq"]
        ch_names = epochs.ch_names
    else:
        raise ValueError(f"Unsupported file type: {eeg_file}")

    n_trials, n_ch, n_t = X.shape
    times = (np.arange(n_t) / sfreq) * 1000.0  # ms since onset
    subjects = np.zeros(n_trials, dtype=int)
    return EEGDataset(X, np.asarray(y), subjects, sfreq, ch_names, times)


def load_all(data_dir: str | Path, sfreq: float = 250.0) -> EEGDataset:
    """Load and concatenate all subjects found under data_dir.

    Expects per-subject files like subject_01.npz / .npy / .fif.
    """
    import mne  # noqa: F401  (kept for .fif support above)
    data_dir = Path(data_dir)
    files = sorted(p for p in data_dir.rglob("subject*")
                   if p.suffix in {".npy", ".npz", ".fif"})
    if not files:
        raise FileNotFoundError(f"No subject files found under {data_dir}")

    ds_list = []
    for i, f in enumerate(files):
        ds = load_subject(f, sfreq=sfreq)
        ds.subjects[:] = i
        ds_list.append(ds)

    X = np.concatenate([d.X for d in ds_list])
    y = np.concatenate([d.y for d in ds_list])
    subjects = np.concatenate([d.subjects for d in ds_list])
    return EEGDataset(X, y, subjects, ds_list[0].sfreq,
                      ds_list[0].ch_names, ds_list[0].times)
