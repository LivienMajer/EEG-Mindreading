"""Torch dataset over preprocessed trials, plus train-time augmentation."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from src.xsubj.eeg.adapter import SubjectData
from src.xsubj.video.embeddings import VideoBank


class Trials:
    """Flat arrays over several subjects. `subj` indexes into the model's subject list (-1 = unseen)."""

    def __init__(self, data: list[SubjectData], bank: VideoBank, subject_list: list[str]):
        self.X = np.concatenate([d.X for d in data])
        self.video_id = np.concatenate([d.video_id for d in data])
        self.y = bank.lookup(self.video_id)
        self.subject = np.concatenate([[d.subject] * len(d.X) for d in data])
        lut = {s: i for i, s in enumerate(subject_list)}
        self.subj = np.array([lut.get(s, -1) for s in self.subject])

    def __len__(self):
        return len(self.X)

    def subset(self, mask: np.ndarray) -> "Trials":
        t = object.__new__(Trials)
        for k in ("X", "video_id", "y", "subject", "subj"):
            setattr(t, k, getattr(self, k)[mask])
        return t


class TrialDataset(Dataset):
    """Training trials with optional augmentation.

    xsubj_avg {prob, k}: with probability `prob`, a trial is replaced by the average of itself and
    k-1 random trials of the *same video* from other subjects (each subject saw a video once),
    rescaled to the original trial's std. Averaging keeps the stimulus-locked response that subjects
    share and cancels the rest, which mimics the cleaner (averaged) test recordings.
    """

    def __init__(self, trials: Trials, aug: dict | None = None, fs: float = 128.0, xsubj_avg: dict | None = None):
        self.t, self.aug, self.fs, self.avg = trials, aug, fs, xsubj_avg
        if self.avg:
            order = np.argsort(trials.y, kind="stable")
            vids, starts = np.unique(trials.y[order], return_index=True)
            groups = np.split(order, starts[1:])
            self.same_video = {int(v): g for v, g in zip(vids, groups)}

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        x = torch.from_numpy(self.t.X[i])
        if self.avg and np.random.rand() < self.avg["prob"]:
            others = self.same_video[int(self.t.y[i])]
            others = others[others != i]
            k = min(int(np.random.choice(self.avg["k"])) - 1, len(others))
            if k > 0:
                pick = np.random.choice(others, k, replace=False)
                avg = (x + torch.from_numpy(self.t.X[pick]).sum(0)) / (k + 1)
                x = avg * (x.std() / (avg.std() + 1e-8))
        if self.aug:
            x = augment(x, self.aug, self.fs)
        return x, int(self.t.y[i]), int(self.t.subj[i])


def augment(x: torch.Tensor, aug: dict, fs: float) -> torch.Tensor:
    max_shift = int(aug.get("time_shift", 0) * fs)
    if max_shift:
        x = torch.roll(x, int(np.random.randint(-max_shift, max_shift + 1)), dims=-1)
    if aug.get("channel_dropout"):
        x = x * (torch.rand(x.shape[0], 1) > aug["channel_dropout"])
    if aug.get("scale"):
        x = x * (1 + (2 * np.random.rand() - 1) * aug["scale"])
    if aug.get("noise_std"):
        x = x + aug["noise_std"] * torch.randn_like(x)
    return x
