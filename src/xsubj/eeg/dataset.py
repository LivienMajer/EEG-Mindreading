"""Torch dataset over preprocessed trials, plus train-time augmentation."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset, Sampler

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


class CrossSubjectBatchSampler(Sampler[list[int]]):
    """Pack complete video groups so every batch contains cross-subject positives."""

    def __init__(self, trials: Trials, batch_size: int):
        if batch_size < 2:
            raise ValueError("Cross-subject batches require batch_size >= 2")
        groups: dict[int, list[int]] = {}
        for index, video_id in enumerate(trials.y):
            groups.setdefault(int(video_id), []).append(index)
        missing_pairs = [
            video_id for video_id, indices in groups.items()
            if len(np.unique(trials.subj[indices])) < 2
        ]
        if missing_pairs:
            raise ValueError(
                "Cross-subject contrast requires at least two subjects per "
                f"training video; {len(missing_pairs)} videos lack a pair")
        self.groups = list(groups.values())
        self.batch_size = batch_size

    def __iter__(self):
        order = np.random.permutation(len(self.groups))
        batch: list[int] = []
        for group_index in order:
            group = self.groups[int(group_index)]
            if batch and len(batch) >= self.batch_size:
                yield batch
                batch = []
            batch.extend(group)
        if batch:
            yield batch

    def __len__(self) -> int:
        return max(1, int(np.ceil(sum(map(len, self.groups)) / self.batch_size)))


class TrialDataset(Dataset):
    """Training trials with optional augmentation.

    xsubj_avg {prob, k}: with probability `prob`, a trial is replaced by the average of itself and
    k-1 random trials of the *same video* from other subjects (each subject saw a video once),
    rescaled to the original trial's std. Averaging keeps the stimulus-locked response that subjects
    share and cancels the rest, which mimics the cleaner (averaged) test recordings.

    same_video_mixup interpolates with one other subject's trial for the same
    video, preserving the target while exposing the model to cross-subject
    variation.
    """

    def __init__(
        self,
        trials: Trials,
        aug: dict | None = None,
        fs: float = 128.0,
        xsubj_avg: dict | None = None,
        eeg_prototype_targets: np.ndarray | None = None,
    ):
        self.t, self.aug, self.fs, self.avg = trials, aug, fs, xsubj_avg
        if eeg_prototype_targets is not None and len(eeg_prototype_targets) != len(trials):
            raise ValueError(
                "EEG prototype targets must match the number of training trials")
        self.eeg_prototype_targets = eeg_prototype_targets
        self.same_video: dict[int, np.ndarray] = {}
        if self.avg or (self.aug and self.aug.get("same_video_mixup_prob", 0)):
            order = np.argsort(trials.y, kind="stable")
            vids, starts = np.unique(trials.y[order], return_index=True)
            groups = np.split(order, starts[1:])
            self.same_video = {int(v): g for v, g in zip(vids, groups)}
        if self.aug:
            mixup_alpha = float(self.aug.get("same_video_mixup_alpha", 0.4))
            if mixup_alpha <= 0:
                raise ValueError("same_video_mixup_alpha must be positive")
            mixup_prob = float(self.aug.get("same_video_mixup_prob", 0.0))
            if not 0 <= mixup_prob <= 1:
                raise ValueError("same_video_mixup_prob must be in [0, 1]")

    def __len__(self):
        return len(self.t)

    def __getitem__(self, i):
        x = torch.from_numpy(self.t.X[i])
        if self.aug and np.random.rand() < self.aug.get(
                "same_video_mixup_prob", 0.0):
            same_video = self.same_video[int(self.t.y[i])]
            other_subjects = same_video[
                self.t.subject[same_video] != self.t.subject[i]]
            if len(other_subjects):
                j = int(np.random.choice(other_subjects))
                alpha = float(self.aug.get("same_video_mixup_alpha", 0.4))
                lam = float(np.random.beta(alpha, alpha))
                x = lam * x + (1.0 - lam) * torch.from_numpy(self.t.X[j])
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
        sample = (x, int(self.t.y[i]), int(self.t.subj[i]))
        if self.eeg_prototype_targets is None:
            return sample
        return (*sample, torch.from_numpy(self.eeg_prototype_targets[i]))


def augment(x: torch.Tensor, aug: dict, fs: float) -> torch.Tensor:
    time_shift = float(aug.get("time_shift", 0))
    if time_shift < 0:
        raise ValueError("time_shift must be non-negative")
    max_shift = int(round(time_shift * fs))
    if max_shift:
        x = torch.roll(x, int(np.random.randint(-max_shift, max_shift + 1)), dims=-1)

    time_mask_prob = float(aug.get("time_mask_prob", 0.0))
    if not 0 <= time_mask_prob <= 1:
        raise ValueError("time_mask_prob must be in [0, 1]")
    if np.random.rand() < time_mask_prob:
        min_mask = int(round(float(aug.get("time_mask_min_ms", 50)) * fs / 1000))
        max_mask = int(round(float(aug.get("time_mask_max_ms", 100)) * fs / 1000))
        if min_mask < 1 or max_mask < min_mask:
            raise ValueError(
                "time mask durations must satisfy 0 < min_ms <= max_ms")
        mask_length = min(int(np.random.randint(min_mask, max_mask + 1)), x.shape[-1])
        start = int(np.random.randint(0, x.shape[-1] - mask_length + 1))
        x = x.clone()
        x[..., start:start + mask_length] = 0

    channel_mask_prob = float(aug.get("channel_mask_prob", 0.0))
    if not 0 <= channel_mask_prob <= 1:
        raise ValueError("channel_mask_prob must be in [0, 1]")
    if np.random.rand() < channel_mask_prob:
        min_channels = int(aug.get("channel_mask_min", 2))
        max_channels = int(aug.get("channel_mask_max", 4))
        if min_channels < 1 or max_channels < min_channels:
            raise ValueError(
                "channel mask counts must satisfy 1 <= min <= max")
        n_mask = min(
            int(np.random.randint(min_channels, max_channels + 1)),
            x.shape[0])
        channels = np.random.choice(x.shape[0], n_mask, replace=False)
        x = x.clone()
        x[channels, :] = 0

    if aug.get("channel_dropout"):
        x = x * (torch.rand(x.shape[0], 1) > aug["channel_dropout"])
    if aug.get("scale"):
        x = x * (1 + (2 * np.random.rand() - 1) * aug["scale"])
    if aug.get("noise_std"):
        x = x + aug["noise_std"] * torch.randn_like(x)
    return x
