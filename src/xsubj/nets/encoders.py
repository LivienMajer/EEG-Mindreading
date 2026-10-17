"""EEG encoders that map a trial (channels x samples) onto the unit sphere of the video embedding space."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SubjectLayer(nn.Module):
    """Per-subject channel mixing (identity-initialised). Unseen subjects (index -1) use the mean matrix."""

    def __init__(self, n_subjects: int, n_channels: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_channels).repeat(n_subjects, 1, 1))

    def forward(self, x, subj):
        W = torch.cat([self.W, self.W.mean(0, keepdim=True)])  # last slot = unseen subject
        return torch.einsum("bcd,bdt->bct", W[subj], x)        # subj == -1 picks the mean


class ShallowBackbone(nn.Module):
    """ShallowConvNet (Schirrmeister et al. 2017): temporal conv -> spatial conv -> pool."""

    def __init__(self, n_channels, n_times, n_filters=40, temporal_kernel=25, pool=75,
                 pool_stride=15, nonlin="elu", dropout=0.5):
        super().__init__()
        self.temporal = nn.Conv2d(1, n_filters, (1, temporal_kernel), padding=(0, temporal_kernel // 2))
        self.spatial = nn.Conv2d(n_filters, n_filters, (n_channels, 1), bias=False)
        self.bn = nn.BatchNorm2d(n_filters)
        self.pool = nn.AvgPool2d((1, pool), (1, pool_stride))
        self.nonlin = nonlin
        self.drop = nn.Dropout(dropout)
        with torch.no_grad():
            self.out_dim = self._features(torch.zeros(1, n_channels, n_times)).shape[1]

    def _features(self, x):
        x = self.bn(self.spatial(self.temporal(x.unsqueeze(1))))
        if self.nonlin == "square_log":
            x = torch.log(torch.clamp(self.pool(x * x), min=1e-6))
        else:
            x = self.pool(F.elu(x))
        return x.flatten(1)

    def forward(self, x):
        return self.drop(self._features(x))


class EEGNetBackbone(nn.Module):
    """EEGNet-8,2 (Lawhern et al. 2018), kernel lengths set for ~128 Hz input."""

    def __init__(self, n_channels, n_times, F1=8, D=2, F2=16, kernel=64, dropout=0.5, **_):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, F1, (1, kernel), padding=(0, kernel // 2), bias=False),
            nn.BatchNorm2d(F1),
            nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),
            nn.BatchNorm2d(F1 * D), nn.ELU(), nn.AvgPool2d((1, 4)), nn.Dropout(dropout),
            nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),
            nn.Conv2d(F1 * D, F2, 1, bias=False),
            nn.BatchNorm2d(F2), nn.ELU(), nn.AvgPool2d((1, 8)), nn.Dropout(dropout),
            nn.Flatten(),
        )
        with torch.no_grad():
            self.out_dim = self.net(torch.zeros(1, 1, n_channels, n_times)).shape[1]

    def forward(self, x):
        return self.net(x.unsqueeze(1))


class FlatBackbone(nn.Module):
    """No convolutions: the trial flattened to channels x samples (what ridge sees, minus binning)."""

    def __init__(self, n_channels, n_times, dropout=0.0, **_):
        super().__init__()
        self.drop = nn.Dropout(dropout)
        self.out_dim = n_channels * n_times

    def forward(self, x):
        return self.drop(x.flatten(1))


class EEGEncoder(nn.Module):
    """Optional input crop (model.crop, seconds after onset) -> subject layer -> backbone -> head.

    arch: shallow | eegnet (convolutional), mlp (flattened input + MLP head),
          linear (flattened input + one linear layer: ridge's model class trained with InfoNCE).
    """

    def __init__(self, n_channels, n_times, emb_dim, n_subjects, m: dict, temperature=0.07,
                 fs: float = 100.0, tmin: float = 0.0):
        super().__init__()
        self.backbone_in = (n_channels, n_times)
        if m.get("crop"):
            self.crop = (max(0, int(round((m["crop"][0] - tmin) * fs))),
                         min(n_times, int(round((m["crop"][1] - tmin) * fs))))
        else:
            self.crop = (0, n_times)
        n_t = self.crop[1] - self.crop[0]
        self.subject_layer = SubjectLayer(n_subjects, n_channels) if m["subject_layer"] else None
        kw = dict(dropout=m["dropout"])
        if m["arch"] == "shallow":
            kw.update(n_filters=m["n_filters"], temporal_kernel=m["temporal_kernel"], pool=m["pool"],
                      pool_stride=m["pool_stride"], nonlin=m["nonlin"])
            self.backbone = ShallowBackbone(n_channels, n_t, **kw)
        elif m["arch"] == "eegnet":
            self.backbone = EEGNetBackbone(n_channels, n_t, **kw)
        elif m["arch"] in ("mlp", "linear"):
            self.backbone = FlatBackbone(n_channels, n_t, dropout=m.get("input_dropout", 0.0))
        else:
            raise ValueError(m["arch"])
        if m["arch"] == "linear":
            self.head = nn.Linear(self.backbone.out_dim, emb_dim)
        else:
            self.head = nn.Sequential(
                nn.Linear(self.backbone.out_dim, m["hidden"]), nn.GELU(), nn.Dropout(m["dropout"]),
                nn.Linear(m["hidden"], emb_dim),
            )
        self.logit_scale = nn.Parameter(torch.tensor(1 / temperature).log())

    def forward(self, x, subj):
        x = x[..., self.crop[0]: self.crop[1]]
        if self.subject_layer is not None:
            x = self.subject_layer(x, subj)
        return F.normalize(self.head(self.backbone(x)), dim=-1)
