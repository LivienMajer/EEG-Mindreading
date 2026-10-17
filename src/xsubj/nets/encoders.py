"""EEG encoders used by the cross-subject retrieval pipeline."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class _Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()


class _Log(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_min(1e-6))


class SubjectChannelMixing(nn.Module):
    """Learn a channel transform per training subject; average for unseen subjects."""

    def __init__(self, n_subjects: int, n_channels: int):
        super().__init__()
        self.W = nn.Parameter(torch.eye(n_channels).repeat(n_subjects, 1, 1))

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        weights = self.W.mean(dim=0).expand(len(x), -1, -1).clone()
        known = subjects >= 0
        if known.any():
            if (subjects[known] >= len(self.W)).any():
                raise ValueError("Subject index exceeds trained subject layers")
            weights[known] = self.W[subjects[known]]
        return torch.bmm(weights, x)


class EEGEncoder(nn.Module):
    """Map preprocessed EEG trials into the normalized video-embedding space."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        embed_dim: int,
        n_subjects: int,
        cfg: dict,
        temperature: float = 0.07,
        fs: float = 128.0,
        tmin: float = 0.0,
    ):
        super().__init__()
        del fs, tmin
        self.backbone_in = (n_channels, n_times)
        self.subject_layer = (
            SubjectChannelMixing(n_subjects, n_channels)
            if cfg.get("subject_layer", True) and n_subjects > 0
            else None
        )
        arch = cfg.get("arch", "eegnet").lower()
        dropout = float(cfg.get("dropout", 0.4))
        width = int(cfg.get("width", 16))

        if arch == "eegnet":
            temporal_kernel = min(max(3, n_times // 2), 64)
            self.backbone = nn.Sequential(
                nn.Conv2d(1, width, (1, temporal_kernel),
                          padding=(0, temporal_kernel // 2), bias=False),
                nn.BatchNorm2d(width),
                nn.Conv2d(width, width * 2, (n_channels, 1),
                          groups=width, bias=False),
                nn.BatchNorm2d(width * 2),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
                nn.Conv2d(width * 2, width * 2, (1, 16),
                          padding=(0, 8), groups=width * 2, bias=False),
                nn.Conv2d(width * 2, width * 2, 1, bias=False),
                nn.BatchNorm2d(width * 2),
                nn.ELU(),
                nn.AdaptiveAvgPool2d((1, 8)),
                nn.Flatten(),
            )
            hidden_dim = width * 2 * 8
        elif arch == "shallow":
            temporal_kernel = min(n_times, 25)
            self.backbone = nn.Sequential(
                nn.Conv2d(1, width, (1, temporal_kernel),
                          padding=(0, temporal_kernel // 2), bias=False),
                nn.Conv2d(width, width, (n_channels, 1), bias=False),
                nn.BatchNorm2d(width),
                _Square(),
                nn.AvgPool2d((1, 8)),
                _Log(),
                nn.AdaptiveAvgPool2d((1, 8)),
                nn.Flatten(),
            )
            hidden_dim = width * 8
        elif arch == "mlp":
            hidden = int(cfg.get("hidden_dim", 512))
            self.backbone = nn.Sequential(
                nn.Flatten(),
                nn.Linear(n_channels * n_times, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, hidden),
                nn.GELU(),
            )
            hidden_dim = hidden
        else:
            raise ValueError(
                f"Unknown EEG model architecture {arch!r}; "
                "choose eegnet, shallow, or mlp")

        self.project = nn.Linear(hidden_dim, embed_dim)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        if self.subject_layer is not None:
            x = self.subject_layer(x, subjects)
        x = self.backbone(x.unsqueeze(1))
        return F.normalize(self.project(x), dim=-1)
