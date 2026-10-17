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
        visual_dim: int | None = None,
        audio_dim: int | None = None,
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

        self.dual_projection = bool(cfg.get("dual_projection", False))
        self.gated_fusion = bool(cfg.get("gated_fusion", False))
        if self.dual_projection and self.gated_fusion:
            raise ValueError(
                "dual_projection and gated_fusion are separate target modes; "
                "enable only one")
        if self.dual_projection and embed_dim % 2:
            raise ValueError(
                "Dual projection requires an even embedding dimension "
                "for equal-sized video and audio spaces")
        if self.gated_fusion and (
                visual_dim is None or audio_dim is None
                or visual_dim < 1 or audio_dim < 1):
            raise ValueError(
                "Gated fusion requires separate visual and audio feature banks")
        output_dim = (
            int(cfg.get("fusion_dim", 512)) if self.gated_fusion else
            embed_dim // 2 if self.dual_projection else embed_dim)
        self.project = nn.Linear(hidden_dim, output_dim)
        self.audio_project = (
            nn.Linear(hidden_dim, output_dim) if self.dual_projection else None
        )
        self.target_visual_proj = (
            nn.Linear(visual_dim, output_dim, bias=False)
            if self.gated_fusion and visual_dim is not None else None)
        self.target_audio_proj = (
            nn.Linear(audio_dim, output_dim, bias=False)
            if self.gated_fusion and audio_dim is not None else None)
        self.gate_layer = (
            nn.Linear(visual_dim + audio_dim, output_dim)
            if self.gated_fusion
            and visual_dim is not None
            and audio_dim is not None else None)
        self.logit_scale = nn.Parameter(
            torch.tensor(math.log(1.0 / temperature), dtype=torch.float32))

    def fuse_video_embeddings(
        self, visual: torch.Tensor, audio: torch.Tensor
    ) -> torch.Tensor:
        if (not self.gated_fusion or self.target_visual_proj is None
                or self.target_audio_proj is None or self.gate_layer is None):
            raise ValueError("Gated video fusion is not enabled")
        gate = torch.sigmoid(self.gate_layer(torch.cat((visual, audio), dim=-1)))
        fused = (
            gate * self.target_visual_proj(visual)
            + (1.0 - gate) * self.target_audio_proj(audio))
        return F.normalize(fused, p=2, dim=-1)

    def _features(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        if self.subject_layer is not None:
            x = self.subject_layer(x, subjects)
        return self.backbone(x.unsqueeze(1))

    def forward_modalities(
        self, x: torch.Tensor, subjects: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not self.dual_projection or self.audio_project is None:
            raise ValueError("Separate modality outputs require dual_projection")
        features = self._features(x, subjects)
        return (
            F.normalize(self.project(features), dim=-1),
            F.normalize(self.audio_project(features), dim=-1),
        )

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        if self.dual_projection:
            video, audio = self.forward_modalities(x, subjects)
            return F.normalize(torch.cat((video, audio), dim=-1), dim=-1)
        return F.normalize(self.project(self._features(x, subjects)), dim=-1)
