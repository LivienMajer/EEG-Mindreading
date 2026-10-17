"""EEG encoders used by the cross-subject retrieval pipeline."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

MIN_TEMPERATURE = 0.01
MAX_TEMPERATURE = 0.10


class _Square(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.square()


class _Log(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.log(x.clamp_min(1e-6))


class ChannelAttention(nn.Module):
    """Squeeze-and-excitation over learned EEG feature channels."""

    def __init__(self, num_channels: int, reduction: int = 4):
        super().__init__()
        if num_channels < 1 or reduction < 1:
            raise ValueError("Channel count and attention reduction must be positive")
        hidden_channels = max(1, num_channels // reduction)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(num_channels, hidden_channels, bias=False),
            nn.ReLU(),
            nn.Linear(hidden_channels, num_channels, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.fc(self.pool(x)).unsqueeze(-1).unsqueeze(-1)
        return x * weights


class MultiScaleTemporalConv(nn.Module):
    """Parallel temporal filters, scaled from a reference sampling rate."""

    def __init__(
        self,
        out_channels: int,
        kernel_sizes: tuple[int, ...],
        reference_fs: float,
        fs: float,
    ):
        super().__init__()
        if not kernel_sizes or any(size < 1 for size in kernel_sizes):
            raise ValueError("Multi-scale temporal kernels must be positive")
        if reference_fs <= 0 or fs <= 0:
            raise ValueError("Sampling rates must be positive")
        scaled_sizes = []
        for size in kernel_sizes:
            scaled = max(1, int(round(size * fs / reference_fs)))
            if scaled % 2 == 0:
                scaled += 1
            scaled_sizes.append(scaled)
        self.kernel_sizes = tuple(scaled_sizes)
        self.branches = nn.ModuleList([
            nn.Conv2d(
                1, out_channels, (1, kernel_size),
                padding=(0, kernel_size // 2), bias=False)
            for kernel_size in self.kernel_sizes
        ])

    @property
    def out_channels(self) -> int:
        return sum(branch.out_channels for branch in self.branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.cat([branch(x) for branch in self.branches], dim=1)


class PatchTransformerBackbone(nn.Module):
    """Encode overlapping EEG temporal patches with a Transformer."""

    def __init__(
        self,
        n_channels: int,
        n_times: int,
        fs: float,
        cfg: dict,
        dropout: float,
    ):
        super().__init__()
        reference_fs = float(cfg.get("patch_reference_fs", 250.0))
        if reference_fs <= 0 or fs <= 0:
            raise ValueError("Patch-transformer sampling rates must be positive")
        patch_length = max(
            1, int(round(int(cfg.get("patch_length", 50)) * fs / reference_fs)))
        patch_stride = max(
            1, int(round(int(cfg.get("patch_stride", 25)) * fs / reference_fs)))
        if patch_length > n_times:
            raise ValueError(
                f"Patch length {patch_length} exceeds EEG window "
                f"({n_times} samples)")
        n_patches = (n_times - patch_length) // patch_stride + 1
        model_dim = int(cfg.get("transformer_dim", 256))
        n_heads = int(cfg.get("transformer_heads", 8))
        n_layers = int(cfg.get("transformer_layers", 3))
        feedforward_dim = int(
            cfg.get("transformer_feedforward_dim", model_dim * 4))
        if model_dim < 1 or n_heads < 1 or model_dim % n_heads:
            raise ValueError(
                "transformer_dim must be positive and divisible by "
                "transformer_heads")
        if n_layers < 1 or feedforward_dim < 1:
            raise ValueError(
                "Transformer layer count and feedforward dimension must be positive")

        self.patch_length = patch_length
        self.patch_stride = patch_stride
        self.num_patches = n_patches
        self.patch_projection = nn.Conv1d(
            n_channels, model_dim, kernel_size=patch_length,
            stride=patch_stride)
        self.position_embedding = nn.Parameter(
            torch.empty(1, n_patches, model_dim))
        nn.init.trunc_normal_(self.position_embedding, std=0.02)
        self.input_dropout = nn.Dropout(dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=n_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.output_norm = nn.LayerNorm(model_dim)
        self.pool_scores = nn.Linear(model_dim, 1)
        self.output_dim = model_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_projection(x).transpose(1, 2)
        tokens = self.input_dropout(tokens + self.position_embedding)
        tokens = self.output_norm(self.transformer(tokens))
        weights = torch.softmax(self.pool_scores(tokens), dim=1)
        return torch.sum(weights * tokens, dim=1)


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
        head_dims: dict[str, int] | None = None,
    ):
        super().__init__()
        if (not math.isfinite(temperature)
                or not MIN_TEMPERATURE <= temperature <= MAX_TEMPERATURE):
            raise ValueError(
                "temperature must be between "
                f"{MIN_TEMPERATURE} and {MAX_TEMPERATURE}")
        del tmin
        self.backbone_in = (n_channels, n_times)
        self.subject_layer = (
            SubjectChannelMixing(n_subjects, n_channels)
            if cfg.get("subject_layer", True) and n_subjects > 0
            else None
        )
        arch = cfg.get("arch", "eegnet").lower()
        self.arch = arch
        dropout = float(cfg.get("dropout", 0.4))
        width = int(cfg.get("width", 16))
        channel_attention = bool(cfg.get("channel_attention", False))
        attention_reduction = int(cfg.get("attention_reduction", 4))
        multi_scale_temporal = bool(cfg.get("multi_scale_temporal", False))
        if channel_attention and arch not in ("eegnet", "shallow"):
            raise ValueError(
                "model.channel_attention requires a convolutional architecture")
        if multi_scale_temporal and arch != "eegnet":
            raise ValueError(
                "model.multi_scale_temporal currently requires arch=eegnet")
        if arch == "patch_transformer" and multi_scale_temporal:
            raise ValueError(
                "model.multi_scale_temporal is specific to arch=eegnet")

        if arch == "eegnet":
            if multi_scale_temporal:
                kernels = tuple(
                    int(kernel) for kernel in cfg.get(
                        "multi_scale_kernels", (64, 32, 16)))
                temporal = MultiScaleTemporalConv(
                    width, kernels,
                    float(cfg.get("multi_scale_reference_fs", 250.0)),
                    fs)
                temporal_channels = temporal.out_channels
                layers: list[nn.Module] = [
                    temporal,
                    nn.BatchNorm2d(temporal_channels),
                ]
            else:
                temporal_kernel = min(max(3, n_times // 2), 64)
                temporal_channels = width
                layers = [
                    nn.Conv2d(1, width, (1, temporal_kernel),
                              padding=(0, temporal_kernel // 2), bias=False),
                    nn.BatchNorm2d(width),
                ]
            if channel_attention:
                layers.append(ChannelAttention(
                    temporal_channels, attention_reduction))
            if multi_scale_temporal:
                layers.extend([
                    nn.Conv2d(
                        temporal_channels, temporal_channels * 2,
                        (n_channels, 1), groups=temporal_channels,
                        bias=False),
                    nn.Conv2d(temporal_channels * 2, width * 2, 1, bias=False),
                    nn.BatchNorm2d(width * 2),
                ])
            else:
                layers.extend([
                    nn.Conv2d(width, width * 2, (n_channels, 1),
                              groups=width, bias=False),
                    nn.BatchNorm2d(width * 2),
                ])
            layers.extend([
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
            ])
            self.backbone = nn.Sequential(*layers)
            hidden_dim = width * 2 * 8
        elif arch == "shallow":
            temporal_kernel = min(n_times, 25)
            layers = [
                nn.Conv2d(1, width, (1, temporal_kernel),
                          padding=(0, temporal_kernel // 2), bias=False),
            ]
            if channel_attention:
                layers.append(ChannelAttention(width, attention_reduction))
            layers.extend([
                nn.Conv2d(width, width, (n_channels, 1), bias=False),
                nn.BatchNorm2d(width),
                _Square(),
                nn.AvgPool2d((1, 8)),
                _Log(),
                nn.AdaptiveAvgPool2d((1, 8)),
                nn.Flatten(),
            ])
            self.backbone = nn.Sequential(*layers)
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
        elif arch == "patch_transformer":
            self.backbone = PatchTransformerBackbone(
                n_channels, n_times, fs, cfg, dropout)
            hidden_dim = self.backbone.output_dim
        else:
            raise ValueError(
                f"Unknown EEG model architecture {arch!r}; "
                "choose eegnet, shallow, mlp, or patch_transformer")

        self.dual_projection = bool(cfg.get("dual_projection", False))
        self.gated_fusion = bool(cfg.get("gated_fusion", False))
        self.multi_head = bool(cfg.get("multi_head", False))
        if sum((self.dual_projection, self.gated_fusion, self.multi_head)) > 1:
            raise ValueError(
                "dual_projection, gated_fusion, and multi_head are separate "
                "target modes; enable only one")
        if self.multi_head and not head_dims:
            raise ValueError("multi_head requires embedding dimensions for each head")
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
        self.head_names = tuple(head_dims or {})
        self.head_projects = nn.ModuleDict({
            name: nn.Linear(hidden_dim, dim)
            for name, dim in (head_dims or {}).items()
        })
        self.project = (
            None if self.multi_head else nn.Linear(hidden_dim, output_dim))
        self.audio_project = (
            nn.Linear(hidden_dim, output_dim) if self.dual_projection else None
        )
        prototype_source_dim = (
            int(head_dims[self.head_names[0]])
            if self.multi_head else output_dim)
        self.eeg_prototype_project = (
            nn.Linear(prototype_source_dim, n_channels * n_times)
            if cfg.get("eeg_prototype_head", False) else None)
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

    def clamp_logit_scale_(self) -> None:
        with torch.no_grad():
            self.logit_scale.clamp_(
                min=math.log(1.0 / MAX_TEMPERATURE),
                max=math.log(1.0 / MIN_TEMPERATURE),
            )

    def bounded_logit_scale(self) -> torch.Tensor:
        self.clamp_logit_scale_()
        return self.logit_scale.exp()

    def project_eeg_prototype(self, embedding: torch.Tensor) -> torch.Tensor:
        if self.eeg_prototype_project is None:
            raise ValueError("EEG prototype projection head is not enabled")
        return F.normalize(self.eeg_prototype_project(embedding), dim=-1)

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
        if self.arch == "patch_transformer":
            return self.backbone(x)
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

    def forward_heads(
        self, x: torch.Tensor, subjects: torch.Tensor
    ) -> dict[str, torch.Tensor]:
        if not self.multi_head:
            raise ValueError("Separate prediction heads are not enabled")
        features = self._features(x, subjects)
        return {
            name: F.normalize(project(features), dim=-1)
            for name, project in self.head_projects.items()
        }

    def forward(self, x: torch.Tensor, subjects: torch.Tensor) -> torch.Tensor:
        if self.multi_head or self.project is None:
            raise ValueError(
                "Use forward_heads() when the encoder has multiple heads")
        if self.dual_projection:
            video, audio = self.forward_modalities(x, subjects)
            return F.normalize(torch.cat((video, audio), dim=-1), dim=-1)
        return F.normalize(self.project(self._features(x, subjects)), dim=-1)
