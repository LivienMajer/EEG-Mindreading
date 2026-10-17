"""EEG encoder: convert EEG trials to embeddings aligned with video space."""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# 1. Classic features + linear projection (fast baseline)
# ---------------------------------------------------------------------------

class LinearProjection:
    """Ridge regression mapping EEG features -> video embedding space."""

    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.W = None

    def fit(self, X_feat: np.ndarray, targets: np.ndarray):
        """X_feat: (n_trials, n_features), targets: (n_trials, embed_dim)."""
        from sklearn.linear_model import Ridge
        self.W = Ridge(alpha=self.alpha).fit(X_feat, targets)
        return self

    def transform(self, X_feat: np.ndarray) -> np.ndarray:
        Z = self.W.predict(X_feat).astype(np.float32)
        norm = np.linalg.norm(Z, axis=-1, keepdims=True) + 1e-8
        return Z / norm

    def __call__(self, X_feat):
        return self.transform(X_feat)


# ---------------------------------------------------------------------------
# 2. Neural EEG encoder (EEGNet-style) in PyTorch
# ---------------------------------------------------------------------------

def build_eegnet(n_channels: int, n_times: int, embed_dim: int = 512,
                 dropout: float = 0.5, k1: int = 64, k2: int = 32):
    """Small EEGNet-style CNN + projection head to the embedding space.

    k1/k2: temporal kernel sizes (samples @100 Hz). Defaults 64/32 fit
    ~300-sample windows; for the short 50-350 ms window (30 samples) use
    k1=16, k2=8.
    """
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.firstconv = nn.Sequential(
                nn.Conv2d(1, 8, (1, k1), padding=(0, k1 // 2), bias=False),
                nn.BatchNorm2d(8))
            self.depthwise = nn.Sequential(
                nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
                nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(dropout))
            self.sepconv = nn.Sequential(
                nn.ZeroPad2d((k2 // 2 - 1, k2 // 2, 0, 1)),
                nn.Conv2d(16, 16, (1, k2), bias=False),
                nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(dropout))
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.LazyLinear(256), nn.ELU(),
                nn.Linear(256, embed_dim))

        def forward(self, x):                       # x: (B, n_ch, n_times)
            x = x.unsqueeze(1)                      # (B, 1, n_ch, n_times)
            x = self.firstconv(x)
            x = self.depthwise(x)
            x = self.sepconv(x)
            return self.head(x)

    net = EEGNet()
    # initialize LazyLinear with a dummy forward pass
    with torch.no_grad():
        net(torch.zeros(1, n_channels, n_times))
    return net
