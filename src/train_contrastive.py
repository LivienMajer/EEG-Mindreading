"""Contrastive (InfoNCE / CLIP-style) training of the EEG encoder.

**Superseded:** the challenge-adapted version lives in
``src/contrastive_pipeline.py`` (branch `contrastive-alignment`), which
mirrors the exact candidate-matching protocol and uses the real challenge
data format. This module only keeps the reusable loss functions.
"""


from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def info_nce_loss(eeg_emb, vid_emb, temperature: float = 0.07):
    """Symmetric InfoNCE. eeg_emb, vid_emb: (B, D) L2-normalized torch tensors."""
    import torch
    import torch.nn.functional as F

    logits = eeg_emb @ vid_emb.T / temperature          # (B, B)
    labels = torch.arange(len(eeg_emb), device=eeg_emb.device)
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    return loss, logits.argmax(dim=1) == labels


def make_dataset(X: np.ndarray, y: np.ndarray, vid_embs: dict,
                 t_min: float, t_max: float, sfreq: float):
    """Torch dataset yielding (eeg, target_video_embedding).

    vid_embs: {video_id: (n_frames, dim)}; target is the mean frame embedding.
    TODO: switch to per-frame positive mining once the simple version works.
    """
    import torch
    from torch.utils.data import Dataset

    vid_ids = sorted(vid_embs.keys())
    id_to_idx = {v: i for i, v in enumerate(vid_ids)}
    V = np.stack([vid_embs[v].mean(axis=0) for v in vid_ids])
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    from .preprocess import crop_window
    Xc = crop_window(X, sfreq, t_min, t_max)
    y_idx = np.array([id_to_idx.get(int(v), -1) for v in y])
    keep = y_idx >= 0
    Xc, y_idx = Xc[keep], y_idx[keep]

    class _DS(Dataset):
        def __len__(self):
            return len(Xc)

        def __getitem__(self, i):
            return torch.from_numpy(Xc[i]), torch.from_numpy(V[y_idx[i]])

    return _DS(), Xc, y_idx


def train(X: np.ndarray, y: np.ndarray, vid_embs: dict, sfreq: float,
          t_min: float = 0.0, t_max: float = 3.0, embed_dim: int = 512,
          batch_size: int = 64, epochs: int = 50, lr: float = 3e-4,
          temperature: float = 0.07, device: str = "cpu",
          model_path: str | Path = "models/eeg_encoder.pt"):
    """Train EEGNet-style encoder with InfoNCE; save weights."""
    import torch
    from torch.utils.data import DataLoader
    from .eeg_encoder import build_eegnet

    ds, Xc, _ = make_dataset(X, y, vid_embs, t_min, t_max, sfreq)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    model = build_eegnet(Xc.shape[1], Xc.shape[2], embed_dim=embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(epochs):
        model.train()
        total, correct, n = 0.0, 0, 0
        for xb, vb in dl:
            xb, vb = xb.to(device).float(), vb.to(device).float()
            ze = model(xb)
            ze = ze / (ze.norm(dim=-1, keepdim=True) + 1e-8)
            loss, hits = info_nce_loss(ze, vb, temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item() * len(xb)
            correct += hits.sum().item(); n += len(xb)
        print(f"epoch {epoch + 1:3d} | loss {total / n:.4f} "
              f"| batch top-1 {correct / n:.3f}")

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_path)
    print(f"saved -> {model_path}")
    return model


if __name__ == "__main__":
    raise SystemExit(
        "Use src.contrastive_pipeline (challenge-adapted, branch "
        "'contrastive-alignment') instead of this module.")
