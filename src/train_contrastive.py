"""Contrastive (InfoNCE / CLIP-style) training of the EEG encoder.

Aligns EEG embeddings with video embeddings so that at test time we can
match an EEG trial to the right candidate video by cosine similarity.
"""

from __future__ import annotations

import argparse
import re
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

    Accepts either {video_id: (n_frames, dim)} or frame-level keys of the
    form "{video_id}_f{frame:02d}". Targets are mean frame embeddings.
    TODO: switch to per-frame positive mining once the simple version works.
    """
    import torch
    from torch.utils.data import Dataset

    frames_by_video = {}
    for key, embedding in vid_embs.items():
        key = str(key)
        frame_match = re.fullmatch(r"(\d+)_f\d+", key)
        video_key = frame_match.group(1) if frame_match else key
        try:
            video_id = int(video_key)
        except ValueError as exc:
            raise ValueError(
                f"Embedding key {key!r} must be a video ID or "
                "'<video_id>_f<frame>'") from exc

        embedding = np.asarray(embedding, dtype=np.float32)
        if embedding.ndim == 1:
            embedding = embedding[None, :]
        if embedding.ndim != 2:
            raise ValueError(
                f"Embedding {key!r} must have shape (dim,) or (frames, dim)")
        frames_by_video.setdefault(video_id, []).append(embedding)

    vid_ids = sorted(frames_by_video)
    id_to_idx = {video_id: i for i, video_id in enumerate(vid_ids)}
    V = np.stack([
        np.concatenate(frames_by_video[video_id], axis=0).mean(axis=0)
        for video_id in vid_ids
    ])
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
          t_min: float = 0.0, t_max: float = 3.0,
          embed_dim: int | None = None,
          batch_size: int = 64, epochs: int = 50, lr: float = 3e-4,
          temperature: float = 0.07, device: str = "cpu",
          model_path: str | Path = "models/eeg_encoder.pt"):
    """Train EEGNet-style encoder with InfoNCE; save weights."""
    import torch
    from torch.utils.data import DataLoader
    from .eeg_encoder import build_eegnet

    ds, Xc, _ = make_dataset(X, y, vid_embs, t_min, t_max, sfreq)
    if len(ds) == 0:
        raise ValueError(
            "No EEG trials matched the video IDs in the embedding file")
    target_dim = ds[0][1].numel()
    if embed_dim is not None and embed_dim != target_dim:
        raise ValueError(
            f"embed_dim={embed_dim} does not match video embedding "
            f"dimension {target_dim}")
    embed_dim = target_dim
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
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default="data/train")
    p.add_argument("--video-embeddings", default="data/video_embeddings.npz")
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))["training"]

    from .load_data import load_all
    from scipy.signal import decimate
    from .preprocess import bandpass, reject_artifacts

    data = load_all(args.data, sfreq=cfg["sfreq"])
    X, y = data.X, data.y
    sfreq = float(cfg["sfreq"])
    decimate_factor = int(cfg.get("decimate", 1))
    if decimate_factor > 1:
        X = decimate(X, decimate_factor, axis=-1, ftype="fir",
                     zero_phase=True).astype(np.float32)
        sfreq /= decimate_factor

    filter_cfg = cfg.get("filter")
    if filter_cfg:
        X = bandpass(X, sfreq, **filter_cfg)

    artifact_threshold = cfg.get("artifact_threshold_uv")
    if artifact_threshold is not None:
        keep = reject_artifacts(X, artifact_threshold)
        X, y = X[keep], y[keep]

    vid_embs = dict(np.load(args.video_embeddings))
    time_window = cfg.get("time_window")
    t_min, t_max = (0.0, X.shape[-1] / sfreq)
    if time_window is not None:
        t_min, t_max = (time_window[0] / sfreq, time_window[1] / sfreq)

    train(
        X, y, vid_embs, sfreq=sfreq, t_min=t_min, t_max=t_max,
        embed_dim=cfg.get("embed_dim"),
        batch_size=cfg["batch_size"], epochs=cfg["epochs"], lr=cfg["lr"],
        temperature=cfg["temperature"], device=cfg["device"],
        model_path=cfg["model_path"])
