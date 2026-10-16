"""Extract embeddings for candidate videos using pretrained models.

Two options:
  1. Frame-level CLIP embeddings (open_clip) - simple, strong baseline.
  2. A video foundation model (InternVideo / SVC) - can be swapped in later.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


def _get_open_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.eval(), preprocess, tokenizer


def encode_frames(video_path: str | Path, model=None, preprocess=None,
                  n_frames: int = 8, device: str = "cpu") -> np.ndarray:
    """Extract n_frames evenly spaced frames and CLIP-embed each.

    Returns (n_frames, embed_dim) L2-normalized embeddings.
    """
    import torch
    import decord
    decord.bridge.set_bridge("torch")

    if model is None:
        model, preprocess, _ = _get_open_clip()
    model = model.to(device)

    vr = decord.VideoReader(str(video_path))
    idx = np.linspace(0, len(vr) - 1, n_frames).astype(int)
    frames = vr.get_batch(idx)  # (n_frames, H, W, C), uint8
    batch = torch.stack([preprocess(f) for f in frames]).to(device)

    with torch.no_grad():
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def encode_image(image_path: str | Path, device: str = "cpu") -> np.ndarray:
    """CLIP-embed a single image (for datasets with keyframes instead of videos)."""
    from PIL import Image
    import torch

    model, preprocess, _ = _get_open_clip()
    model = model.to(device)
    img = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        emb = model.encode_image(img)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def encode_all_candidates(candidate_dir: str | Path, n_frames: int = 8,
                          out_file: str | Path | None = None,
                          device: str = "cpu") -> dict[str, np.ndarray]:
    """Embed every video in candidate_dir.

    Returns {video_id: (n_frames, dim) embedding matrix}.
    If out_file given, caches result to .npz under data/ (gitignored).

    Support either .mp4/... videos, or one representative frame per video
    (e.g. video_001.jpg).
    """
    candidate_dir = Path(candidate_dir)
    model, preprocess, _ = _get_open_clip()

    results: dict[str, np.ndarray] = {}
    exts = {".mp4", ".avi", ".mkv", ".mov", ".webm", ".jpg", ".jpeg", ".png"}
    files = sorted(p for p in candidate_dir.iterdir() if p.suffix.lower() in exts)
    for f in files:
        if f.suffix.lower() in {".jpg", ".jpeg", ".png"}:
            results[f.stem] = encode_image(f, device=device)
        else:
            results[f.stem] = encode_frames(f, model, preprocess,
                                             n_frames=n_frames, device=device)

    if out_file:
        np.savez(out_file, **results)
    return results
