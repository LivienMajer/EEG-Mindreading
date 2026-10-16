"""Extract CLIP image and CLAP audio embeddings for candidate videos.

The CLIP and CLAP embeddings are saved separately because their latent spaces
are not directly comparable. They can be trained/evaluated as separate targets
before choosing a multimodal fusion strategy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_FRAMES = 8  # number of frames to sample per video for CLIP embedding


class NoAudioStreamError(ValueError):
    """Raised when a video has no audio track."""


def _get_open_clip(model_name: str = "ViT-B-32", pretrained: str = "openai"):
    import open_clip
    model, _, preprocess = open_clip.create_model_and_transforms(
        model_name, pretrained=pretrained)
    tokenizer = open_clip.get_tokenizer(model_name)
    return model.eval(), preprocess, tokenizer


def _get_clap(device: str = DEVICE):
    import laion_clap

    model = laion_clap.CLAP_Module(enable_fusion=False, device=device)
    model.load_ckpt()
    return model


def encode_frames(video_path: str | Path, model=None, preprocess=None,
                  n_frames: int = NUM_FRAMES, device: str = DEVICE) -> np.ndarray:
    """Extract n_frames evenly spaced frames and CLIP-embed each.

    Returns (n_frames, embed_dim) L2-normalized embeddings.
    """
    import torch
    import decord
    decord.bridge.set_bridge("torch")

    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    if model is None:
        model, preprocess, _ = _get_open_clip()
    model = model.to(device)

    vr = decord.VideoReader(str(video_path))
    if len(vr) == 0:
        raise ValueError(f"Video contains no frames: {video_path}")
    idx = np.linspace(0, len(vr) - 1, n_frames).astype(int)
    frames = vr.get_batch(idx)  # (n_frames, H, W, C), uint8
    batch = torch.stack([preprocess(f) for f in frames]).to(device)

    with torch.no_grad():
        emb = model.encode_image(batch)
        emb = emb / emb.norm(dim=-1, keepdim=True)
    return emb.cpu().numpy().astype(np.float32)


def encode_audio(video_path: str | Path, model=None, device: str = DEVICE
                 ) -> np.ndarray:
    """Extract one CLAP embedding from the video's complete audio track.

    Audio is decoded/resampled to mono 48 kHz. The challenge clips are 3
    seconds long; CLAP's preprocessing repeat-pads this short input to its
    expected 10-second length.

    Returns a (1, embed_dim) L2-normalized array, matching the frame-level
    embedding convention used by the training and evaluation helpers.
    """
    import av

    if model is None:
        model = _get_clap(device)

    sample_rate = 48_000
    chunks = []

    with av.open(str(video_path), metadata_errors="replace") as container:
        audio_streams = container.streams.audio
        if not audio_streams:
            raise NoAudioStreamError(
                f"Video has no audio stream: {video_path}")

        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=sample_rate)
        for frame in container.decode(audio_streams[0]):
            resampled = resampler.resample(frame)
            if resampled is None:
                continue
            if not isinstance(resampled, list):
                resampled = [resampled]
            chunks.extend(
                frame.to_ndarray().reshape(-1) for frame in resampled)

        flushed = resampler.resample(None)
        if flushed is not None:
            if not isinstance(flushed, list):
                flushed = [flushed]
            chunks.extend(frame.to_ndarray().reshape(-1) for frame in flushed)

    if not chunks:
        raise ValueError(f"Could not decode audio from video: {video_path}")

    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    embeddings = model.get_audio_embedding_from_data(
        x=waveform[np.newaxis, :], use_tensor=False)
    embeddings = np.asarray(embeddings, dtype=np.float32)
    embeddings /= np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-8
    mean_embedding = embeddings.mean(axis=0, keepdims=True)
    mean_embedding /= np.linalg.norm(mean_embedding, axis=-1, keepdims=True) + 1e-8
    return mean_embedding.astype(np.float32)


def fuse_embeddings(
        clip_emb: np.ndarray, clap_emb: np.ndarray | None) -> np.ndarray:
    """Fuse pooled CLIP and CLAP vectors with equal modality weight.

    When CLAP is absent, return the normalized visual embedding followed by
    zeros, preserving visual-only matching for silent videos.
    """
    clip_emb = np.asarray(clip_emb, dtype=np.float32)
    if (clip_emb.ndim != 2 or clip_emb.shape[0] == 0
            or not np.isfinite(clip_emb).all()):
        raise ValueError("Expected finite CLIP embeddings with shape (n_frames, D)")
    clip_norm = np.linalg.norm(clip_emb, axis=1, keepdims=True)
    if np.any(clip_norm <= 1e-8):
        raise ValueError("CLIP embeddings must have nonzero norms")
    clip_unit = clip_emb / clip_norm
    pooled_clip = clip_unit.mean(axis=0, keepdims=True)
    pooled_clip_norm = np.linalg.norm(pooled_clip, axis=1, keepdims=True)
    if pooled_clip_norm[0, 0] <= 1e-8:
        raise ValueError("Pooled CLIP embedding has zero norm")
    pooled_clip /= pooled_clip_norm

    if clap_emb is None:
        return np.concatenate(
            (pooled_clip, np.zeros((1, 512), dtype=np.float32)), axis=1)

    clap_emb = np.asarray(clap_emb, dtype=np.float32)
    if (clap_emb.ndim != 2 or clap_emb.shape[0] != 1
            or not np.isfinite(clap_emb).all()):
        raise ValueError("Expected finite CLAP embeddings with shape (1, D)")
    clap_norm = np.linalg.norm(clap_emb, axis=1, keepdims=True)
    if np.any(clap_norm <= 1e-8):
        raise ValueError("CLAP embeddings must have nonzero norms")
    clap_unit = clap_emb / clap_norm
    return np.concatenate((pooled_clip, clap_unit), axis=1) / np.sqrt(2.0)


def encode_image(image_path: str | Path, device: str = DEVICE) -> np.ndarray:
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


def encode_all_candidates(candidate_dir: str | Path, n_frames: int = NUM_FRAMES,
                          out_file: str | Path | None = None,
                          device: str = DEVICE) -> dict[str, np.ndarray]:
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


def encode_all_candidates_multimodal(
        candidate_dir: str | Path, n_frames: int = 8,
        clip_out_file: str | Path | None = None,
        clap_out_file: str | Path | None = None,
        device: str = "cpu") -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Extract frame-level CLIP and clip-level CLAP embeddings for videos.

    Returns ``(clip_embeddings, clap_embeddings)``. Each mapping is keyed by
    video stem and has shape ``(n_frames, D)`` or ``(1, D)`` respectively.
    Optional output paths save each modality to its own ``.npz`` file.
    """
    candidate_dir = Path(candidate_dir)
    video_exts = {".mp4", ".avi", ".mkv", ".mov", ".webm"}
    files = sorted(
        path for path in candidate_dir.iterdir()
        if path.suffix.lower() in video_exts)
    if not files:
        raise FileNotFoundError(
            f"No supported video files found in {candidate_dir}")
    if n_frames < 1:
        raise ValueError("n_frames must be at least 1")
    model, preprocess, _ = _get_open_clip()
    clap_model = _get_clap(device)

    clip_embeddings: dict[str, np.ndarray] = {}
    clap_embeddings: dict[str, np.ndarray] = {}
    for path in files:
        clip_embeddings[path.stem] = encode_frames(
            path, model, preprocess, n_frames=n_frames, device=device)
        clap_embeddings[path.stem] = encode_audio(
            path, clap_model, device=device)

    for output_path, embeddings in (
            (clip_out_file, clip_embeddings),
            (clap_out_file, clap_embeddings)):
        if output_path is not None:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(output_path, **embeddings)

    return clip_embeddings, clap_embeddings
