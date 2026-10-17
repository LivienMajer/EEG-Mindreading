"""Extract one pooled audio-model embedding for each video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import av
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

SAMPLE_RATE = 48_000


class NoAudioStreamError(ValueError):
    """Raised when a video does not contain an audio stream."""


@dataclass
class AudioEncoder:
    """Model adapter used by the model-independent extraction pipeline."""

    encode: Callable[[np.ndarray], np.ndarray]
    model_name: str
    checkpoint: str


def load_encoder(
    backend: str,
    model_name: str,
    checkpoint: str | None,
    device: str,
) -> AudioEncoder:
    """Load an audio-embedding backend.

    Add another backend here while keeping media decoding and output handling
    independent of the model library.
    """
    if backend == "laion_clap":
        import laion_clap

        model = laion_clap.CLAP_Module(
            enable_fusion=False, device=device, amodel=model_name)
        if checkpoint is None:
            model.load_ckpt()
            checkpoint_name = "default"
        else:
            model.load_ckpt(ckpt=checkpoint)
            checkpoint_name = str(checkpoint)
        return AudioEncoder(
            encode=lambda waveform: model.get_audio_embedding_from_data(
                [waveform], use_tensor=False)[0],
            model_name=model_name,
            checkpoint=checkpoint_name,
        )
    raise ValueError(
        f"Unsupported backend {backend!r}; currently supported: laion_clap")


def decode_audio(video_path: Path) -> np.ndarray:
    """Decode and resample a video's audio track to mono 48 kHz float audio."""
    with av.open(str(video_path), metadata_errors="replace") as container:
        if not container.streams.audio:
            raise NoAudioStreamError(f"No audio stream in {video_path}")

        stream = container.streams.audio[0]
        resampler = av.AudioResampler(
            format="fltp", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(stream):
            chunks.extend(
                output.to_ndarray().reshape(-1)
                for output in resampler.resample(frame)
            )
        chunks.extend(
            output.to_ndarray().reshape(-1)
            for output in resampler.resample(None)
        )

    if not chunks:
        raise ValueError(f"Audio stream decoded no samples: {video_path}")
    waveform = np.concatenate(chunks).astype(np.float32, copy=False)
    if not np.isfinite(waveform).all():
        raise ValueError(f"Audio contains non-finite samples: {video_path}")
    return waveform


@torch.inference_mode()
def extract_embeddings(
    video_dir: Path,
    output_path: Path,
    backend: str = "laion_clap",
    model_name: str = "HTSAT-tiny",
    checkpoint: str | None = None,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    """Encode all videos as normalized audio embeddings."""
    if limit is not None and limit < 1:
        raise ValueError("--limit must be at least 1")
    if output_path.exists() and not overwrite:
        raise FileExistsError(
            f"Output already exists: {output_path}. Use --overwrite to replace it.")

    videos = [
        path for path in video_dir.iterdir()
        if path.is_file() and path.suffix.lower() == ".mp4"
    ]
    if not videos:
        raise FileNotFoundError(f"No .mp4 videos found in {video_dir}")
    for path in videos:
        if not path.stem.isdigit():
            raise ValueError(
                f"Video filename must have a numeric ID, got {path.name}")
    videos.sort(key=lambda path: int(path.stem))
    if limit is not None:
        videos = videos[:limit]
    video_ids = [int(path.stem) for path in videos]
    if len(set(video_ids)) != len(video_ids):
        raise ValueError(f"Duplicate numeric video IDs found in {video_dir}")

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but this PyTorch installation cannot use CUDA.")

    print(f"Loading {backend}:{model_name} on {device}")
    encoder = load_encoder(backend, model_name, checkpoint, device)

    embeddings_by_id = {}
    missing_audio_ids = []
    embedding_dim = None
    for video_path in tqdm(videos, desc="Encoding audio"):
        try:
            waveform = decode_audio(video_path)
        except NoAudioStreamError:
            missing_audio_ids.append(int(video_path.stem))
            continue
        except Exception as exc:
            raise RuntimeError(f"Failed to decode audio from {video_path}") from exc

        try:
            embedding = np.asarray(encoder.encode(waveform), dtype=np.float32)
        except Exception as exc:
            raise RuntimeError(f"Failed to encode audio from {video_path}") from exc
        if embedding.ndim != 1 or embedding.size == 0:
            raise ValueError(
                f"Audio backend returned invalid embedding shape "
                f"{embedding.shape} for {video_path}")
        if not np.isfinite(embedding).all():
            raise ValueError(
                f"Audio backend returned non-finite values for {video_path}")
        embedding = F.normalize(torch.from_numpy(embedding), dim=0).numpy()
        embedding_dim = embedding.size
        embeddings_by_id[int(video_path.stem)] = embedding

    if embedding_dim is None:
        raise ValueError(
            f"No audio streams found in videos under {video_dir}; "
            "cannot determine the embedding dimension.")

    embeddings = np.stack([
        embeddings_by_id.get(video_id, np.zeros(embedding_dim, dtype=np.float32))
        for video_id in video_ids
    ])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_ids=np.asarray(video_ids, dtype=np.int32),
        embeddings=embeddings,
        backend=np.asarray(backend),
        model=np.asarray(encoder.model_name),
        checkpoint=np.asarray(encoder.checkpoint),
        sample_rate=np.asarray(SAMPLE_RATE, dtype=np.int32),
        missing_audio_video_ids=np.asarray(missing_audio_ids, dtype=np.int32),
    )
    print(f"Saved {len(video_ids)} embeddings (dim {embedding_dim})")
    if missing_audio_ids:
        print(
            f"WARNING: {len(missing_audio_ids)} videos have no audio stream; "
            f"saved zero vectors for IDs: {', '.join(map(str, missing_audio_ids))}")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pooled audio-model embeddings from MP4 videos.")
    parser.add_argument("--videos", type=Path, required=True,
                        help="Folder containing numeric-ID .mp4 files")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output .npz archive")
    parser.add_argument("--backend", default="laion_clap",
                        choices=("laion_clap",),
                        help="Audio-model library to use")
    parser.add_argument("--model", default="HTSAT-tiny",
                        help="Architecture supported by the selected backend")
    parser.add_argument("--checkpoint", type=Path,
                        help="Optional checkpoint file; defaults to LAION CLAP weights")
    parser.add_argument("--device", choices=("cuda", "cpu"))
    parser.add_argument("--limit", type=int,
                        help="Only encode the first N videos (smoke testing)")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace the output file if it already exists")
    args = parser.parse_args()
    extract_embeddings(
        video_dir=args.videos,
        output_path=args.out,
        backend=args.backend,
        model_name=args.model,
        checkpoint=str(args.checkpoint) if args.checkpoint else None,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
