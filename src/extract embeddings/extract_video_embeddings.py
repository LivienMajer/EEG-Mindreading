"""Extract one pooled image-model embedding for each video."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader, cpu
from PIL import Image
from tqdm import tqdm


@dataclass
class ImageEncoder:
    """Model adapter used by the model-independent extraction pipeline."""

    encode: Callable[[torch.Tensor], torch.Tensor]
    preprocess: Callable[[Image.Image], torch.Tensor]
    model_name: str
    pretrained: str


def load_encoder(
    backend: str,
    model_name: str,
    pretrained: str,
    device: str,
) -> ImageEncoder:
    """Load an image-embedding backend.

    Add another backend here while keeping video decoding, pooling, and output
    format independent of the model library.
    """
    if backend == "open_clip":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        model = model.eval().to(device)
        return ImageEncoder(
            encode=model.encode_image,
            preprocess=preprocess,
            model_name=model_name,
            pretrained=pretrained,
        )
    raise ValueError(
        f"Unsupported backend {backend!r}; currently supported: open_clip")


def sample_frames(video_path: Path, n_frames: int) -> list[Image.Image]:
    """Decode evenly spaced RGB frames from a video."""
    reader = VideoReader(str(video_path), ctx=cpu(0))
    if len(reader) == 0:
        raise ValueError(f"No frames found in video: {video_path}")

    indices = np.linspace(0, len(reader) - 1, n_frames, dtype=np.int64)
    frames = reader.get_batch(indices).asnumpy()
    return [Image.fromarray(frame).convert("RGB") for frame in frames]


@torch.inference_mode()
def extract_embeddings(
    video_dir: Path,
    output_path: Path,
    backend: str = "open_clip",
    model_name: str = "ViT-B-32",
    pretrained: str = "openai",
    n_frames: int = 8,
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    """Encode videos as normalized means of normalized frame embeddings."""

    if n_frames < 1:
        raise ValueError("--n-frames must be at least 1")
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

    print(f"Loading {backend}:{model_name} ({pretrained}) on {device}")
    encoder = load_encoder(backend, model_name, pretrained, device)

    embeddings = []
    for video_path in tqdm(videos, desc="Encoding videos"):
        try:
            frames = sample_frames(video_path, n_frames)
            batch = torch.stack(
                [encoder.preprocess(frame) for frame in frames]).to(device)
            frame_embeddings = F.normalize(encoder.encode(batch), dim=-1)
            video_embedding = F.normalize(
                frame_embeddings.mean(dim=0), dim=0)
            embeddings.append(video_embedding.cpu().numpy().astype(np.float32))
        except Exception as exc:
            raise RuntimeError(f"Failed to encode {video_path}") from exc

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        video_ids=np.asarray(video_ids, dtype=np.int32),
        embeddings=np.stack(embeddings),
        backend=np.asarray(backend),
        model=np.asarray(encoder.model_name),
        pretrained=np.asarray(encoder.pretrained),
        n_frames=np.asarray(n_frames, dtype=np.int32),
    )
    print(f"Saved {len(video_ids)} embeddings (dim {embeddings[0].shape[0]})")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract pooled image-model embeddings from MP4 videos.")
    parser.add_argument("--videos", type=Path, required=True,
                        help="Folder containing numeric-ID .mp4 files")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output .npz archive")
    parser.add_argument("--backend", default="open_clip",
                        choices=("open_clip",),
                        help="Image-model library to use")
    parser.add_argument("--model", default="ViT-B-32",
                        help="Model name supported by the selected backend")
    parser.add_argument("--pretrained", default="openai",
                        help="Pretrained weights tag supported by the backend")
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Number of evenly spaced frames per video")
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
        pretrained=args.pretrained,
        n_frames=args.n_frames,
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
