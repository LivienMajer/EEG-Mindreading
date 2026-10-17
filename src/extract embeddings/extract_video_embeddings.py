"""Extract one image or temporal-video embedding for each video."""

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

    encode: Callable[[torch.Tensor], torch.Tensor] | None
    encode_multilayer: Callable[
        [torch.Tensor, tuple[int, ...]], torch.Tensor] | None
    preprocess: Callable[[Image.Image], torch.Tensor] | None
    model_name: str
    pretrained: str
    encode_video: Callable[[list[Image.Image]], torch.Tensor] | None = None


def load_encoder(
    backend: str,
    model_name: str,
    pretrained: str | None,
    device: str,
) -> ImageEncoder:
    """Load an image-embedding backend.

    Add another backend here while keeping video decoding, pooling, and output
    format independent of the model library.
    """
    if backend == "open_clip":
        import open_clip

        pretrained = pretrained or "openai"
        model, _, preprocess = open_clip.create_model_and_transforms(
            model_name, pretrained=pretrained)
        model = model.eval().to(device)

        def encode_multilayer(
            images: torch.Tensor, layers: tuple[int, ...]
        ) -> torch.Tensor:
            blocks = getattr(getattr(model.visual, "transformer", None),
                             "resblocks", None)
            if blocks is None:
                raise ValueError(
                    f"{model_name} does not expose ViT transformer blocks")
            if any(layer < 1 or layer > len(blocks) for layer in layers):
                raise ValueError(
                    f"Requested layers {layers} are outside 1..{len(blocks)}")

            activations: dict[int, torch.Tensor] = {}
            handles = []
            for layer in layers:
                handles.append(blocks[layer - 1].register_forward_hook(
                    lambda *hook_args, layer=layer:
                    activations.__setitem__(layer, hook_args[-1])))
            try:
                final = model.encode_image(images)
            finally:
                for handle in handles:
                    handle.remove()

            spaces = []
            for layer in layers:
                activation = activations.get(layer)
                if not isinstance(activation, torch.Tensor) or activation.ndim != 3:
                    raise RuntimeError(
                        f"ViT block {layer} did not return a sequence tensor")
                if activation.shape[0] == images.shape[0]:
                    batch_first = activation
                elif activation.shape[1] == images.shape[0]:
                    batch_first = activation.permute(1, 0, 2)
                else:
                    raise RuntimeError(
                        f"Unexpected ViT block {layer} output shape: "
                        f"{tuple(activation.shape)}")
                pooled = batch_first[:, 1:, :].mean(dim=1)
                spaces.append(F.normalize(pooled, dim=-1))
            spaces.append(F.normalize(final, dim=-1))
            combined = torch.cat(spaces, dim=-1) / np.sqrt(len(spaces))
            return F.normalize(combined, dim=-1)

        return ImageEncoder(
            encode=model.encode_image,
            encode_multilayer=encode_multilayer,
            preprocess=preprocess,
            model_name=model_name,
            pretrained=pretrained,
        )
    if backend == "videomae":
        if pretrained is not None:
            raise ValueError(
                "VideoMAE weights are selected through --model; omit --pretrained")
        from transformers import VideoMAEImageProcessor, VideoMAEModel

        processor = VideoMAEImageProcessor.from_pretrained(model_name)
        model = VideoMAEModel.from_pretrained(model_name).eval().to(device)

        def encode_video(frames: list[Image.Image]) -> torch.Tensor:
            video_frames = [np.asarray(frame.convert("RGB")) for frame in frames]
            inputs = processor(video_frames, return_tensors="pt")
            pixels = inputs["pixel_values"].to(device)
            if pixels.shape[1] != model.config.num_frames:
                raise ValueError(
                    f"{model_name} requires {model.config.num_frames} frames; "
                    f"received {pixels.shape[1]}")
            tokens = model(pixel_values=pixels).last_hidden_state
            return F.normalize(tokens.mean(dim=1), dim=-1)

        return ImageEncoder(
            encode=None,
            encode_multilayer=None,
            preprocess=None,
            model_name=model_name,
            pretrained="huggingface",
            encode_video=encode_video,
        )
    raise ValueError(
        f"Unsupported backend {backend!r}; choose open_clip or videomae")


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
    pretrained: str | None = None,
    n_frames: int = 8,
    intermediate_layers: tuple[int, ...] = (),
    device: str | None = None,
    limit: int | None = None,
    overwrite: bool = False,
) -> None:
    """Extract normalized per-video image or temporal-video embeddings.

    When intermediate layers are requested, mean-pooled patch activations from
    those 1-based ViT blocks are concatenated with the final image embedding.
    Each layer space is normalized before concatenation.
    VideoMAE receives the full ordered frame sequence and mean-pools its
    contextualized spatiotemporal output tokens per video.
    """

    if n_frames < 1:
        raise ValueError("--n-frames must be at least 1")
    if len(set(intermediate_layers)) != len(intermediate_layers):
        raise ValueError("--intermediate-layers must not contain duplicates")
    if any(layer < 1 for layer in intermediate_layers):
        raise ValueError("--intermediate-layers are 1-based and must be positive")
    if backend == "videomae" and intermediate_layers:
        raise ValueError("--intermediate-layers only applies to the open_clip backend")
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

    print(f"Loading {backend}:{model_name} ({pretrained or 'model default'}) on {device}")
    encoder = load_encoder(backend, model_name, pretrained, device)
    if backend == "videomae" and n_frames != 16:
        raise ValueError(
            f"{model_name} expects 16 frames; pass --n-frames 16 (got {n_frames})")

    embeddings = []
    for video_path in tqdm(videos, desc="Encoding videos"):
        try:
            frames = sample_frames(video_path, n_frames)
            if encoder.encode_video is not None:
                video_embedding = encoder.encode_video(frames)[0]
            else:
                if encoder.encode is None or encoder.preprocess is None:
                    raise RuntimeError(
                        f"Incomplete image encoder adapter for {backend}")
                batch = torch.stack(
                    [encoder.preprocess(frame) for frame in frames]).to(device)
                if intermediate_layers:
                    if encoder.encode_multilayer is None:
                        raise ValueError(
                            "This image backend does not support intermediate layers")
                    frame_embeddings = encoder.encode_multilayer(
                        batch, intermediate_layers)
                else:
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
        intermediate_layers=np.asarray(intermediate_layers, dtype=np.int32),
    )
    print(f"Saved {len(video_ids)} embeddings (dim {embeddings[0].shape[0]})")
    print(f"Output: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract image or temporal-video embeddings from MP4 videos.")
    parser.add_argument("--videos", type=Path, required=True,
                        help="Folder containing numeric-ID .mp4 files")
    parser.add_argument("--out", type=Path, required=True,
                        help="Output .npz archive")
    parser.add_argument("--backend", default="open_clip",
                        choices=("open_clip", "videomae"),
                        help="Image encoder or temporal video encoder")
    parser.add_argument("--model", default="ViT-B-32",
                        help="Backend model name or Hugging Face model ID")
    parser.add_argument("--pretrained", default=None,
                        help="OpenCLIP pretrained tag (VideoMAE uses --model)")
    parser.add_argument("--n-frames", type=int, default=8,
                        help="Number of evenly spaced frames per video")
    parser.add_argument("--intermediate-layers", type=int, nargs="+", default=(),
                        metavar="LAYER",
                        help="Also pool these 1-based ViT block activations "
                             "before the final image embedding")
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
        intermediate_layers=tuple(args.intermediate_layers),
        device=args.device,
        limit=args.limit,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
