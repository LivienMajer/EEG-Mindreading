"""Model-agnostic video embedding extraction (run on a GPU machine).

Extracts N frames per video and encodes them with a selectable model.
Saves PER-FRAME embeddings keyed f"{video_id:05d}_f{k:02d}" in one .npz,
so downstream code can frame-mean, frame-match, or time-align freely.

NOTE (Windows): import torch BEFORE decord and use the numpy bridge
(decord .asnumpy()), otherwise decord's ffmpeg DLLs clash with torch.

Usage (GPU machine):
  python -m src.extract_embeddings --models clip-vitb32 clip-vitl14 dinov2 \
      --stimuli-dir <...>/stimuli_train --out work/emb_train.npz --device cuda
  python -m src.extract_embeddings --models clip-vitb32 \
      --stimuli-dir <...>/stimuli_test  --out work/emb_test.npz  --device cuda

Models (all public, pre-hackathon release):
  clip-vitb32   OpenCLIP ViT-B-32 laion/openai     (512-d)
  clip-vitl14   OpenCLIP ViT-L-14 openai           (768-d)
  clap          LAION CLAP audio embeddings         (512-d)
  clip-clap     equal-weight CLIP + CLAP concatenation
  siglip2-b16   SigLIP 2 base                      (768-d)  [pip install siglip2 or timm]
  dinov2-b14    DINOv2 ViT-B/14 self-supervised    (768-d)  [torch.hub fairstudio]
  jepa2         V-JEPA 2 world-model embeddings   (Meta, video-native)
  alexnet-f3..f10  torchvision AlexNet feature layers (organizer baseline)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

# import order matters on Windows (see module docstring)
import torch  # noqa: E402
from PIL import Image  # noqa: E402

MODEL_KEYS = ["clip-vitb32", "clip-vitl14", "siglip2-b16", "dinov2-b14",
              "jepa2", "alexnet-f3", "alexnet-f6", "alexnet-f8", "alexnet-f10",
              "clap", "clip-clap"]


# ---------------------------------------------------------------------------
# video -> frames (decord, numpy bridge)
# ---------------------------------------------------------------------------

def extract_frames(video_path: Path, n_frames: int) -> np.ndarray:
    import decord
    vr = decord.VideoReader(str(video_path))
    total = len(vr)
    if total == 0:
        return np.zeros((0, 3, 224, 224), dtype=np.uint8)
    picks = np.linspace(0, max(total - 1, 0), n_frames).astype(int)
    batch = vr.get_batch(picks).asnumpy()  # (n, H, W, C) uint8
    return batch


# ---------------------------------------------------------------------------
# model zoo
# ---------------------------------------------------------------------------

def load_model(key: str, device: str):
    """Returns (callable: uint8 (n,H,W,C) -> float (n, D))."""
    if key.startswith("clip-"):
        import open_clip
        spec = {"clip-vitb32": ("ViT-B-32", "openai"),
                "clip-vitl14": ("ViT-L-14", "openai")}[key]
        model, _, preprocess = open_clip.create_model_and_transforms(*spec)
        model = model.eval().to(device)

        def encode(frames):
            x = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
            with torch.no_grad():
                e = model.encode_image(x.to(device))
            return e.float().cpu().numpy()
        return encode

    if key.startswith("alexnet-"):
        layer = {"alexnet-f3": 3, "alexnet-f6": 6,
                 "alexnet-f8": 8, "alexnet-f10": 10}[key]
        from torchvision import transforms
        from torchvision.models import alexnet, AlexNet_Weights
        model = alexnet(weights=AlexNet_Weights.IMAGENET1K_V1).to(device).eval()

        # which submodule: f=conv index in model.features
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
        feats = torch.nn.Sequential(*list(model.features.children())[:layer + 1])

        def encode(frames):
            x = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
            with torch.no_grad():
                e = feats(x.to(device))
            return e.flatten(1).float().cpu().numpy()
        return encode

    if key == "dinov2-b14":
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14",
                               verbose=False).to(device).eval()
        from torchvision import transforms
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(236, antialias=True),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

        def encode(frames):
            x = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
            with torch.no_grad():
                e = model(x.to(device))
            return e.float().cpu().numpy()
        return encode

    if key == "siglip2-b16":
        from torchvision import transforms
        model = torch.hub.load("google/timm", "siglip2_base_patch16_224",
                               pretrained=True, verbose=False)
        model = model.to(device).eval()
        preprocess = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224, antialias=True),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        def encode(frames):
            x = torch.stack([preprocess(Image.fromarray(f)) for f in frames])
            with torch.no_grad():
                e = model(x.to(device))
            return e.float().cpu().numpy()
        return encode

    if key == "jepa2":
        raise NotImplementedError(
            "V-JEPA 2: pip install -e git+https://github.com/facebookresearch/jepa"
            " then adapt this loader (embedding dim 3968; needs 16-frame clips).")

    raise ValueError(f"unknown model key {key}; choose from {MODEL_KEYS}")


# ---------------------------------------------------------------------------
# main sweep
# ---------------------------------------------------------------------------

def run(model_keys, stimuli_dir, out_file, n_frames, device, limit=None):
    stimuli_dir = Path(stimuli_dir)
    videos = sorted(stimuli_dir.glob("*.mp4"))
    if limit:
        videos = videos[:limit]
    if not videos:
        raise FileNotFoundError(f"no .mp4 under {stimuli_dir}")
    print(f"{len(videos)} videos in {stimuli_dir}")

    ids = [int(video.stem) for video in videos]
    all_frames = None
    if any(key != "clap" for key in model_keys):
        t0 = time.time()
        all_frames = [extract_frames(video, n_frames) for video in videos]
        print(f"frames extracted in {time.time() - t0:.0f}s "
              f"({len(videos)} x {n_frames} frames)")

    for key in model_keys:
        out_path = Path(out_file).with_name(
            Path(out_file).stem + f"__{key}.npz")
        if out_path.exists():
            print(f"[{key}] exists -> {out_path} (skip; delete to recompute)")
            continue

        t0 = time.time()
        emb = {}
        if key == "clap":
            from .video_embeddings import (
                NoAudioStreamError, _get_clap, encode_audio)
            model = _get_clap(device)
            audio_dim = None
            missing_audio = []
            audio_embeddings = {}
            for vid, video in zip(ids, videos):
                try:
                    e = encode_audio(video, model, device=device)
                except NoAudioStreamError:
                    missing_audio.append(video.stem)
                    continue
                audio_dim = e.shape[1]
                audio_embeddings[vid] = e[0].astype(np.float32)
            if audio_dim is None:
                raise ValueError(
                    f"No videos with audio streams found under {stimuli_dir}")
            for vid in ids:
                vector = audio_embeddings.get(
                    vid, np.zeros(audio_dim, dtype=np.float32))
                emb[f"{vid:05d}_f00"] = vector
            if missing_audio:
                print(
                    f"[{key}] WARNING: {len(missing_audio)} videos have no "
                    f"audio; saved zero audio vectors for IDs: "
                    f"{', '.join(map(str, missing_audio))}")
        elif key == "clip-clap":
            from .video_embeddings import (
                NoAudioStreamError, _get_clap, encode_audio,
                fuse_embeddings)
            encode_clip = load_model("clip-vitb32", device)
            clap_model = _get_clap(device)
            missing_audio = []
            for vid, video, frames in zip(ids, videos, all_frames):
                clip_emb = encode_clip(frames).mean(axis=0, keepdims=True)
                try:
                    clap_emb = encode_audio(video, clap_model, device=device)
                except NoAudioStreamError:
                    clap_emb = None
                    missing_audio.append(video.stem)
                fused = fuse_embeddings(clip_emb, clap_emb)
                emb[f"{vid:05d}_f00"] = fused[0].astype(np.float32)
            if missing_audio:
                print(
                    f"[{key}] WARNING: {len(missing_audio)} videos have no "
                    f"audio; saved CLIP-only fused vectors for IDs: "
                    f"{', '.join(map(str, missing_audio))}")
        else:
            encode = load_model(key, device)
            for vid, frames in zip(ids, all_frames):
                e = encode(frames)  # (n_frames, D)
                for k in range(e.shape[0]):
                    emb[f"{vid:05d}_f{k:02d}"] = e[k].astype(np.float32)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(out_path, **emb)
        print(f"[{key}] {len(emb)} embeddings -> {out_path} "
              f"in {time.time() - t0:.0f}s "
              f"(dim {next(iter(emb.values())).shape[0]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", required=True, choices=MODEL_KEYS)
    p.add_argument("--stimuli-dir", required=True)
    p.add_argument("--out", required=True, help="output npz (model key appended)")
    p.add_argument("--n-frames", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--limit", type=int, default=None, help="only first N videos")
    args = p.parse_args()
    print(f"device: {args.device}")
    run(args.models, args.stimuli_dir, args.out, args.n_frames, args.device,
        args.limit)


if __name__ == "__main__":
    main()
