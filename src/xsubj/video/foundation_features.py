"""Video and audio embeddings from public pre-trained models, same output format as clip_features.

Image models (N evenly spaced frames per video, L2-normalised per frame):
  clip-l14   open_clip ViT-L-14 / openai                        (768-d)
  dinov2     timm vit_base_patch14_dinov2.lvd142m at 224 px      (768-d, CLS token)
Audio model (the video's sound track, mono 48 kHz):
  clap       transformers laion/clap-htsat-unfused               (512-d)
             writes <out>.npz (whole clip) and <out>_onset.npz (first second only)

Writes <out>.npz with ids + emb (mean over frames) and, for image models, <out>_frames.npz with
ids + frames (n_videos, n_frames, d). Videos without a sound track get the embedding of silence.

    python -m src.xsubj.video.foundation_features --model clip-l14 \
        --stim data/eeg-mind-reading-challenge/stimuli_train data/eeg-mind-reading-challenge/stimuli_test \
        --out models/xsubj/features/clip_l14
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from src.xsubj.utils import get_device, resolve
from src.xsubj.video.clip_features import read_frames

SAMPLE_RATE = 48_000


def image_encoder(name: str, device):
    """Returns encode(list of PIL images) -> (n, d) float array, plus a version string."""
    if name == "clip-l14":
        import open_clip
        model, _, preprocess = open_clip.create_model_and_transforms("ViT-L-14", pretrained="openai")
        model = model.eval().to(device)
        version = f"open_clip {open_clip.__version__} ViT-L-14/openai"
        forward = model.encode_image
    elif name == "dinov2":
        import timm
        model = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, num_classes=0,
                                  img_size=224).eval().to(device)
        cfg = timm.data.resolve_model_data_config(model)
        cfg["input_size"] = (3, 224, 224)                # the pretrained config says 518 px
        preprocess = timm.data.create_transform(**cfg)
        version = f"timm {timm.__version__} vit_base_patch14_dinov2.lvd142m (224 px)"
        forward = model
    else:
        raise ValueError(name)

    @torch.no_grad()
    def encode(images):
        e = forward(torch.stack([preprocess(im) for im in images]).to(device)).float()
        return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
    return encode, version


def read_audio(path) -> np.ndarray:
    """Mono float32 waveform at 48 kHz; empty array if the video has no sound track."""
    import av
    with av.open(str(path)) as container:
        if not container.streams.audio:
            return np.zeros(0, dtype=np.float32)
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SAMPLE_RATE)
        chunks = []
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    return np.concatenate(chunks).astype(np.float32) if chunks else np.zeros(0, dtype=np.float32)


def clap_encoder(device):
    import transformers
    from transformers import ClapFeatureExtractor, ClapModel
    name = "laion/clap-htsat-unfused"
    model = ClapModel.from_pretrained(name).eval().to(device)
    fx = ClapFeatureExtractor.from_pretrained(name)

    @torch.no_grad()
    def encode(waveforms: list[np.ndarray]) -> np.ndarray:
        """(n, 512) for a batch of waveforms."""
        waveforms = [w if w.size else np.zeros(SAMPLE_RATE, dtype=np.float32) for w in waveforms]
        inputs = fx(waveforms, sampling_rate=SAMPLE_RATE, return_tensors="pt").to(device)
        e = model.get_audio_features(**inputs)
        e = getattr(e, "pooler_output", e).float()
        return (e / e.norm(dim=-1, keepdim=True)).cpu().numpy()
    return encode, f"transformers {transformers.__version__} {name}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["clip-l14", "dinov2", "clap"])
    ap.add_argument("--stim", nargs="+", required=True, help="folders with <id>.mp4")
    ap.add_argument("--out", required=True, help="output path without .npz")
    ap.add_argument("--n-frames", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="first N videos per folder (smoke test)")
    args = ap.parse_args()

    device = get_device()
    files = [f for d in args.stim for f in sorted(resolve(d).glob("*.mp4"))[:args.limit]]
    ids = [str(int(f.stem)) for f in files]          # "0035" -> "35", matching the EEG keys
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    if args.model == "clap":
        encode, version = clap_encoder(device)
        full, onset, silent = [], [], 0
        for i, f in enumerate(files):
            w = read_audio(f)
            silent += w.size == 0
            e = encode([w, w[:SAMPLE_RATE]])
            full.append(e[0])
            onset.append(e[1])
            if i % 100 == 0:
                print(f"{i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
        np.savez(f"{out}.npz", ids=np.array(ids), emb=np.stack(full).astype(np.float32))
        np.savez(f"{out}_onset.npz", ids=np.array(ids), emb=np.stack(onset).astype(np.float32))
        print(f"saved {len(ids)} videos ({silent} without sound) -> {out}.npz, {out}_onset.npz ({version})")
        return

    encode, version = image_encoder(args.model, device)
    per_frame = []
    for i, f in enumerate(files):
        per_frame.append(encode(read_frames(f, args.n_frames)))
        if i % 100 == 0:
            print(f"{i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)
    frames = np.stack(per_frame).astype(np.float32)
    np.savez(f"{out}.npz", ids=np.array(ids), emb=frames.mean(1))
    np.savez(f"{out}_frames.npz", ids=np.array(ids), frames=frames)
    print(f"saved {len(ids)} videos x {frames.shape[1]} frames x {frames.shape[2]} -> {out}.npz ({version})")


if __name__ == "__main__":
    main()
