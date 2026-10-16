"""CLIP image embeddings per video (public weights: open_clip ViT-B-32, pretrained='openai').

N evenly spaced frames per video are embedded and L2-normalised. Writes
  <out>.npz         ids + emb: mean over frames (input for the video bank)
  <out>_frames.npz  ids + frames: (n_videos, n_frames, d), for time-resolved analyses

    python -m src.xsubj.video.clip_features --stim data/eeg-mind-reading-challenge/stimuli_train \
        data/eeg-mind-reading-challenge/stimuli_test --out models/xsubj/features/clip_vitb32
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np
import torch
from PIL import Image

from src.xsubj.utils import get_device, resolve


def read_frames(path, n_frames: int) -> list[Image.Image]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fno in np.linspace(0, max(total - 1, 0), n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
        ok, frame = cap.read()
        if ok:
            frames.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read {path}")
    while len(frames) < n_frames:                    # rare short reads: repeat the last frame
        frames.append(frames[-1])
    return frames


@torch.no_grad()
def main():
    import open_clip

    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", nargs="+", required=True, help="folders with <id>.mp4")
    ap.add_argument("--out", default="models/xsubj/features/clip_vitb32")
    ap.add_argument("--model", default="ViT-B-32")
    ap.add_argument("--pretrained", default="openai")
    ap.add_argument("--n-frames", type=int, default=8)
    args = ap.parse_args()

    device = get_device()
    model, _, preprocess = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    model = model.eval().to(device)
    files = [f for d in args.stim for f in sorted(resolve(d).glob("*.mp4"))]
    ids, per_frame, t0 = [], [], time.time()
    for i, f in enumerate(files):
        batch = torch.stack([preprocess(im) for im in read_frames(f, args.n_frames)]).to(device)
        e = model.encode_image(batch).float()
        per_frame.append((e / e.norm(dim=-1, keepdim=True)).cpu().numpy())
        ids.append(str(int(f.stem)))                 # "0035" -> "35", matching the EEG keys
        if i % 100 == 0:
            print(f"{i}/{len(files)}  {time.time() - t0:.0f}s", flush=True)

    frames = np.stack(per_frame).astype(np.float32)
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f"{out}.npz", ids=np.array(ids), emb=frames.mean(1))
    np.savez(f"{out}_frames.npz", ids=np.array(ids), frames=frames)
    print(f"saved {len(ids)} videos x {frames.shape[1]} frames x {frames.shape[2]} -> {out}.npz "
          f"(open_clip {open_clip.__version__}, {args.model}/{args.pretrained})")


if __name__ == "__main__":
    main()
