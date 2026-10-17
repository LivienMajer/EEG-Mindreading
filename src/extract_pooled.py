"""Video embeddings in the megafusion's file format, pooled over frames at extraction.

extract_embeddings saves one entry per frame (alexnet-f3: 8 frames x 140k dims ~ 4.5 GB);
bench_embeddings.load_pooled then averages the frames of each video. This does the same
average while extracting and writes the video-level layout that load_pooled passes
through unchanged ({"00001": (D,), ...}), split into train (0001-1000) and test
(1001-1102) files like extract_embeddings' two runs. Same frames (extract_frames, 8 per
video) and same encoder (load_model) as extract_embeddings.

It can also convert an ids/emb bank (e.g. src.xsubj's CLIP features) into that layout.

Usage:
  python -m src.extract_pooled --model alexnet-f3 --stimuli-root <data> --out-dir work
  python -m src.extract_pooled --convert models/xsubj/features/clip_vitb32.npz \
      --name clip-vitb32 --out-dir work
-> work/emb_train__<name>.npz, work/emb_test__<name>.npz
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch  # noqa: F401  (before decord, see extract_embeddings)

from .extract_embeddings import extract_frames, load_model


def save_split(emb: dict, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    for split, keep in (("train", lambda v: v <= 1000), ("test", lambda v: v > 1000)):
        part = {f"{v:05d}": e for v, e in emb.items() if keep(v)}
        path = out_dir / f"emb_{split}__{name}.npz"
        np.savez(path, **part)
        print(f"saved {path} ({len(part)} videos, dim {next(iter(part.values())).shape[0]})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", help="extract_embeddings model key, e.g. alexnet-f3")
    p.add_argument("--stimuli-root", help="folder with stimuli_train/ and stimuli_test/")
    p.add_argument("--convert", help="ids/emb npz to convert instead of extracting")
    p.add_argument("--name", help="name in the output files (default: --model)")
    p.add_argument("--out-dir", default="work")
    p.add_argument("--n-frames", type=int, default=8)
    args = p.parse_args()
    name = args.name or args.model

    if args.convert:
        z = np.load(args.convert)
        emb = {int(i): e.astype(np.float32) for i, e in zip(z["ids"], z["emb"])}
    else:
        root = Path(args.stimuli_root)
        videos = sorted((root / "stimuli_train").glob("*.mp4")) + \
            sorted((root / "stimuli_test").glob("*.mp4"))
        encode = load_model(args.model, "cpu")
        emb, t0 = {}, time.time()
        for i, v in enumerate(videos):
            emb[int(v.stem)] = encode(extract_frames(v, args.n_frames)).mean(0).astype(np.float32)
            if (i + 1) % 200 == 0:
                print(f"  {i + 1}/{len(videos)} videos, {time.time() - t0:.0f} s", flush=True)
    save_split(emb, Path(args.out_dir), name)


if __name__ == "__main__":
    main()
