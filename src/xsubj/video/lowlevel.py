"""Low-level visual features per video: what early visual cortex (and the early ERP) responds to.

Per sampled frame: an 8x8 grayscale thumbnail, mean RGB, luminance, RMS contrast, saturation
and edge energy. Per video: the onset frame (drives the P1/N1), the mean over frames, and
motion energy (mean absolute difference between consecutive sampled frames).

    python -m src.xsubj.video.lowlevel --stim data/eeg-mind-reading-challenge/stimuli_train \
        data/eeg-mind-reading-challenge/stimuli_test --out models/xsubj/features/lowlevel.npz
"""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from src.xsubj.utils import resolve


def frame_features(bgr: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32) / 255
    thumb = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).ravel()
    edges = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1)).mean()
    rgb = bgr[..., ::-1].reshape(-1, 3).mean(0) / 255
    return np.concatenate([thumb, rgb, [gray.mean(), gray.std(), hsv[..., 1].mean(), edges]])


def video_features(path: Path, n_frames: int = 16, size: int = 128) -> np.ndarray:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames = []
    for fno in np.linspace(0, max(total - 1, 0), n_frames).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
        ok, frame = cap.read()
        if ok:
            frames.append(cv2.resize(frame, (size, size), interpolation=cv2.INTER_AREA))
    cap.release()
    if not frames:
        raise RuntimeError(f"could not read {path}")
    feats = np.stack([frame_features(f) for f in frames])
    gray = np.stack([cv2.cvtColor(f, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255 for f in frames])
    motion = np.abs(np.diff(gray, axis=0)).mean((1, 2)) if len(gray) > 1 else np.zeros(1)
    lum = feats[:, 67]
    return np.concatenate([feats[0], feats.mean(0), [motion.mean(), motion.max(), lum.std()]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim", nargs="+", required=True, help="folders with <id>.mp4")
    ap.add_argument("--out", default="models/xsubj/features/lowlevel.npz")
    args = ap.parse_args()
    ids, emb = [], []
    for d in args.stim:
        for f in sorted(resolve(d).glob("*.mp4")):
            ids.append(str(int(f.stem)))          # "0035" -> "35", matching the EEG keys
            emb.append(video_features(f))
    out = resolve(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, ids=np.array(ids), emb=np.stack(emb).astype(np.float32))
    print(f"saved {len(ids)} videos x {emb[0].size} features -> {out}")


if __name__ == "__main__":
    main()
