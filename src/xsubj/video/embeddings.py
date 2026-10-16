"""Frozen video embeddings produced by the video side of the pipeline (CLIP / InternVideo / ...).

Accepted .npz layouts:
  * `ids` (n_videos,) + `emb` (n_videos, d)
  * one key per video id -> (n_frames, d) or (d,), as written by
    `src.video_embeddings.encode_all_candidates`; frames are averaged.
Swapping the video encoder only means pointing `data.video_emb` at a different file;
a list of files combines feature sets.
"""
from __future__ import annotations

import numpy as np

from src.xsubj.eeg.adapter import synthetic_video_embeddings
from src.xsubj.utils import resolve


class VideoBank:
    def __init__(self, ids: np.ndarray, emb: np.ndarray, standardize: bool = False):
        emb = emb.astype(np.float32)
        if standardize:  # z-score each feature over all videos (unsupervised; needed for raw features)
            emb = (emb - emb.mean(0)) / (emb.std(0) + 1e-8)
        self.ids = np.asarray(ids).astype(str)
        self.emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.index = {v: i for i, v in enumerate(self.ids)}

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    def lookup(self, video_ids) -> np.ndarray:
        missing = [v for v in video_ids if v not in self.index]
        if missing:
            raise KeyError(f"{len(missing)} video ids have no embedding, e.g. {missing[:3]}")
        return np.array([self.index[v] for v in video_ids])


def load_video_bank(cfg: dict) -> VideoBank:
    if cfg["data"]["format"] == "synthetic":
        return VideoBank(*synthetic_video_embeddings(cfg["data"]["synthetic"]))
    std = cfg["data"].get("video_standardize", False)
    paths = cfg["data"]["video_emb"]
    if isinstance(paths, str):
        return VideoBank(*_load_file(paths), std)
    # several feature sets: z-score each, weight by 1/sqrt(dim) so every set counts equally, concatenate
    parts = [_load_file(p) for p in paths]
    ids = sorted(set.intersection(*(set(i) for i, _ in parts)), key=lambda v: (len(v), v))
    blocks = []
    for i, e in parts:
        e = e[[{v: k for k, v in enumerate(i)}[v] for v in ids]]
        e = (e - e.mean(0)) / (e.std(0) + 1e-8)
        blocks.append(e / np.sqrt(e.shape[1]))
    return VideoBank(np.array(ids), np.concatenate(blocks, 1), standardize=False)


def _load_file(path) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(resolve(path), allow_pickle=False)
    if set(z.files) == {"ids", "emb"}:
        return z["ids"].astype(str), z["emb"].astype(np.float32)
    ids = sorted(z.files)
    emb = np.stack([z[v].reshape(-1, z[v].shape[-1]).mean(0) for v in ids])  # mean over frames
    return np.array(ids), emb.astype(np.float32)
