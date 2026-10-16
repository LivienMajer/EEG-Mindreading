"""Frozen video embeddings produced by the video side of the pipeline (CLIP / InternVideo / ...).

Accepted .npz layouts:
  * `ids` (n_videos,) + `emb` (n_videos, d)
  * one key per video id -> (n_frames, d) or (d,), as written by
    `src.video_embeddings.encode_all_candidates`; frames are averaged.
Swapping the video encoder only means pointing `data.video_emb` at a different file.
"""
from __future__ import annotations

import numpy as np

from src.xsubj.eeg.adapter import synthetic_video_embeddings
from src.xsubj.utils import resolve


class VideoBank:
    def __init__(self, ids: np.ndarray, emb: np.ndarray):
        emb = emb.astype(np.float32)
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
    z = np.load(resolve(cfg["data"]["video_emb"]), allow_pickle=True)
    if set(z.files) == {"ids", "emb"}:
        return VideoBank(z["ids"], z["emb"])
    ids = sorted(z.files)
    emb = np.stack([z[v].reshape(-1, z[v].shape[-1]).mean(0) for v in ids])  # mean over frames
    return VideoBank(np.array(ids), emb)
