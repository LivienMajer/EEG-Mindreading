"""Load and combine per-video embedding archives into a lookup bank."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.xsubj.eeg.adapter import synthetic_video_embeddings
from src.xsubj.utils import resolve


def _scalar(archive, name: str, default: str) -> str:
    return str(archive[name].item()) if name in archive else default


def _video_key(video_id) -> str:
    text = str(video_id)
    return str(int(text)) if text.isdigit() else text


@dataclass
class VideoBank:
    """Sorted video identifiers and their aligned embedding vectors."""

    ids: np.ndarray
    emb: np.ndarray

    @property
    def dim(self) -> int:
        return self.emb.shape[1]

    def lookup(self, video_ids) -> np.ndarray:
        requested = np.asarray(video_ids).astype(str)
        normalized = np.asarray([_video_key(value) for value in requested])
        positions = {video_id: i for i, video_id in enumerate(self.ids)}
        missing = sorted(set(normalized) - positions.keys())
        if missing:
            raise KeyError(f"Video IDs missing from embedding bank: {missing[:10]}")
        return np.asarray([positions[video_id] for video_id in normalized],
                          dtype=np.int64)


def load_video_bank(cfg: dict) -> VideoBank:
    """Load one or more standardized extractor archives and fuse model spaces.

    Archives from the same model (e.g. train and test splits) share one feature
    slot. Distinct model spaces are independently normalized and concatenated
    at equal norm, so CLIP and CLAP vectors are never mixed by raw scale.
    """
    if cfg["data"]["format"] == "synthetic":
        ids, embeddings = synthetic_video_embeddings(cfg["data"]["synthetic"])
        return VideoBank(ids.astype(str), embeddings.astype(np.float32))

    paths = cfg["data"]["video_emb"]
    if isinstance(paths, (str, Path)):
        paths = [paths]
    if not paths:
        raise ValueError("data.video_emb must contain at least one archive")

    slots: dict[str, dict] = {}
    for configured_path in paths:
        path = resolve(configured_path)
        if not path.is_file():
            raise FileNotFoundError(f"Video embedding archive not found: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if "video_ids" not in archive or "embeddings" not in archive:
                raise ValueError(
                    f"{path} must contain 'video_ids' and 'embeddings'; "
                    "re-extract with the current video/audio extractor.")
            ids = np.asarray([_video_key(value)
                              for value in archive["video_ids"]])
            embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
            if embeddings.ndim != 2 or len(ids) != len(embeddings):
                raise ValueError(
                    f"Invalid video_ids/embeddings shapes in {path}: "
                    f"{ids.shape}, {embeddings.shape}")
            if not np.isfinite(embeddings).all():
                raise ValueError(f"Non-finite embeddings in {path}")

            backend = _scalar(archive, "backend", path.stem)
            model = _scalar(archive, "model", "")
            weights = _scalar(
                archive, "pretrained",
                _scalar(archive, "checkpoint", ""))
            slot_name = ":".join(part for part in (backend, model, weights) if part)
            slot = slots.setdefault(slot_name, {
                "dim": embeddings.shape[1], "values": {}})
            if slot["dim"] != embeddings.shape[1]:
                raise ValueError(
                    f"Embedding dimension changed for model {slot_name}: "
                    f"{slot['dim']} vs {embeddings.shape[1]} in {path}")
            if len(set(ids.tolist())) != len(ids):
                raise ValueError(f"Duplicate video IDs in {path}")
            overlap = set(ids.tolist()) & slot["values"].keys()
            if overlap:
                raise ValueError(
                    f"Duplicate video IDs for model {slot_name} in {path}: "
                    f"{sorted(overlap)[:5]}")
            for video_id, embedding in zip(ids, embeddings):
                slot["values"][video_id] = embedding

    video_ids = sorted(set().union(
        *(slot["values"].keys() for slot in slots.values())))
    if not video_ids:
        raise ValueError("No video embeddings found in the configured archives")
    vectors = []
    scale = np.sqrt(len(slots))
    for video_id in video_ids:
        parts = []
        for slot in slots.values():
            value = slot["values"].get(video_id)
            if value is None:
                value = np.zeros(slot["dim"], dtype=np.float32)
            norm = np.linalg.norm(value)
            parts.append(value / max(float(norm), 1e-8) / scale)
        vectors.append(np.concatenate(parts))
    embeddings = np.stack(vectors).astype(np.float32)
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    invalid = np.flatnonzero(norms[:, 0] <= 1e-8)
    if len(invalid):
        raise ValueError(
            "All configured model embeddings are zero for video IDs: "
            f"{[video_ids[i] for i in invalid[:10]]}")
    embeddings /= norms
    return VideoBank(np.asarray(video_ids), embeddings)
