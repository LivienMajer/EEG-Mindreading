"""Load and combine per-video embedding archives into a lookup bank."""
from __future__ import annotations

import re
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


def _legacy_identity(path: Path) -> tuple[str, str, str]:
    """Infer one shared feature slot for legacy train/test archive filenames."""
    stem = re.sub(r"^(?:emb_)?(?:train|test)__?", "", path.stem)
    stem = re.sub(r"_(?:train|test)$", "", stem)
    stem = re.sub(r"_pooled$", "", stem)
    if not stem:
        raise ValueError(f"Cannot infer model identity from legacy archive name: {path}")
    return "legacy", stem, ""


def _feature_modality(
    backend: str, model: str, weights: str, path: Path
) -> str:
    identity = " ".join((backend, model, weights, path.stem)).lower()
    is_audio = "clap" in identity or "audio" in identity
    is_visual = any(
        token in identity
        for token in ("open_clip", "clip", "videomae", "slowfast", "x3d"))
    if is_audio and not is_visual:
        return "audio"
    if is_visual and not is_audio:
        return "visual"
    return "unknown"


@dataclass
class VideoBank:
    """Sorted video identifiers and their aligned embedding vectors."""

    ids: np.ndarray
    emb: np.ndarray
    visual_emb: np.ndarray | None = None
    audio_emb: np.ndarray | None = None
    spaces: dict[str, np.ndarray] | None = None

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
        feature_layer = 0
        with np.load(path, allow_pickle=False) as archive:
            if "video_ids" in archive and "embeddings" in archive:
                ids = np.asarray([_video_key(value)
                                  for value in archive["video_ids"]])
                embeddings = np.asarray(archive["embeddings"], dtype=np.float32)
                backend = _scalar(archive, "backend", path.stem)
                model = _scalar(archive, "model", "")
                weights = _scalar(
                    archive, "pretrained",
                    _scalar(archive, "checkpoint", ""))
                feature_layer = int(_scalar(archive, "feature_layer", "0"))
                if feature_layer < 0:
                    raise ValueError(
                        f"Invalid feature_layer {feature_layer} in {path}")
            else:
                grouped: dict[str, list[tuple[int, np.ndarray]]] = {}
                for key in archive.files:
                    match = re.fullmatch(r"(\d+)(?:_f(\d+))?", key)
                    if match is None:
                        raise ValueError(
                            f"{path} is not a supported video embedding archive: "
                            "expected 'video_ids'/'embeddings' arrays or numeric "
                            "video/frame keys.")
                    video_id = _video_key(match.group(1))
                    frame_id = int(match.group(2) or 0)
                    grouped.setdefault(video_id, []).append(
                        (frame_id, np.asarray(archive[key], dtype=np.float32)))
                if not grouped:
                    raise ValueError(f"No video embeddings found in {path}")
                ids = np.asarray(sorted(grouped))
                pooled = []
                for video_id in ids:
                    frames = sorted(grouped[video_id], key=lambda item: item[0])
                    shapes = {frame.shape for _, frame in frames}
                    if len(shapes) != 1:
                        raise ValueError(
                            f"Frame embedding shapes differ for video {video_id} "
                            f"in {path}: {sorted(shapes)}")
                    pooled.append(np.mean(np.stack(
                        [frame for _, frame in frames]), axis=0))
                embeddings = np.stack(pooled).astype(np.float32)
                backend, model, weights = _legacy_identity(path)

            if embeddings.ndim != 2 or len(ids) != len(embeddings):
                raise ValueError(
                    f"Invalid video_ids/embeddings shapes in {path}: "
                    f"{ids.shape}, {embeddings.shape}")
            if not np.isfinite(embeddings).all():
                raise ValueError(f"Non-finite embeddings in {path}")

            slot_parts = [part for part in (backend, model, weights) if part]
            if feature_layer:
                slot_parts.append(f"layer{feature_layer}")
            slot_name = ":".join(slot_parts)
            slot = slots.setdefault(slot_name, {
                "dim": embeddings.shape[1], "values": {},
                "modality": _feature_modality(
                    backend, model, weights, path),
                "feature_layer": feature_layer})
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
    if cfg.get("model", {}).get("gated_fusion", False):
        unknown_slots = [
            name for name, slot in slots.items()
            if slot["modality"] == "unknown"]
        if unknown_slots:
            raise ValueError(
                "Gated fusion cannot classify these embedding archives as "
                f"visual or audio: {unknown_slots}. Use separate archives "
                "with model/backend metadata (e.g. OpenCLIP/VideoMAE and "
                "LAION CLAP).")
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

    def modality_embeddings(modality: str) -> np.ndarray | None:
        modality_slots = [
            slot for slot in slots.values()
            if slot["modality"] == modality]
        if not modality_slots:
            return None
        rows = []
        for video_id in video_ids:
            parts = []
            for slot in modality_slots:
                value = slot["values"].get(video_id)
                if value is None:
                    value = np.zeros(slot["dim"], dtype=np.float32)
                norm = float(np.linalg.norm(value))
                parts.append(
                    value / max(norm, 1e-8) / np.sqrt(len(modality_slots)))
            rows.append(np.concatenate(parts))
        result = np.stack(rows).astype(np.float32)
        row_norms = np.linalg.norm(result, axis=1, keepdims=True)
        np.divide(result, row_norms, out=result, where=row_norms > 1e-8)
        return result

    head_slots: dict[str, dict] = {}
    for name, slot in slots.items():
        identity = name.lower()
        if "videomae" in identity:
            layer = slot["feature_layer"]
            head_name = f"videomae_layer{layer}" if layer else "videomae"
        elif "alexnet" in identity or "fc6" in identity:
            head_name = "alexnet"
        elif "clap" in identity or "audio" in identity:
            head_name = "clap"
        elif "clip" in identity:
            head_name = "clip"
        else:
            continue
        if head_name in head_slots:
            raise ValueError(
                f"Multiple embedding archives map to the {head_name!r} "
                "head; configure one archive/model space per head.")
        head_slots[head_name] = slot

    spaces: dict[str, np.ndarray] = {}
    for head_name, slot in head_slots.items():
        rows = []
        for video_id in video_ids:
            value = slot["values"].get(video_id)
            if value is None:
                value = np.zeros(slot["dim"], dtype=np.float32)
            norm = float(np.linalg.norm(value))
            rows.append(value / max(norm, 1e-8))
        spaces[head_name] = np.stack(rows).astype(np.float32)

    return VideoBank(
        np.asarray(video_ids), embeddings,
        visual_emb=modality_embeddings("visual"),
        audio_emb=modality_embeddings("audio"),
        spaces=spaces)
