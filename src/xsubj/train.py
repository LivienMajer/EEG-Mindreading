"""Contrastive training of the EEG encoder against frozen video embeddings.

    python -m src.xsubj.train --config configs/xsubj.yaml                  # trn splits of all subjects
    python -m src.xsubj.train --config configs/xsubj.yaml --holdout sub-01 # leave one subject out
    python -m src.xsubj.train --config configs/xsubj.yaml --finetune models/xsubj/checkpoints/all/model.pt --subject sub-05
    python -m src.xsubj.eval --config configs/xsubj.yaml --models encoder --mode official --set model.dual_projection=true
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader

from src.xsubj.eeg.dataset import CrossSubjectBatchSampler, TrialDataset, Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import retrieval_metrics, weighted_csls_similarity
from src.xsubj.nets.encoders import EEGEncoder
from src.xsubj.utils import get_device, load_config, resolve, seed_everything
from src.xsubj.video.embeddings import VideoBank, load_video_bank

DEFAULT_MULTI_HEAD_NAMES = ("videomae", "clip", "alexnet")
AVAILABLE_MULTI_HEAD_NAMES = (*DEFAULT_MULTI_HEAD_NAMES, "clap")


def configured_head_names(
    cfg: dict, available_names: set[str] | None = None
) -> tuple[str, ...]:
    names = tuple(cfg["model"].get("head_names", DEFAULT_MULTI_HEAD_NAMES))
    if len(names) < 2 or len(set(names)) != len(names):
        raise ValueError("Multi-head mode requires at least two unique head names")
    valid_names = set(AVAILABLE_MULTI_HEAD_NAMES)
    if available_names is not None:
        valid_names.update(
            name for name in available_names
            if name.startswith("videomae_layer"))
    unknown = set(names) - valid_names
    if unknown:
        raise ValueError(
            f"Unknown multi-head spaces {sorted(unknown)}; choose from "
            f"{', '.join(sorted(valid_names))}")
    return names


def split_by_video(trials: Trials, frac: float, seed: int) -> tuple[Trials, Trials]:
    """Hold out whole videos (all their trials) so validation measures generalization to new videos."""
    vids = np.unique(trials.y)
    rng = np.random.default_rng(seed)
    val_vids = rng.choice(vids, max(2, int(round(frac * len(vids)))), replace=False)
    m = np.isin(trials.y, val_vids)
    return trials.subset(~m), trials.subset(m)


@torch.no_grad()
def embed(model, X: np.ndarray, subj: np.ndarray, device, bs: int = 256) -> np.ndarray:
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i: i + bs]).to(device)
        s = torch.from_numpy(subj[i: i + bs]).long().to(device)
        out.append(model(x, s).float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def embed_heads(
    model: EEGEncoder, X: np.ndarray, subj: np.ndarray, device, bs: int = 256
) -> dict[str, np.ndarray]:
    model.eval()
    out = {name: [] for name in model.head_names}
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i: i + bs]).to(device)
        s = torch.from_numpy(subj[i: i + bs]).long().to(device)
        for name, embedding in model.forward_heads(x, s).items():
            out[name].append(embedding.float().cpu().numpy())
    return {name: np.concatenate(embeddings) for name, embeddings in out.items()}


@torch.no_grad()
def encode_features(
    model: EEGEncoder, X: np.ndarray, subj: np.ndarray, device, bs: int = 256
) -> np.ndarray:
    """Extract the frozen encoder's shared EEG feature vectors for downstream Ridge."""
    model.eval()
    out = []
    for i in range(0, len(X), bs):
        x = torch.from_numpy(X[i: i + bs]).to(device)
        s = torch.from_numpy(subj[i: i + bs]).long().to(device)
        out.append(model._features(x, s).float().cpu().numpy())
    return np.concatenate(out)


def embed_tta(
    model: EEGEncoder,
    X: np.ndarray,
    subj: np.ndarray,
    device,
    fs: float,
    offsets: list[float],
    duration: float,
) -> np.ndarray | dict[str, np.ndarray]:
    """Average normalized encoder outputs from overlapping windows in each EEG trial."""
    if not offsets:
        raise ValueError("TTA requires at least one temporal offset")
    if duration <= 0 or any(offset < 0 for offset in offsets):
        raise ValueError("TTA duration must be positive and offsets non-negative")
    if any(not np.isfinite(value) for value in [duration, *offsets]):
        raise ValueError("TTA duration and offsets must be finite")
    if not np.isclose(fs, 128.0):
        raise ValueError(f"TTA expected preprocessed 128 Hz EEG, got {fs} Hz")

    crop_size = int(round(duration * fs))
    starts = [int(round(offset * fs)) for offset in offsets]
    if crop_size < 1 or any(
            start + crop_size > X.shape[-1] for start in starts):
        raise ValueError(
            f"TTA crops of {duration:g}s at offsets {offsets} do not fit "
            f"inside the {X.shape[-1] / fs:g}s EEG window")
    if crop_size != model.backbone_in[1] and model.arch == "mlp":
        raise ValueError(
            "Temporal crop lengths that differ from training are not "
            "supported by the MLP encoder")

    crop_outputs = []
    for start in starts:
        crop = X[..., start:start + crop_size]
        output = (
            embed_heads(model, crop, subj, device)
            if model.multi_head else embed(model, crop, subj, device))
        crop_outputs.append(output)

    if model.multi_head:
        averaged = {
            name: np.mean(
                [output[name] for output in crop_outputs], axis=0)
            for name in model.head_names
        }
        return {
            name: values / np.maximum(
                np.linalg.norm(values, axis=1, keepdims=True), 1e-8)
            for name, values in averaged.items()
        }
    averaged = np.mean(crop_outputs, axis=0)
    return averaged / np.maximum(
        np.linalg.norm(averaged, axis=1, keepdims=True), 1e-8)


def rank_score(pred, y, bank_emb) -> float:
    """1 = true video always ranked first, 0.5 = chance. Smoother than top-1 for early stopping."""
    pool = np.unique(y)
    sim = pred @ bank_emb[pool].T
    s_true = sim[np.arange(len(y)), np.searchsorted(pool, y)]
    return float(1 - ((sim > s_true[:, None]).sum(1) / max(1, len(pool) - 1)).mean())


@torch.no_grad()
def target_embeddings(
    model: EEGEncoder, bank: VideoBank, device
) -> np.ndarray | dict[str, np.ndarray]:
    if model.multi_head:
        if bank.spaces is None or any(
                name not in bank.spaces for name in model.head_names):
            raise ValueError(
                "Multi-head training requires separate VideoMAE and CLIP "
                "embedding archives")
        return {name: bank.spaces[name] for name in model.head_names}
    if not model.gated_fusion:
        return bank.emb
    if bank.visual_emb is None or bank.audio_emb is None:
        raise ValueError(
            "Gated fusion needs separate visual and audio embedding archives")
    visual = torch.from_numpy(bank.visual_emb).to(device)
    audio = torch.from_numpy(bank.audio_emb).to(device)
    return model.fuse_video_embeddings(visual, audio).float().cpu().numpy()


def contrastive_loss(z, y, E, train_pool, scale, t: dict):
    if t["negatives"] == "all":
        logits = scale * z @ E[train_pool].T
        loss = F.cross_entropy(logits, torch.searchsorted(train_pool, y))
    else:  # in-batch, with duplicate videos merged so they are never treated as negatives
        u, inv = torch.unique(y, return_inverse=True)
        logits = scale * z @ E[u].T                                 # (B, U)
        pos = F.one_hot(inv, len(u)).T.bool()                       # (U, B)
        lt = logits.T
        v2e = -(torch.logsumexp(lt.masked_fill(~pos, -1e4), 1) - torch.logsumexp(lt, 1)).mean()
        loss = (F.cross_entropy(logits, inv) + v2e) / 2
    if t["mse_weight"]:
        loss = loss + t["mse_weight"] * (1 - (z * E[y]).sum(-1)).mean()
    return loss


def cross_subject_contrastive_loss(
    z: torch.Tensor,
    y: torch.Tensor,
    subjects: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    """Align same-video embeddings from different subjects and repel other videos."""
    if z.ndim != 2 or len(z) < 2:
        raise ValueError("Cross-subject contrast needs a 2D batch of at least two trials")
    same_video = y[:, None] == y[None, :]
    different_subject = subjects[:, None] != subjects[None, :]
    not_self = ~torch.eye(len(z), dtype=torch.bool, device=z.device)
    positives = same_video & different_subject & not_self
    different_video = ~same_video
    denominator = (positives | different_video) & not_self
    positive_counts = positives.sum(dim=1)
    anchors = positive_counts > 0
    if not anchors.any():
        raise ValueError(
            "No cross-subject same-video positives found in a batch; "
            "use CrossSubjectBatchSampler")
    logits = scale.float() * (z.float() @ z.float().T)
    logits = logits - logits.max(dim=1, keepdim=True).values.detach()
    log_prob = logits - torch.logsumexp(
        logits.masked_fill(~denominator, -torch.inf),
        dim=1,
        keepdim=True,
    )
    positive_log_prob = log_prob.masked_fill(~positives, 0.0).sum(dim=1)
    return -(positive_log_prob[anchors] / positive_counts[anchors]).mean()


def _head_weights(cfg: dict, names: tuple[str, ...]) -> dict[str, float]:
    configured = cfg["model"].get("head_weights", {})
    weights = {name: float(configured.get(name, 1.0)) for name in names}
    if any(not np.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("model.head_weights must be finite and non-negative")
    if sum(weights.values()) <= 0:
        raise ValueError("At least one model.head_weights value must be positive")
    return weights


def _multi_head_rank_score(predictions, y, embeddings, weights) -> float:
    pool = np.unique(y)
    candidate_embeddings = {
        name: embeddings[name][pool] for name in predictions}
    scores = weighted_csls_similarity(
        predictions, candidate_embeddings, weights, k=10)
    true_col = np.searchsorted(pool, y)
    outranked = (
        scores > scores[np.arange(len(y)), true_col, None]).sum(axis=1)
    return float(1 - (outranked / max(1, len(pool) - 1)).mean())


def leave_one_subject_out_eeg_prototypes(trials: Trials) -> np.ndarray:
    """Build each trial's EEG target from same-video trials of other subjects only."""
    groups: dict[int, list[int]] = {}
    for index, video_id in enumerate(trials.y):
        groups.setdefault(int(video_id), []).append(index)
    targets = np.empty_like(trials.X, dtype=np.float32)
    for video_id, indices in groups.items():
        for index in indices:
            other_indices = [
                other for other in indices
                if trials.subject[other] != trials.subject[index]]
            if not other_indices:
                raise ValueError(
                    "EEG prototype training needs another subject for every "
                    f"training trial; video {video_id} has no cross-subject target")
            target = trials.X[other_indices].mean(axis=0)
            flat = target.reshape(-1)
            norm = np.linalg.norm(flat)
            if norm <= 1e-8:
                raise ValueError(
                    f"EEG prototype for video {video_id} has zero norm")
            targets[index] = target / norm
    return targets.reshape(len(targets), -1)


def fit(cfg, train: Trials, val: Trials | None, bank: VideoBank, subject_list, fs,
        params=None, log=print) -> EEGEncoder:
    """Train a new encoder (or, with `params`, only those parameters of `cfg['_model']`)."""
    t, device = cfg["train"], get_device()
    _, n_ch, n_t = train.X.shape
    head_dims = None
    if cfg["model"].get("multi_head", False):
        head_names = configured_head_names(
            cfg, set(bank.spaces or {}))
        if bank.spaces is None or any(name not in bank.spaces for name in head_names):
            raise ValueError(
                "model.multi_head requires a separate embedding archive for "
                f"each configured head: {', '.join(head_names)}")
        head_dims = {
            name: bank.spaces[name].shape[1] for name in head_names}
    model = cfg.get("_model") or EEGEncoder(n_ch, n_t, bank.dim, len(subject_list), cfg["model"], t["temperature"],
                                            fs=fs, tmin=cfg["preprocess"]["tmin"],
                                            visual_dim=(bank.visual_emb.shape[1]
                                                        if bank.visual_emb is not None else None),
                                            audio_dim=(bank.audio_emb.shape[1]
                                                       if bank.audio_emb is not None else None),
                                            head_dims=head_dims)
    gated_fusion = model.gated_fusion
    multi_head = model.multi_head
    if gated_fusion and (bank.visual_emb is None or bank.audio_emb is None):
        raise ValueError(
            "Gated fusion requires separate visual and audio archives in "
            "data.video_emb")
    model.to(device)
    E = None if gated_fusion or multi_head else torch.from_numpy(bank.emb).to(device)
    E_visual = (
        torch.from_numpy(bank.visual_emb).to(device)
        if gated_fusion and bank.visual_emb is not None else None)
    E_audio = (
        torch.from_numpy(bank.audio_emb).to(device)
        if gated_fusion and bank.audio_emb is not None else None)
    dual_projection = model.dual_projection
    audio_loss_weight = float(t.get("audio_loss_weight", 1.0))
    if dual_projection:
        if bank.dim % 2:
            raise ValueError(
                "Dual projection expects equally sized video and audio "
                f"embedding spaces; got combined dimension {bank.dim}")
        if audio_loss_weight <= 0:
            raise ValueError("train.audio_loss_weight must be greater than zero")
        E_video, E_audio = (F.normalize(part, dim=-1) for part in E.chunk(2, dim=-1))
    E_heads = (
        {name: torch.from_numpy(bank.spaces[name]).to(device)
         for name in model.head_names}
        if multi_head and bank.spaces is not None else {})
    head_weights = _head_weights(cfg, model.head_names) if multi_head else {}
    xsubj_weight = float(t.get("xsubj_contrastive_weight", 0.0))
    if not np.isfinite(xsubj_weight) or xsubj_weight < 0:
        raise ValueError(
            "train.xsubj_contrastive_weight must be finite and non-negative")
    if xsubj_weight and len(np.unique(train.subj)) < 2:
        raise ValueError(
            "Cross-subject contrast requires training trials from at least "
            "two subjects")
    prototype_weight = float(t.get("eeg_prototype_weight", 0.0))
    if not np.isfinite(prototype_weight) or prototype_weight < 0:
        raise ValueError(
            "train.eeg_prototype_weight must be finite and non-negative")
    if prototype_weight and model.eeg_prototype_project is None:
        raise ValueError(
            "EEG prototype loss requires model.eeg_prototype_head=true")
    if prototype_weight and len(np.unique(train.subject)) < 2:
        raise ValueError(
            "EEG prototype loss requires trials from at least two subjects")
    if params is not None:
        prototype_weight = 0.0
    train_pool = torch.from_numpy(np.unique(train.y)).to(device)
    opt = torch.optim.AdamW(params or model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t["epochs"])
    prototype_targets = (
        leave_one_subject_out_eeg_prototypes(train)
        if prototype_weight else None)
    dataset = TrialDataset(
        train, t["aug"], fs, t.get("xsubj_avg"), prototype_targets)
    if xsubj_weight:
        loader = DataLoader(
            dataset,
            batch_sampler=CrossSubjectBatchSampler(train, t["batch_size"]))
    else:
        loader = DataLoader(
            dataset,
            batch_size=t["batch_size"],
            shuffle=True,
            drop_last=len(train) > t["batch_size"])
    use_amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_state, bad = -math.inf, None, 0
    for epoch in range(t["epochs"]):
        model.train(params is None)  # fine-tuning keeps BatchNorm/Dropout of the frozen backbone fixed
        total = 0.0
        total_xsubj = 0.0
        total_prototype = 0.0
        subject_dropout = (
            float(t.get("subject_dropout", 0.0)) if params is None else 0.0)
        if not np.isfinite(subject_dropout) or not 0 <= subject_dropout < 1:
            raise ValueError("train.subject_dropout must be in [0, 1)")
        for batch in loader:
            if prototype_weight:
                x, y, s, eeg_prototype = batch
                eeg_prototype = eeg_prototype.to(device)
            else:
                x, y, s = batch
            x, y, s = x.to(device), y.to(device), s.to(device)
            model_subjects = s
            if subject_dropout:
                dropped = torch.rand(len(s), device=device) < subject_dropout
                model_subjects = s.masked_fill(dropped, -1)
            with torch.autocast(device.type, enabled=use_amp):
                if dual_projection:
                    z_video, z_audio = (
                        z.float() for z in model.forward_modalities(x, model_subjects))
                elif multi_head:
                    z_heads = {
                        name: z.float()
                        for name, z in model.forward_heads(x, model_subjects).items()}
                else:
                    z = model(x, model_subjects).float()
            scale = model.bounded_logit_scale()
            if dual_projection:
                video_loss = contrastive_loss(
                    z_video, y, E_video, train_pool, scale, t)
                audio_loss = contrastive_loss(
                    z_audio, y, E_audio, train_pool, scale, t)
                loss = (video_loss + audio_loss_weight * audio_loss) / (
                    1 + audio_loss_weight)
            elif multi_head:
                head_losses = [
                    contrastive_loss(
                        z_heads[name], y, E_heads[name], train_pool, scale, t)
                    for name in model.head_names]
                loss = torch.stack(head_losses).mean()
            else:
                if gated_fusion:
                    assert E_visual is not None and E_audio is not None
                    E = model.fuse_video_embeddings(E_visual, E_audio)
                assert E is not None
                loss = contrastive_loss(z, y, E, train_pool, scale, t)
            if prototype_weight:
                source = (
                    z_heads[model.head_names[0]] if multi_head else
                    z_video if dual_projection else z)
                prototype_prediction = model.project_eeg_prototype(source.float())
                prototype_loss = (
                    1 - (prototype_prediction * eeg_prototype).sum(dim=-1)
                ).mean()
                loss = loss + prototype_weight * prototype_loss
                total_prototype += prototype_loss.item() * len(x)
            if xsubj_weight:
                if multi_head:
                    xsubj_loss = torch.stack([
                        cross_subject_contrastive_loss(
                            z_heads[name], y, s, scale)
                        for name in model.head_names
                    ]).mean()
                elif dual_projection:
                    xsubj_loss = torch.stack([
                        cross_subject_contrastive_loss(
                            z_video, y, s, scale),
                        cross_subject_contrastive_loss(
                            z_audio, y, s, scale),
                    ]).mean()
                else:
                    xsubj_loss = cross_subject_contrastive_loss(z, y, s, scale)
                loss = loss + xsubj_weight * xsubj_loss
                total_xsubj += xsubj_loss.item() * len(x)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            model.clamp_logit_scale_()
            total += loss.item() * len(x)
        sched.step()

        if val is not None and len(val):
            if multi_head:
                score = _multi_head_rank_score(
                    embed_heads(model, val.X, val.subj, device), val.y,
                    target_embeddings(model, bank, device), head_weights)
            else:
                score = rank_score(
                    embed(model, val.X, val.subj, device), val.y,
                    target_embeddings(model, bank, device))
            if score > best:
                best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if epoch % 10 == 0 or bad == 0:
                log(f"epoch {epoch:3d}  loss {total / len(train):.4f}  val rank-score {score:.3f}"
                    f"{f'  xsubj {total_xsubj / len(train):.4f}' if xsubj_weight else ''}"
                    f"{f'  eeg-proto {total_prototype / len(train):.4f}' if prototype_weight else ''}"
                    f"{'  *' if bad == 0 else ''}")
            if bad >= t["patience"]:
                log(f"early stop at epoch {epoch}, best val rank-score {best:.3f}")
                break
        elif epoch % 10 == 0:
            log(f"epoch {epoch:3d}  loss {total / len(train):.4f}"
                f"{f'  xsubj {total_xsubj / len(train):.4f}' if xsubj_weight else ''}"
                f"{f'  eeg-proto {total_prototype / len(train):.4f}' if prototype_weight else ''}")
    if best_state is not None:
        model.load_state_dict(best_state)
    return model


def train_split(cfg, data, bank, holdout: str | None, log=print):
    """Train on the `trn` split of every subject except `holdout`; validation = held-out training
    videos. Test splits are never used for training (challenge rule)."""
    train_subj = [d for d in data if d.split == "trn" and d.subject != holdout and d.video_id is not None]
    subject_list = sorted({d.subject for d in train_subj})
    trials = Trials(train_subj, bank, subject_list)
    tr, va = split_by_video(trials, cfg["train"]["val_frac"], cfg["seed"])
    log(f"train {len(tr)} trials / {len(np.unique(tr.y))} videos, val {len(va)} trials, "
        f"subjects {subject_list}")
    model = fit(cfg, tr, va, bank, subject_list, data[0].fs, log=log)
    validation_video_ids = sorted(set(va.video_id.astype(str).tolist()))
    return model, subject_list, validation_video_ids


def finetune_subject(
    cfg,
    model: EEGEncoder,
    trials: Trials,
    bank,
    fs,
    subject: str,
    subject_list: list[str],
    validation_video_ids: list[str],
    learning_rate: float,
    epochs: int,
    patience: int,
    log=print,
) -> EEGEncoder:
    """Adapt one subject's channel transform using globally held-out validation videos."""
    sl = model.subject_layer
    if sl is None:
        raise ValueError("model has no subject layer to fine-tune")
    if subject in subject_list:
        subject_index = subject_list.index(subject)
    else:
        subject_index = len(subject_list)
        with torch.no_grad():
            sl.W = torch.nn.Parameter(torch.cat(
                [sl.W, sl.W.mean(0, keepdim=True)]))
        subject_list.append(subject)
    if not np.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("Fine-tuning learning rate must be finite and positive")
    if epochs < 1 or patience < 1:
        raise ValueError("Fine-tuning epochs and patience must be positive")
    trials.subj[:] = subject_index
    validation_mask = np.isin(
        trials.video_id.astype(str), np.asarray(validation_video_ids, dtype=str))
    if not validation_mask.any():
        raise ValueError(
            f"No globally held-out validation trials are available for {subject}; "
            "train the joint checkpoint with a larger validation fraction")
    train_trials = trials.subset(~validation_mask)
    validation_trials = trials.subset(validation_mask)
    if not len(train_trials):
        raise ValueError(f"No fine-tuning trials remain for {subject}")

    for p in model.parameters():
        p.requires_grad_(False)
    sl.W.requires_grad_(True)
    gradient_mask = torch.zeros_like(sl.W, device=get_device())
    gradient_mask[subject_index] = 1
    mask_hook = sl.W.register_hook(lambda grad: grad * gradient_mask)
    ft_cfg = {
        **cfg,
        "_model": model,
        "train": {
            **cfg["train"],
            "lr": learning_rate,
            "weight_decay": 0.0,
            "epochs": epochs,
            "patience": patience,
            "xsubj_contrastive_weight": 0.0,
            "eeg_prototype_weight": 0.0,
            "subject_dropout": 0.0,
        },
    }
    log(
        f"fine-tune {subject}: {len(train_trials)} trials, "
        f"{len(validation_trials)} globally held-out validation trials, "
        f"adapter index {subject_index}, lr {learning_rate:g}")
    try:
        return fit(
            ft_cfg, train_trials, validation_trials, bank, subject_list, fs,
            params=[sl.W], log=log)
    finally:
        mask_hook.remove()


def save_checkpoint(
    path: Path, model, cfg, subject_list, fs,
    validation_video_ids: list[str] | None = None,
):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": model.state_dict(),
        "cfg": cfg,
        "subject_list": subject_list,
        "fs": fs,
        "n_channels": model.backbone_in[0],
        "n_times": model.backbone_in[1],
        "validation_video_ids": validation_video_ids,
    }, path)


def load_checkpoint(path, bank: VideoBank):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    n_subj = ck["state_dict"]["subject_layer.W"].shape[0] if cfg["model"]["subject_layer"] else 0
    head_dims = None
    if cfg["model"].get("multi_head", False):
        head_names = configured_head_names(
            cfg, set(bank.spaces or {}))
        if bank.spaces is None or any(
                name not in bank.spaces
                for name in head_names):
            raise ValueError(
                "Checkpoint uses multi_head but these embedding spaces were "
                f"not all loaded: {', '.join(head_names)}")
        head_dims = {
            name: bank.spaces[name].shape[1] for name in head_names}
    model = EEGEncoder(
        ck["n_channels"], ck["n_times"], bank.dim, n_subj, cfg["model"],
        fs=ck["fs"], tmin=cfg["preprocess"]["tmin"],
        visual_dim=(bank.visual_emb.shape[1]
                    if bank.visual_emb is not None else None),
        audio_dim=(bank.audio_emb.shape[1]
                   if bank.audio_emb is not None else None),
        head_dims=head_dims)
    model.load_state_dict(ck["state_dict"])
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--holdout", default=None, help="subject to leave out (LOSO)")
    ap.add_argument(
        "--finetune", default=None,
        help="joint checkpoint to adapt using --subject or --subjects")
    ap.add_argument("--subject", default=None)
    ap.add_argument(
        "--subjects", nargs="+", default=None,
        help="subjects to fine-tune; use 'all' for every subject in the checkpoint")
    ap.add_argument("--finetune-lr", type=float, default=5e-5,
                    help="learning rate for subject-adapter fine-tuning")
    ap.add_argument("--finetune-epochs", type=int, default=50,
                    help="maximum subject-adapter fine-tuning epochs")
    ap.add_argument("--finetune-patience", type=int, default=8,
                    help="early-stopping patience for subject-adapter fine-tuning")
    ap.add_argument("--video-emb", nargs="+", default=None,
                    help="override data.video_emb with one or more archives")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. model.dual_projection=true")
    ap.add_argument("--out", default=None, help="run name under checkpoints/")
    args = ap.parse_args()

    if args.finetune:
        if args.holdout:
            raise SystemExit("--holdout cannot be combined with --finetune.")
        if args.set or args.video_emb:
            raise SystemExit(
                "Fine-tuning inherits model, preprocessing, and embedding "
                "settings from the checkpoint; omit --set and --video-emb.")
        pretrained = torch.load(
            resolve(args.finetune), map_location="cpu", weights_only=False)
        cfg = pretrained["cfg"]
        validation_video_ids = pretrained.get("validation_video_ids")
        if not validation_video_ids:
            raise SystemExit(
                "This checkpoint does not record its global validation videos. "
                "Create a joint checkpoint with the current train command "
                "before subject fine-tuning.")
        if args.subject and args.subjects:
            raise SystemExit("Use either --subject or --subjects, not both.")
        if args.subject:
            subjects_to_finetune = [args.subject]
        elif args.subjects:
            if "all" in args.subjects and args.subjects != ["all"]:
                raise SystemExit("Use 'all' by itself in --subjects.")
            if len(set(args.subjects)) != len(args.subjects):
                raise SystemExit("--subjects must not contain duplicates.")
            subjects_to_finetune = (
                list(pretrained["subject_list"])
                if args.subjects == ["all"] else args.subjects)
        else:
            raise SystemExit(
                "Fine-tuning requires --subject NAME or --subjects all.")
    else:
        if args.subject or args.subjects:
            raise SystemExit("--subject/--subjects require --finetune.")
        cfg = load_config(args.config)
        validation_video_ids = None
        subjects_to_finetune = []
        for kv in args.set:
            key, value = kv.split("=", 1)
            *path, last = key.split(".")
            node = cfg
            for k in path:
                node = node[k]
            node[last] = yaml.safe_load(value)
        if args.video_emb:
            cfg["data"]["video_emb"] = (
                args.video_emb[0] if len(args.video_emb) == 1 else args.video_emb)
    seed_everything(cfg["seed"])
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)
    fs = data[0].fs

    if args.finetune:
        model, ck = load_checkpoint(args.finetune, bank)
        model.to(get_device())
        subject_list = list(ck["subject_list"])
        for subject in subjects_to_finetune:
            matching_data = [
                d for d in data
                if d.subject == subject and d.split == "trn"
                and d.video_id is not None]
            if not matching_data:
                raise SystemExit(
                    f"No labeled training trials found for {subject}.")
            trials = Trials(matching_data, bank, subject_list)
            model = finetune_subject(
                cfg, model, trials, bank, fs, subject, subject_list,
                validation_video_ids, args.finetune_lr,
                args.finetune_epochs, args.finetune_patience)
        suffix = (
            "all" if args.subjects == ["all"]
            else "_".join(subjects_to_finetune))
        run = args.out or f"{Path(args.finetune).parent.name}_ft_{suffix}"
    else:
        model, subject_list, validation_video_ids = train_split(
            cfg, data, bank, args.holdout)
        run = args.out or (f"loso_{args.holdout}" if args.holdout else "all")

    out = resolve("models/xsubj/checkpoints") / run / "model.pt"
    save_checkpoint(
        out, model, cfg, subject_list, fs, validation_video_ids)
    print(f"saved {out}")

    if args.holdout:
        d = next(d for d in data if d.subject == args.holdout and d.video_id is not None
                 and d.split == ("tst" if any(x.split == "tst" for x in data) else "trn"))
        test = Trials([d], bank, subject_list)
        if model.multi_head:
            score = _multi_head_rank_score(
                embed_heads(model, test.X, test.subj, get_device()),
                test.y, target_embeddings(model, bank, get_device()),
                _head_weights(cfg, model.head_names))
            print(json.dumps({"weighted_csls_rank_score": score}, indent=2))
        else:
            pred = embed(model, test.X, test.subj, get_device())
            print(json.dumps(retrieval_metrics(pred, test.y, bank.emb, cfg["eval"]["k_candidates"]), indent=2))


if __name__ == "__main__":
    main()
