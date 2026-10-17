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

from src.xsubj.eeg.dataset import TrialDataset, Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import retrieval_metrics
from src.xsubj.nets.encoders import EEGEncoder
from src.xsubj.utils import get_device, load_config, resolve, seed_everything
from src.xsubj.video.embeddings import VideoBank, load_video_bank


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


def rank_score(pred, y, bank_emb) -> float:
    """1 = true video always ranked first, 0.5 = chance. Smoother than top-1 for early stopping."""
    pool = np.unique(y)
    sim = pred @ bank_emb[pool].T
    s_true = sim[np.arange(len(y)), np.searchsorted(pool, y)]
    return float(1 - ((sim > s_true[:, None]).sum(1) / max(1, len(pool) - 1)).mean())


@torch.no_grad()
def target_embeddings(model: EEGEncoder, bank: VideoBank, device) -> np.ndarray:
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


def fit(cfg, train: Trials, val: Trials | None, bank: VideoBank, subject_list, fs,
        params=None, log=print) -> EEGEncoder:
    """Train a new encoder (or, with `params`, only those parameters of `cfg['_model']`)."""
    t, device = cfg["train"], get_device()
    _, n_ch, n_t = train.X.shape
    model = cfg.get("_model") or EEGEncoder(n_ch, n_t, bank.dim, len(subject_list), cfg["model"], t["temperature"],
                                            fs=fs, tmin=cfg["preprocess"]["tmin"],
                                            visual_dim=(bank.visual_emb.shape[1]
                                                        if bank.visual_emb is not None else None),
                                            audio_dim=(bank.audio_emb.shape[1]
                                                       if bank.audio_emb is not None else None))
    gated_fusion = model.gated_fusion
    if gated_fusion and (bank.visual_emb is None or bank.audio_emb is None):
        raise ValueError(
            "Gated fusion requires separate visual and audio archives in "
            "data.video_emb")
    model.to(device)
    E = None if gated_fusion else torch.from_numpy(bank.emb).to(device)
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
    train_pool = torch.from_numpy(np.unique(train.y)).to(device)
    opt = torch.optim.AdamW(params or model.parameters(), lr=t["lr"], weight_decay=t["weight_decay"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, t["epochs"])
    loader = DataLoader(TrialDataset(train, t["aug"], fs, t.get("xsubj_avg")), batch_size=t["batch_size"], shuffle=True,
                        drop_last=len(train) > t["batch_size"])
    use_amp = t["amp"] and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    best, best_state, bad = -math.inf, None, 0
    for epoch in range(t["epochs"]):
        model.train(params is None)  # fine-tuning keeps BatchNorm/Dropout of the frozen backbone fixed
        total = 0.0
        for x, y, s in loader:
            x, y, s = x.to(device), y.to(device), s.to(device)
            with torch.autocast(device.type, enabled=use_amp):
                if dual_projection:
                    z_video, z_audio = (
                        z.float() for z in model.forward_modalities(x, s))
                else:
                    z = model(x, s).float()
            scale = model.logit_scale.clamp(max=math.log(100)).exp()
            if dual_projection:
                video_loss = contrastive_loss(
                    z_video, y, E_video, train_pool, scale, t)
                audio_loss = contrastive_loss(
                    z_audio, y, E_audio, train_pool, scale, t)
                loss = (video_loss + audio_loss_weight * audio_loss) / (
                    1 + audio_loss_weight)
            else:
                if gated_fusion:
                    assert E_visual is not None and E_audio is not None
                    E = model.fuse_video_embeddings(E_visual, E_audio)
                assert E is not None
                loss = contrastive_loss(z, y, E, train_pool, scale, t)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total += loss.item() * len(x)
        sched.step()

        if val is not None and len(val):
            score = rank_score(
                embed(model, val.X, val.subj, device), val.y,
                target_embeddings(model, bank, device))
            if score > best:
                best, best_state, bad = score, copy.deepcopy(model.state_dict()), 0
            else:
                bad += 1
            if epoch % 10 == 0 or bad == 0:
                log(f"epoch {epoch:3d}  loss {total / len(train):.4f}  val rank-score {score:.3f}"
                    f"{'  *' if bad == 0 else ''}")
            if bad >= t["patience"]:
                log(f"early stop at epoch {epoch}, best val rank-score {best:.3f}")
                break
        elif epoch % 10 == 0:
            log(f"epoch {epoch:3d}  loss {total / len(train):.4f}")
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
    return model, subject_list


def finetune_subject(cfg, model: EEGEncoder, trials: Trials, bank, fs, log=print) -> EEGEncoder:
    """Partially held-out subject: add a subject-layer slot (init = mean) and train only that."""
    sl = model.subject_layer
    if sl is None:
        raise ValueError("model has no subject layer to fine-tune")
    with torch.no_grad():
        sl.W = torch.nn.Parameter(torch.cat([sl.W, sl.W.mean(0, keepdim=True)]))
    for p in model.parameters():
        p.requires_grad_(False)
    sl.W.requires_grad_(True)
    trials.subj[:] = sl.W.shape[0] - 1
    tr, va = split_by_video(trials, cfg["train"]["val_frac"], cfg["seed"])
    ft_cfg = {**cfg, "_model": model, "train": {**cfg["train"], "epochs": 100, "patience": 15}}
    return fit(ft_cfg, tr, va, bank, None, fs, params=[sl.W], log=log)


def save_checkpoint(path: Path, model, cfg, subject_list, fs):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": cfg, "subject_list": subject_list, "fs": fs,
                "n_channels": model.backbone_in[0], "n_times": model.backbone_in[1]}, path)


def load_checkpoint(path, bank: VideoBank):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    n_subj = ck["state_dict"]["subject_layer.W"].shape[0] if cfg["model"]["subject_layer"] else 0
    model = EEGEncoder(
        ck["n_channels"], ck["n_times"], bank.dim, n_subj, cfg["model"],
        fs=ck["fs"], tmin=cfg["preprocess"]["tmin"],
        visual_dim=(bank.visual_emb.shape[1]
                    if bank.visual_emb is not None else None),
        audio_dim=(bank.audio_emb.shape[1]
                   if bank.audio_emb is not None else None))
    model.load_state_dict(ck["state_dict"])
    return model, ck


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--holdout", default=None, help="subject to leave out (LOSO)")
    ap.add_argument("--finetune", default=None, help="checkpoint to adapt to --subject")
    ap.add_argument("--subject", default=None)
    ap.add_argument("--video-emb", nargs="+", default=None,
                    help="override data.video_emb with one or more archives")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. model.dual_projection=true")
    ap.add_argument("--out", default=None, help="run name under checkpoints/")
    args = ap.parse_args()

    cfg = load_config(args.config)
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
        d = next(d for d in data if d.subject == args.subject and d.split == "trn")
        trials = Trials([d], bank, [])
        model = finetune_subject(cfg, model, trials, bank, fs)
        subject_list = ck["subject_list"] + [args.subject]
        run = args.out or f"{Path(args.finetune).parent.name}_ft_{args.subject}"
    else:
        model, subject_list = train_split(cfg, data, bank, args.holdout)
        run = args.out or (f"loso_{args.holdout}" if args.holdout else "all")

    out = resolve("models/xsubj/checkpoints") / run / "model.pt"
    save_checkpoint(out, model, cfg, subject_list, fs)
    print(f"saved {out}")

    if args.holdout:
        d = next(d for d in data if d.subject == args.holdout and d.video_id is not None
                 and d.split == ("tst" if any(x.split == "tst" for x in data) else "trn"))
        test = Trials([d], bank, subject_list)
        pred = embed(model, test.X, test.subj, get_device())
        print(json.dumps(retrieval_metrics(pred, test.y, bank.emb, cfg["eval"]["k_candidates"]), indent=2))


if __name__ == "__main__":
    main()
