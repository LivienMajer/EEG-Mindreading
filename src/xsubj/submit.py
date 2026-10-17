"""Write the challenge submission files: submissions/sub-XX_ranked_candidate_list.npy.

Each file is an int array of shape 102 x 102. Row i is the i-th recording of tst/sub-XX_tst.npy
in its original order, holding the 102 candidate video IDs (1001..1102) from most to least
likely. Subjects with tst_stim labels (sub-01..04) are scored with the organisers' formula,
which checks the format end to end.

Default model: ensemble = mean of the ridge and the encoder predictions in CLIP space, ranked
after per-candidate normalisation (dev: top-1 10.5%, top-5 35.3%, mean rank 16.8; without
normalisation 11.0% / 33.1% / 17.7; ridge alone 9.8% / 33.1% / 18.9).

    python -m src.xsubj.train                              # encoder -> models/xsubj/checkpoints/all/model.pt
    python -m src.xsubj.submit                             # ensemble, sub-01..06
    python -m src.xsubj.submit --model ridge               # ridge only (fits on all trn splits)
    python -m src.xsubj.submit ... --subjects sub-01 sub-05 sub-06 --no-col-norm
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from src.xsubj.eeg.dataset import Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import csls_similarity, similarity
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.train import embed, load_checkpoint, target_embeddings
from src.xsubj.utils import get_device, load_config, resolve, seed_everything
from src.xsubj.video.embeddings import load_video_bank

CANDIDATES = np.arange(1001, 1103)


def ranked_candidates(
    pred: np.ndarray,
    cand_emb: np.ndarray,
    col_norm: bool,
    csls_k: int | None = None,
) -> np.ndarray:
    sim = (csls_similarity(pred, cand_emb, csls_k)
           if csls_k is not None else similarity(pred, cand_emb, col_norm))
    return CANDIDATES[np.argsort(-sim, axis=1, kind="stable")]


def fuse_rankings(rankings: list[np.ndarray]) -> np.ndarray:
    """Fuse candidate rankings by equal-weight Borda rank (lower mean rank wins)."""
    if len(rankings) < 2:
        raise ValueError("Rank fusion requires at least two model rankings")
    stacked = np.stack(rankings)
    if stacked.ndim != 3 or stacked.shape[2] != len(CANDIDATES):
        raise ValueError(
            "Each model ranking must have shape (trials, 102 candidates)")
    if not np.isin(stacked, CANDIDATES).all():
        raise ValueError("Model rankings contain candidate IDs outside 1001..1102")
    if not np.all(np.sort(stacked, axis=2) == CANDIDATES):
        raise ValueError("Each model ranking row must contain every candidate once")

    candidate_order = np.searchsorted(CANDIDATES, stacked)
    ranks = np.empty_like(candidate_order)
    positions = np.broadcast_to(
        np.arange(len(CANDIDATES)), candidate_order.shape)
    np.put_along_axis(ranks, candidate_order, positions, axis=2)
    mean_rank = ranks.mean(axis=0)
    return CANDIDATES[np.argsort(mean_rank, axis=1, kind="stable")]


def check_and_score(ranked: np.ndarray, truth: np.ndarray | None) -> str:
    assert ranked.shape == (102, 102) and ranked.dtype.kind == "i", ranked.shape
    assert (np.sort(ranked, axis=1) == CANDIDATES).all(), "every row must be a permutation of 1001..1102"
    if truth is None:
        return "format ok (no labels)"
    top1 = (ranked[:, 0] == truth).mean()
    top5 = (ranked[:, :5] == truth[:, None]).any(axis=1).mean()
    return f"format ok  top-1 {top1:.4f}  top-5 {top5:.4f}  (chance {1 / 102:.4f} / {5 / 102:.4f})"


def submit_rank_ensemble(args) -> None:
    import torch

    if not args.checkpoints or len(args.checkpoints) < 2:
        raise SystemExit("--model rank-ensemble requires at least two --checkpoints")
    checkpoint_paths = [resolve(path) for path in args.checkpoints]
    if len(set(checkpoint_paths)) != len(checkpoint_paths):
        raise SystemExit("--checkpoints must not contain duplicates")

    checkpoint_configs = [
        torch.load(path, map_location="cpu", weights_only=False)["cfg"]
        for path in checkpoint_paths
    ]
    base_cfg = checkpoint_configs[0]
    for path, cfg in zip(checkpoint_paths[1:], checkpoint_configs[1:]):
        if (cfg["preprocess"] != base_cfg["preprocess"]
                or cfg["data"]["format"] != base_cfg["data"]["format"]
                or cfg["data"]["challenge_dir"] != base_cfg["data"]["challenge_dir"]):
            raise SystemExit(
                f"Checkpoint {path} uses different EEG preprocessing or challenge data; "
                "rank fusion requires matching EEG inputs.")

    seed_everything(base_cfg["seed"])
    data = load_preprocessed(base_cfg)
    device = get_device()
    models = []
    candidate_banks = []
    subject_slots = []
    for path, cfg in zip(checkpoint_paths, checkpoint_configs):
        bank = load_video_bank(cfg)
        bank_ids = bank.ids[bank.lookup(CANDIDATES.astype(str))]
        if not np.array_equal(bank_ids, CANDIDATES.astype(str)):
            raise SystemExit(
                f"Checkpoint {path} has a different candidate-video order")
        model, ck = load_checkpoint(path, bank)
        model.to(device)
        models.append(model)
        model_bank = target_embeddings(model, bank, device)
        candidate_banks.append(
            model_bank[bank.lookup(CANDIDATES.astype(str))])
        subject_slots.append(ck["subject_list"])

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        test = next(
            (d for d in data if d.subject == subject and d.split == "tst"),
            None)
        if test is None:
            raise ValueError(f"No test EEG data found for {subject}")
        member_rankings = []
        for path, model, bank_emb, slots in zip(
                checkpoint_paths, models, candidate_banks, subject_slots):
            slot = slots.index(subject) if subject in slots else -1
            pred = embed(
                model, test.X, np.full(len(test.X), slot), device)
            ranking = ranked_candidates(
                pred, bank_emb, args.col_norm,
                csls_k=10 if args.csls else None).astype(np.int64)
            member_rankings.append(ranking)
            truth = test.video_id.astype(int) if test.video_id is not None else None
            print(f"{subject} [{path.parent.name}]: {check_and_score(ranking, truth)}")

        fused = fuse_rankings(member_rankings).astype(np.int64)
        np.save(out / f"{subject}_ranked_candidate_list.npy", fused)
        truth = test.video_id.astype(int) if test.video_id is not None else None
        print(f"{subject} [Borda rank fusion]: {check_and_score(fused, truth)}")
    print(f"written to {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--model", choices=["ensemble", "encoder", "ridge", "rank-ensemble"], default="ensemble")
    ap.add_argument("--checkpoint", default="models/xsubj/checkpoints/all/model.pt",
                    help="encoder checkpoint (from src.xsubj.train)")
    ap.add_argument("--checkpoints", nargs="+", default=None,
                    help="two or more encoder checkpoints for Borda rank fusion")
    ap.add_argument("--subjects", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--col-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="z-score each candidate's similarities over the subject's test recordings "
                         "(paired test on sub-01..04: top-1 21 vs 23 recordings, top-5 35 vs 26, "
                         "mean rank 17.7 -> 16.8; default on)")
    ap.add_argument("--csls", action="store_true",
                    help="rank using Cross-Domain Similarity Local Scaling (k=10) instead "
                         "of cosine/column normalization")
    ap.add_argument("--out", default="submissions")
    args = ap.parse_args()

    if args.model == "rank-ensemble":
        submit_rank_ensemble(args)
        return

    import torch
    cfg = load_config(args.config)
    if args.model in ("encoder", "ensemble"):
        ck_cfg = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)["cfg"]
        if args.model == "encoder":
            cfg = ck_cfg
        elif ck_cfg["preprocess"] != cfg["preprocess"] or ck_cfg["data"] != cfg["data"]:
            raise SystemExit("encoder checkpoint was trained with other data/preprocessing settings "
                             f"than {args.config}; retrain it with python -m src.xsubj.train")
        if (args.model == "ensemble"
                and ck_cfg["model"].get("gated_fusion", False)):
            raise SystemExit(
                "The ridge+encoder ensemble cannot combine the gated target "
                "space with ridge's raw embedding space; submit with "
                "--model encoder.")
    seed_everything(cfg["seed"])
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)

    if args.model in ("encoder", "ensemble"):
        model, ck = load_checkpoint(resolve(args.checkpoint), bank)
        model.to(get_device())
        lut = {s: i for i, s in enumerate(ck["subject_list"])}
        candidate_bank = target_embeddings(model, bank, get_device())
    else:
        candidate_bank = bank.emb
    cand_emb = candidate_bank[bank.lookup(CANDIDATES.astype(str))]
    if args.model in ("ridge", "ensemble"):
        train = Trials([d for d in data if d.split == "trn"], bank, [])
        ridge = RidgeDecoder(cfg["ridge"], data[0].fs, data[0].tmin).fit(train.X, bank.emb[train.y])
        path = resolve("models/xsubj/checkpoints/ridge/ridge.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": ridge, "cfg": cfg}, path)
        print(f"saved {path}")

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        d = next(d for d in data if d.subject == subject and d.split == "tst")  # original recording order
        preds = []
        if args.model in ("encoder", "ensemble"):
            preds.append(embed(model, d.X, np.full(len(d.X), lut.get(subject, -1)), get_device()))
        if args.model in ("ridge", "ensemble"):
            preds.append(ridge.predict(d.X))
        pred = np.sum(preds, 0)
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        ranked = ranked_candidates(
            pred, cand_emb, args.col_norm,
            csls_k=10 if args.csls else None).astype(np.int64)
        np.save(out / f"{subject}_ranked_candidate_list.npy", ranked)
        truth = d.video_id.astype(int) if d.video_id is not None else None
        print(f"{subject}: {check_and_score(ranked, truth)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
