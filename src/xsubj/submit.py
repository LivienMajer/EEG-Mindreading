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
from src.xsubj.metrics import (
    csls_similarity,
    similarity,
    weighted_csls_similarity,
)
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.train import (
    embed,
    embed_heads,
    embed_tta,
    encode_features,
    load_checkpoint,
    target_embeddings,
)
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


def check_and_score(ranked: np.ndarray, truth: np.ndarray | None) -> str:
    assert ranked.shape == (102, 102) and ranked.dtype.kind == "i", ranked.shape
    assert (np.sort(ranked, axis=1) == CANDIDATES).all(), "every row must be a permutation of 1001..1102"
    if truth is None:
        return "format ok (no labels)"
    top1 = (ranked[:, 0] == truth).mean()
    top5 = (ranked[:, :5] == truth[:, None]).any(axis=1).mean()
    return f"format ok  top-1 {top1:.4f}  top-5 {top5:.4f}  (chance {1 / 102:.4f} / {5 / 102:.4f})"


def _normalize_score_rows(scores: np.ndarray) -> np.ndarray:
    scores = np.asarray(scores, dtype=np.float32)
    if scores.ndim != 2 or not np.isfinite(scores).all():
        raise ValueError("Score fusion expects a finite 2D similarity matrix")
    centered = scores - scores.mean(axis=1, keepdims=True)
    scale = scores.std(axis=1, keepdims=True)
    return centered / np.maximum(scale, 1e-8)


def _checkpoint_scores(
    model,
    checkpoint: dict,
    bank,
    X: np.ndarray,
    subject: str,
    args,
    device,
    head_weights_override: dict[str, float] | None = None,
) -> np.ndarray:
    subject_index = {
        name: index for index, name in enumerate(checkpoint["subject_list"])
    }.get(subject, -1)
    subjects = np.full(len(X), subject_index, dtype=np.int64)
    predictions = (
        embed_tta(
            model, X, subjects, device, checkpoint["fs"],
            args.tta_offsets, args.tta_duration)
        if args.tta_offsets else
        embed_heads(model, X, subjects, device)
        if model.multi_head else
        embed(model, X, subjects, device)
    )
    if model.multi_head:
        if head_weights_override is None:
            configured_weights = checkpoint["cfg"]["model"].get("head_weights", {})
            weights = {
                name: float(configured_weights.get(name, 1.0))
                for name in model.head_names
            }
        else:
            weights = head_weights_override
        candidate_banks = {
            name: bank.spaces[name][bank.lookup(CANDIDATES.astype(str))]
            for name in model.head_names
        } if bank.spaces is not None else {}
        if set(candidate_banks) != set(model.head_names):
            raise ValueError(
                "A fused multi-head checkpoint is missing a candidate "
                "embedding space")
        return weighted_csls_similarity(
            predictions, candidate_banks, weights, k=10)
    if isinstance(predictions, dict):
        raise ValueError("A single-head checkpoint returned multiple embeddings")
    candidate_embeddings = bank.emb[bank.lookup(CANDIDATES.astype(str))]
    return (
        csls_similarity(predictions, candidate_embeddings, k=10)
        if args.csls else
        similarity(predictions, candidate_embeddings, args.col_norm)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--model", choices=["ensemble", "encoder", "ridge"], default="ensemble")
    ap.add_argument("--checkpoint", default="models/xsubj/checkpoints/all/model.pt",
                    help="encoder checkpoint (from src.xsubj.train)")
    ap.add_argument(
        "--fusion-checkpoints", nargs="+", default=None,
        help="additional encoder checkpoints for normalized score-level late fusion")
    ap.add_argument(
        "--ridge-feature-checkpoint", default=None,
        help="fit Ridge on frozen shared EEG features from this checkpoint")
    ap.add_argument(
        "--score-fusion-weights", nargs="+", type=float, default=None,
        metavar="WEIGHT",
        help="non-negative weights for primary encoder, each --fusion-checkpoint, "
             "then --ridge-feature-checkpoint, in that order")
    ap.add_argument("--subjects", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--col-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="z-score each candidate's similarities over the subject's test recordings "
                         "(paired test on sub-01..04: top-1 21 vs 23 recordings, top-5 35 vs 26, "
                         "mean rank 17.7 -> 16.8; default on)")
    ap.add_argument("--csls", action="store_true",
                    help="rank using Cross-Domain Similarity Local Scaling (k=10) instead "
                         "of cosine/column normalization; multi-head checkpoints always "
                         "use independent per-head CSLS")
    ap.add_argument("--head-weights", nargs="+", default=None, metavar="HEAD=WEIGHT",
                    help="multi-head late-fusion weights, e.g. videomae=1 clip=1")
    ap.add_argument("--tta-offsets", nargs="+", type=float, default=None,
                    metavar="SECONDS",
                    help="enable EEG test-time crop averaging at these start offsets")
    ap.add_argument("--tta-duration", type=float, default=0.8,
                    help="duration in seconds for each TTA crop (default: 0.8)")
    ap.add_argument("--out", default="submissions")
    args = ap.parse_args()

    import torch
    if (args.fusion_checkpoints or args.ridge_feature_checkpoint) and args.model != "encoder":
        raise SystemExit(
            "Additional score members require --model encoder; Borda/rank "
            "fusion is not used.")
    if args.score_fusion_weights is not None and not (
            args.fusion_checkpoints or args.ridge_feature_checkpoint):
        raise SystemExit(
            "--score-fusion-weights requires an additional score-fusion member.")
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
        if args.model == "ensemble" and model.multi_head:
            raise SystemExit(
                "Multi-head late fusion uses separate target spaces; submit "
                "with --model encoder instead of the Ridge+encoder ensemble.")
        lut = {s: i for i, s in enumerate(ck["subject_list"])}
        candidate_bank = target_embeddings(model, bank, get_device())
        if model.multi_head:
            candidate_head_banks = {
                name: bank.spaces[name][bank.lookup(CANDIDATES.astype(str))]
                for name in model.head_names
            } if bank.spaces is not None else {}
            if set(candidate_head_banks) != set(model.head_names):
                raise ValueError(
                    "Multi-head checkpoint requires each configured feature "
                    "space to be loaded: "
                    f"{', '.join(model.head_names)}. Available spaces: "
                    f"{', '.join(sorted(candidate_head_banks)) or 'none'}")
            configured_weights = ck["cfg"]["model"].get("head_weights", {})
            head_weights = {
                name: float(configured_weights.get(name, 1.0))
                for name in model.head_names
            }
            if args.head_weights:
                head_weights = {}
                for item in args.head_weights:
                    name, separator, value = item.partition("=")
                    if not separator or not name or name in head_weights:
                        raise SystemExit(
                            "--head-weights entries must be unique HEAD=WEIGHT pairs")
                    try:
                        head_weights[name] = float(value)
                    except ValueError as exc:
                        raise SystemExit(
                            f"Invalid fusion weight in {item!r}") from exc
            if set(head_weights) != set(model.head_names):
                raise SystemExit(
                    "Provide one fusion weight for each of "
                    f"{', '.join(model.head_names)}")
            if (any(not np.isfinite(weight) or weight < 0
                    for weight in head_weights.values())
                    or sum(head_weights.values()) <= 0):
                raise SystemExit(
                    "Fusion weights must be finite, non-negative, and not all zero")
    else:
        candidate_bank = bank.emb
    cand_emb = (
        None if args.model == "encoder" and model.multi_head
        else candidate_bank[bank.lookup(CANDIDATES.astype(str))])
    if args.model in ("ridge", "ensemble"):
        train = Trials([d for d in data if d.split == "trn"], bank, [])
        ridge = RidgeDecoder(cfg["ridge"], data[0].fs, data[0].tmin).fit(train.X, bank.emb[train.y])
        path = resolve("models/xsubj/checkpoints/ridge/ridge.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": ridge, "cfg": cfg}, path)
        print(f"saved {path}")

    fusion_members = []
    if args.fusion_checkpoints:
        for member_path in args.fusion_checkpoints:
            member_path = resolve(member_path)
            member_ck = torch.load(
                member_path, map_location="cpu", weights_only=False)
            member_cfg = member_ck["cfg"]
            if member_cfg["preprocess"] != cfg["preprocess"]:
                raise SystemExit(
                    f"Fusion checkpoint {member_path} uses a different EEG "
                    "preprocessing window/settings; retrain or preprocess "
                    "with matching settings before score fusion.")
            if (member_cfg["data"].get("format")
                    != cfg["data"].get("format")
                    or member_cfg["data"].get("challenge_dir")
                    != cfg["data"].get("challenge_dir")):
                raise SystemExit(
                    f"Fusion checkpoint {member_path} was trained against a "
                    "different EEG data source.")
            member_bank = load_video_bank(member_cfg)
            member_model, member_ck = load_checkpoint(member_path, member_bank)
            member_model.to(get_device()).eval()
            if member_model.backbone_in != (data[0].X.shape[1], data[0].X.shape[2]):
                raise SystemExit(
                    f"Fusion checkpoint {member_path} expects EEG shape "
                    f"{member_model.backbone_in}, got "
                    f"{(data[0].X.shape[1], data[0].X.shape[2])}.")
            fusion_members.append((member_model, member_ck, member_bank))

    ridge_feature = None
    if args.ridge_feature_checkpoint:
        feature_path = resolve(args.ridge_feature_checkpoint)
        feature_ck = torch.load(
            feature_path, map_location="cpu", weights_only=False)
        feature_cfg = feature_ck["cfg"]
        if feature_cfg["preprocess"] != cfg["preprocess"]:
            raise SystemExit(
                "Frozen-feature Ridge checkpoint uses different EEG "
                "preprocessing settings.")
        if (feature_cfg["data"].get("format") != cfg["data"].get("format")
                or feature_cfg["data"].get("challenge_dir")
                != cfg["data"].get("challenge_dir")):
            raise SystemExit(
                "Frozen-feature Ridge checkpoint was trained against a "
                "different EEG data source.")
        feature_bank = load_video_bank(feature_cfg)
        feature_model, feature_ck = load_checkpoint(feature_path, feature_bank)
        if feature_model.multi_head or feature_model.dual_projection or feature_model.gated_fusion:
            raise SystemExit(
                "Frozen-feature Ridge currently requires a single target-space "
                "encoder checkpoint.")
        if feature_model.backbone_in != (
                data[0].X.shape[1], data[0].X.shape[2]):
            raise SystemExit(
                "Frozen-feature Ridge checkpoint expects EEG shape "
                f"{feature_model.backbone_in}, got "
                f"{(data[0].X.shape[1], data[0].X.shape[2])}.")
        feature_model.to(get_device()).eval()
        feature_subjects = list(feature_ck["subject_list"])
        feature_train = Trials(
            [d for d in data if d.split == "trn" and d.video_id is not None],
            feature_bank,
            feature_subjects,
        )
        feature_train_X = encode_features(
            feature_model, feature_train.X, feature_train.subj, get_device())
        feature_ridge = RidgeDecoder(
            cfg["ridge"], data[0].fs, data[0].tmin).fit_features(
                feature_train_X, feature_bank.emb[feature_train.y])
        feature_ridge_path = resolve(
            "models/xsubj/checkpoints/ridge/frozen_feature_ridge.joblib")
        feature_ridge_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"model": feature_ridge, "cfg": cfg,
             "feature_checkpoint": str(feature_path)},
            feature_ridge_path)
        print(f"saved {feature_ridge_path}")
        ridge_feature = (
            feature_model, feature_ck, feature_bank, feature_ridge)

    fusion_requested = bool(fusion_members or ridge_feature)
    fusion_weights = (
        args.score_fusion_weights
        if args.score_fusion_weights is not None else
        [1.0] * (1 + len(fusion_members) + int(ridge_feature is not None)))
    if fusion_requested and len(fusion_weights) != (
            1 + len(fusion_members) + int(ridge_feature is not None)):
        raise SystemExit(
            "--score-fusion-weights must supply one weight for the primary "
            "encoder and every additional member.")
    if fusion_requested and (
            any(not np.isfinite(weight) or weight < 0
                for weight in fusion_weights)
            or sum(fusion_weights) <= 0):
        raise SystemExit(
            "Score-fusion weights must be finite, non-negative, and not all zero.")

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        d = next(d for d in data if d.subject == subject and d.split == "tst")  # original recording order
        preds = []
        if fusion_requested:
            primary_scores = _checkpoint_scores(
                model, ck, bank, d.X, subject, args, get_device(),
                head_weights if model.multi_head else None)
            score_members = [primary_scores]
            for member_model, member_ck, member_bank in fusion_members:
                score_members.append(_checkpoint_scores(
                    member_model, member_ck, member_bank,
                    d.X, subject, args, get_device()))
            if ridge_feature is not None:
                feature_model, feature_ck, feature_bank, feature_ridge = ridge_feature
                feature_subject_index = {
                    name: index
                    for index, name in enumerate(feature_ck["subject_list"])
                }.get(subject, -1)
                feature_subjects = np.full(
                    len(d.X), feature_subject_index, dtype=np.int64)
                test_features = encode_features(
                    feature_model, d.X, feature_subjects, get_device())
                prediction = feature_ridge.predict_features(test_features)
                candidate_embeddings = feature_bank.emb[
                    feature_bank.lookup(CANDIDATES.astype(str))]
                ridge_scores = (
                    csls_similarity(prediction, candidate_embeddings, k=10)
                    if args.csls else
                    similarity(prediction, candidate_embeddings, args.col_norm)
                )
                score_members.append(ridge_scores)
            fused_scores = sum(
                weight * _normalize_score_rows(scores)
                for weight, scores in zip(fusion_weights, score_members)
            ) / sum(fusion_weights)
            ranked = CANDIDATES[
                np.argsort(-fused_scores, axis=1, kind="stable")
            ].astype(np.int64)
            print(
                "  normalized similarity-score fusion: "
                f"{len(score_members)} members, weights={fusion_weights}")
            np.save(out / f"{subject}_ranked_candidate_list.npy", ranked)
            truth = d.video_id.astype(int) if d.video_id is not None else None
            print(f"{subject}: {check_and_score(ranked, truth)}")
            continue
        if (args.model in ("encoder", "ensemble")
                and not model.multi_head):
            subj = np.full(len(d.X), lut.get(subject, -1))
            pred = (
                embed_tta(
                    model, d.X, subj, get_device(), d.fs,
                    args.tta_offsets, args.tta_duration)
                if args.tta_offsets else embed(model, d.X, subj, get_device()))
            if isinstance(pred, dict):
                raise ValueError("Single-head submission received multiple embeddings")
            preds.append(pred)
        if args.model in ("ridge", "ensemble"):
            preds.append(ridge.predict(d.X))
        if args.model == "encoder" and model.multi_head:
            subj = np.full(len(d.X), lut.get(subject, -1))
            head_predictions = (
                embed_tta(
                    model, d.X, subj, get_device(), d.fs,
                    args.tta_offsets, args.tta_duration)
                if args.tta_offsets else embed_heads(
                    model, d.X, subj, get_device()))
            if not isinstance(head_predictions, dict):
                raise ValueError("Multi-head submission expected per-head embeddings")
            scores = weighted_csls_similarity(
                head_predictions, candidate_head_banks, head_weights, k=10)
            ranked = CANDIDATES[
                np.argsort(-scores, axis=1, kind="stable")].astype(np.int64)
            print(
                "  weighted independent CSLS heads: "
                + ", ".join(f"{name}={weight:g}"
                            for name, weight in head_weights.items()))
        else:
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
