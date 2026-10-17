"""Validation for the ridge baseline and the contrastive encoder.

    python -m src.xsubj.eval                       # official protocol, both models
    python -m src.xsubj.eval --models ridge        # baseline only (seconds)
    python -m src.xsubj.eval --mode loso           # also hide the scored subject's training data
    python -m src.xsubj.eval --models encoder --set model.arch=mlp train.xsubj_avg.prob=0
    python -m src.xsubj.eval --models ridge --video-emb models/xsubj/features/lowlevel.npz         models/xsubj/features/clip_vitb32.npz      # compare / combine video feature sets

official: train on the trn split of all six subjects, score the labeled tst splits
          (sub-01..04) among their 102 candidates; mirrors how sub-05/06 are scored.
loso:     as official, but the scored subject is left out of training entirely.
Test splits are only ever scored, never trained on.
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import yaml

from src.xsubj.eeg.dataset import Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import (
    average_repeats,
    permutation_test,
    retrieval_metrics,
    weighted_csls_similarity,
)
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.train import (
    embed,
    embed_heads,
    embed_tta,
    target_embeddings,
    train_split,
)
from src.xsubj.utils import get_device, load_config, resolve, seed_everything
from src.xsubj.video.embeddings import load_video_bank


def evaluate(
    pred, test: Trials, bank, e: dict, seed: int,
    candidate_embeddings: np.ndarray | None = None,
) -> dict:
    res = {}
    embeddings = candidate_embeddings if candidate_embeddings is not None else bank.emb
    for name, cn in (("raw", False), ("colnorm", True)):
        res[name] = retrieval_metrics(pred, test.y, embeddings, e["k_candidates"], seed, col_norm=cn)
        res[name]["permutation"] = permutation_test(pred, test.y, embeddings, e["n_permutations"], seed, cn)
    csls_k = e.get("csls_k", 10)
    res["csls"] = retrieval_metrics(
        pred, test.y, embeddings, e["k_candidates"], seed, csls_k=csls_k)
    res["csls"]["permutation"] = permutation_test(
        pred, test.y, embeddings, e["n_permutations"], seed, csls_k=csls_k)
    if e["average_repeats"]:
        p, y, _ = average_repeats(pred, test.y, test.subject)
        if len(y) < len(test.y):
            res["repeats_averaged"] = retrieval_metrics(p, y, embeddings, e["k_candidates"], seed)
    return res


def evaluate_late_fusion(
    predictions, test: Trials, bank, cfg: dict
) -> dict:
    if bank.spaces is None or any(
            name not in bank.spaces for name in predictions):
        raise ValueError("Multi-head evaluation requires all head embedding spaces")
    pool = np.unique(test.y)
    embeddings = {
        name: bank.spaces[name][pool] for name in predictions}
    weights = {
        name: float(cfg["model"].get("head_weights", {}).get(name, 1.0))
        for name in predictions}
    scores = weighted_csls_similarity(
        predictions, embeddings, weights, k=10)
    true_columns = np.searchsorted(pool, test.y)
    ranks = (
        scores > scores[np.arange(len(test.y)), true_columns, None]
    ).sum(axis=1)
    n_candidates = len(pool)
    result = {
        "n_trials": int(len(test.y)),
        "n_videos": int(n_candidates),
        "top1@all": float((ranks == 0).mean()),
        "top5@all": float((ranks < 5).mean()),
        "chance1@all": 1.0 / n_candidates,
        "mean_rank@all": float(ranks.mean() + 1),
    }
    best = scores.argmax(axis=1)
    rng = np.random.default_rng(cfg["seed"])
    null = np.array([
        (best == rng.permutation(true_columns)).mean()
        for _ in range(cfg["eval"]["n_permutations"])
    ])
    result["permutation"] = {
        "top1@all": result["top1@all"],
        "null_mean": float(null.mean()),
        "p_value": float(
            (1 + (null >= result["top1@all"]).sum()) / (1 + len(null))),
    }
    return {"csls": result}


def folds(data, mode: str):
    """[(holdout subject or None, test SubjectData list)]"""
    tst = [d for d in data if d.split == "tst" and d.video_id is not None]
    if not tst:  # synthetic / npz data has no test split: score each subject's trials LOSO
        return [(d.subject, [d]) for d in data if d.video_id is not None]
    if mode == "official":
        return [(None, tst)]
    return [(d.subject, [d]) for d in tst]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--mode", default="official", choices=["official", "loso"])
    ap.add_argument("--models", nargs="+", default=["ridge", "encoder"], choices=["ridge", "encoder"])
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs (quick runs)")
    ap.add_argument("--tta-offsets", nargs="+", type=float, default=None,
                    metavar="SECONDS",
                    help="score encoder with EEG test-time crop averaging")
    ap.add_argument("--tta-duration", type=float, default=0.8,
                    help="duration in seconds for each TTA crop (default: 0.8)")
    ap.add_argument("--video-emb", nargs="+", default=None, help="override data.video_emb (several = combined)")
    ap.add_argument("--set", nargs="+", default=[], metavar="KEY=VALUE",
                    help="override config values, e.g. model.arch=mlp train.xsubj_avg.prob=0")
    args = ap.parse_args()

    cfg = load_config(args.config)
    for kv in args.set:
        key, value = kv.split("=", 1)
        *path, last = key.split(".")
        node = cfg
        for k in path:
            node = node[k]
        node[last] = yaml.safe_load(value)
    if cfg["model"].get("gated_fusion", False) and "ridge" in args.models:
        raise SystemExit(
            "Gated fusion changes the target embedding space; run with "
            "--models encoder (ridge cannot share this feature space).")
    if cfg["model"].get("multi_head", False) and "ridge" in args.models:
        raise SystemExit(
            "Multi-head late fusion requires --models encoder; Ridge cannot "
            "share its independent target spaces.")
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    if args.video_emb:
        cfg["data"]["video_emb"] = args.video_emb[0] if len(args.video_emb) == 1 else args.video_emb
    seed_everything(cfg["seed"])
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)
    fs, tmin = data[0].fs, data[0].tmin
    results = {m: {} for m in args.models}
    if len(args.models) == 2:
        results["ensemble"] = {}
    preds = {}

    for holdout, tests in folds(data, args.mode):
        print(f"\n=== train without: {holdout or '-'} | score: {[t.subject for t in tests]} ===")
        train_data = [d for d in data if d.split == "trn" and d.subject != holdout and d.video_id is not None]
        if "ridge" in args.models:
            tr = Trials(train_data, bank, [])
            ridge = RidgeDecoder(cfg["ridge"], fs, tmin).fit(tr.X, bank.emb[tr.y])
            for d in tests:
                test = Trials([d], bank, [])
                preds[("ridge", d.subject)] = ridge.predict(test.X)
                results["ridge"][d.subject] = evaluate(preds[("ridge", d.subject)], test, bank, cfg["eval"], cfg["seed"])
                print(f"ridge   {d.subject}", _short(results["ridge"][d.subject]))
        if "encoder" in args.models:
            seed_everything(cfg["seed"])
            model, subject_list, _ = train_split(cfg, data, bank, holdout)
            candidate_emb = target_embeddings(model, bank, get_device())
            for d in tests:
                test = Trials([d], bank, subject_list)
                pred = None
                if model.multi_head:
                    head_predictions = (
                        embed_tta(
                            model, test.X, test.subj, get_device(), d.fs,
                            args.tta_offsets, args.tta_duration)
                        if args.tta_offsets else
                        embed_heads(model, test.X, test.subj, get_device()))
                    if not isinstance(head_predictions, dict):
                        raise ValueError(
                            "Multi-head evaluation expected per-head embeddings")
                    results["encoder"][d.subject] = evaluate_late_fusion(
                        head_predictions, test, bank, cfg)
                else:
                    pred = (
                        embed_tta(
                            model, test.X, test.subj, get_device(), d.fs,
                            args.tta_offsets, args.tta_duration)
                        if args.tta_offsets else
                        embed(model, test.X, test.subj, get_device()))
                    if isinstance(pred, dict):
                        raise ValueError(
                            "Single-head evaluation received multiple embeddings")
                    results["encoder"][d.subject] = evaluate(
                        pred, test, bank, cfg["eval"], cfg["seed"],
                        candidate_embeddings=candidate_emb)
                print(f"encoder {d.subject}", _short(results["encoder"][d.subject]))
                if "ensemble" in results:  # average of both predicted CLIP vectors
                    ens = preds[("ridge", d.subject)] + pred
                    ens /= np.linalg.norm(ens, axis=1, keepdims=True)
                    results["ensemble"][d.subject] = evaluate(ens, test, bank, cfg["eval"], cfg["seed"])
                    print(f"ensemble {d.subject}", _short(results["ensemble"][d.subject]))

    print(f"\n=== {args.mode}: mean over scored subjects ===")
    for m, res in results.items():
        per_subj = dict(res)
        for name in ("raw", "colnorm", "csls"):
            if name not in next(iter(per_subj.values())):
                continue
            keys = [k for k in next(iter(per_subj.values()))[name] if k.startswith(("top", "chance", "mean_rank"))]
            mean = {k: float(np.mean([r[name][k] for r in per_subj.values()])) for k in keys}
            results[m][f"_mean_{name}"] = mean
            print(f"{m:9s}{name:8s}", "  ".join(f"{k}={v:.3f}" for k, v in mean.items()))

    out = resolve("models/xsubj/results") / f"{args.mode}_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "mode": args.mode, "results": results}, indent=2))
    print(f"\nsaved {out}")


def _short(r: dict) -> str:
    def one(t):
        return (f"top1={t['top1@all']:.3f} top5={t.get('top5@all', float('nan')):.3f} "
                f"(n={t['n_videos']}, p={t['permutation']['p_value']:.3f})")
    return " | ".join(f"{name} {one(metrics)}" for name, metrics in r.items())


if __name__ == "__main__":
    main()
