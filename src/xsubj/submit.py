"""Write the challenge submission files: submissions/sub-XX_ranked_candidate_list.npy.

Each file is an int array of shape 102 x 102. Row i is the i-th recording of tst/sub-XX_tst.npy
in its original order, holding the 102 candidate video IDs (1001..1102) from most to least
likely. Subjects with tst_stim labels (sub-01..04) are scored with the organisers' formula,
which checks the format end to end.

Default model: ensemble of three equal votes in CLIP space, ranked after per-candidate
normalisation:
  1. the mean of the encoders listed under `submit.encoders` (5x ShallowConvNet, 3x MLP, 1x EEGNet)
  2. pooled ridge, trained on the trn splits of all six subjects
  3. per-subject ridge, trained on the subject's own trn split only
Dev (sub-01..04): top-1 17.6%, top-5 47.8%, mean rank 11.5 with low-pass-only preprocessing and no
average reference (13.2% / 43.1% / 14.2 with the earlier 0.5-40 Hz band-pass + average reference).

    python -m src.xsubj.submit                             # ensemble, sub-01..06
    python -m src.xsubj.submit --model ridge               # pooled ridge only (fits on all trn splits)
    python -m src.xsubj.submit --model subject-ridge       # per-subject ridge only
    python -m src.xsubj.submit --encoders all mlp_s0       # other encoder set (checkpoint folder names)
    python -m src.xsubj.submit ... --subjects sub-01 sub-05 sub-06 --no-col-norm
"""
from __future__ import annotations

import argparse

import joblib
import numpy as np

from src.xsubj.eeg.dataset import Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import similarity
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.train import embed, load_checkpoint
from src.xsubj.utils import get_device, load_config, resolve, seed_everything
from src.xsubj.video.embeddings import load_video_bank

CANDIDATES = np.arange(1001, 1103)


def ranked_candidates(pred: np.ndarray, cand_emb: np.ndarray, col_norm: bool) -> np.ndarray:
    sim = similarity(pred, cand_emb, col_norm)
    return CANDIDATES[np.argsort(-sim, axis=1, kind="stable")]


def check_and_score(ranked: np.ndarray, truth: np.ndarray | None) -> str:
    assert ranked.shape == (102, 102) and ranked.dtype.kind == "i", ranked.shape
    assert (np.sort(ranked, axis=1) == CANDIDATES).all(), "every row must be a permutation of 1001..1102"
    if truth is None:
        return "format ok (no labels)"
    top1 = (ranked[:, 0] == truth).mean()
    top5 = (ranked[:, :5] == truth[:, None]).any(axis=1).mean()
    return f"format ok  top-1 {top1:.4f}  top-5 {top5:.4f}  (chance {1 / 102:.4f} / {5 / 102:.4f})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--model", choices=["ensemble", "encoder", "ridge", "subject-ridge"], default="ensemble")
    ap.add_argument("--encoders", nargs="+", default=None,
                    help="checkpoint folders under models/xsubj/checkpoints (default: submit.encoders)")
    ap.add_argument("--subjects", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--col-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="z-score each candidate's similarities over the subject's test recordings "
                         "(paired test on sub-01..04: top-1 21 vs 23 recordings, top-5 35 vs 26, "
                         "mean rank 17.7 -> 16.8; default on)")
    ap.add_argument("--out", default="submissions")
    ap.add_argument("--save-scores", action="store_true",
                    help="also save the similarity table next to each ranking (src/score_dump.py)")
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config)
    names = args.encoders or cfg["submit"]["encoders"]
    paths = [resolve("models/xsubj/checkpoints") / n / "model.pt" for n in names]
    if args.model in ("encoder", "ensemble"):
        for path in paths:
            ck_cfg = torch.load(path, map_location="cpu", weights_only=False)["cfg"]
            same_data = {k: v for k, v in ck_cfg["data"].items() if k != "cache_dir"} ==                         {k: v for k, v in cfg["data"].items() if k != "cache_dir"}
            if ck_cfg["preprocess"] != cfg["preprocess"] or not same_data:
                raise SystemExit(f"{path} was trained with other data/preprocessing settings than "
                                 f"{args.config}; retrain it (see the submit section of the config)")
    seed_everything(cfg["seed"])
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)
    cand_emb = bank.emb[bank.lookup(CANDIDATES.astype(str))]

    encoders = []
    if args.model in ("encoder", "ensemble"):
        for path in paths:
            model, ck = load_checkpoint(path, bank.dim)
            encoders.append((model.to(get_device()), {s: i for i, s in enumerate(ck["subject_list"])}))
        print(f"{len(encoders)} encoders: {' '.join(names)}")
    fs, tmin = data[0].fs, data[0].tmin
    if args.model in ("ridge", "ensemble"):
        train = Trials([d for d in data if d.split == "trn"], bank, [])
        ridge = RidgeDecoder(cfg["ridge"], fs, tmin).fit(train.X, bank.emb[train.y])
        path = resolve("models/xsubj/checkpoints/ridge/ridge.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": ridge, "cfg": cfg}, path)
        print(f"saved {path}")
    if args.model in ("subject-ridge", "ensemble"):
        subject_ridge = {}
        for d in data:
            if d.split == "trn":
                tr = Trials([d], bank, [])
                subject_ridge[d.subject] = RidgeDecoder(cfg["ridge"], fs, tmin).fit(tr.X, bank.emb[tr.y])
        path = resolve("models/xsubj/checkpoints/ridge_subject/ridge_subject.joblib")
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"models": subject_ridge, "cfg": cfg}, path)
        print(f"saved {path} ({len(subject_ridge)} subjects)")

    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        d = next(d for d in data if d.subject == subject and d.split == "tst")  # original recording order
        preds = []
        if encoders:  # mean of the encoders' unit vectors, renormalised -> one vote next to ridge
            enc = np.mean([embed(m, d.X, np.full(len(d.X), lut.get(subject, -1)), get_device())
                           for m, lut in encoders], 0)
            preds.append(enc / np.linalg.norm(enc, axis=1, keepdims=True))
        if args.model in ("ridge", "ensemble"):
            preds.append(ridge.predict(d.X))
        if args.model in ("subject-ridge", "ensemble"):
            preds.append(subject_ridge[subject].predict(d.X))
        pred = np.sum(preds, 0)
        pred /= np.linalg.norm(pred, axis=1, keepdims=True)
        ranked = ranked_candidates(pred, cand_emb, args.col_norm).astype(np.int64)
        np.save(out / f"{subject}_ranked_candidate_list.npy", ranked)
        if args.save_scores:                  # for score fusion (src/score_dump.py); not for submissions/
            from src.score_dump import save_scores
            save_scores(out / f"{subject}_ranked_candidate_list.npy", similarity(pred, cand_emb, args.col_norm))
        truth = d.video_id.astype(int) if d.video_id is not None else None
        print(f"{subject}: {check_and_score(ranked, truth)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
