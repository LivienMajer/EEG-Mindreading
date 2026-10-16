"""Write the challenge submission files: submissions/sub-XX_ranked_candidate_list.npy.

Each file is an int array of shape 102 x 102. Row i is the i-th recording of tst/sub-XX_tst.npy
in its original order, holding the 102 candidate video IDs (1001..1102) from most to least
likely. Subjects with tst_stim labels (sub-01..04) are scored with the organisers' formula,
which checks the format end to end.

    python -m src.xsubj.submit --model encoder --checkpoint models/xsubj/checkpoints/all/model.pt
    python -m src.xsubj.submit --model ridge              # fits ridge on all trn splits first
    python -m src.xsubj.submit ... --subjects sub-01 sub-05 sub-06 --col-norm
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
    ap.add_argument("--model", choices=["encoder", "ridge"], required=True)
    ap.add_argument("--checkpoint", default=None, help="encoder checkpoint (from src.xsubj.train)")
    ap.add_argument("--subjects", nargs="+", default=["sub-01", "sub-05", "sub-06"])
    ap.add_argument("--col-norm", action="store_true", help="z-score each candidate's similarities")
    ap.add_argument("--out", default="submissions")
    args = ap.parse_args()

    if args.model == "encoder":
        import torch
        cfg = torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"]
    else:
        cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)
    cand_emb = bank.emb[bank.lookup(CANDIDATES.astype(str))]

    if args.model == "encoder":
        model, ck = load_checkpoint(args.checkpoint, bank.dim)
        model.to(get_device())
        lut = {s: i for i, s in enumerate(ck["subject_list"])}
    else:
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
        if args.model == "encoder":
            pred = embed(model, d.X, np.full(len(d.X), lut.get(subject, -1)), get_device())
        else:
            pred = ridge.predict(d.X)
        ranked = ranked_candidates(pred, cand_emb, args.col_norm).astype(np.int64)
        np.save(out / f"{subject}_ranked_candidate_list.npy", ranked)
        truth = d.video_id.astype(int) if d.video_id is not None else None
        print(f"{subject}: {check_and_score(ranked, truth)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
