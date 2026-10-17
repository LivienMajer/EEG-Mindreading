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
    ap.add_argument("--model", choices=["ensemble", "encoder", "ridge"], default="ensemble")
    ap.add_argument("--checkpoint", default="models/xsubj/checkpoints/all/model.pt",
                    help="encoder checkpoint (from src.xsubj.train)")
    ap.add_argument("--subjects", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--col-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="z-score each candidate's similarities over the subject's test recordings "
                         "(paired test on sub-01..04: top-1 21 vs 23 recordings, top-5 35 vs 26, "
                         "mean rank 17.7 -> 16.8; default on)")
    ap.add_argument("--out", default="submissions")
    args = ap.parse_args()

    import torch
    cfg = load_config(args.config)
    if args.model in ("encoder", "ensemble"):
        ck_cfg = torch.load(resolve(args.checkpoint), map_location="cpu", weights_only=False)["cfg"]
        if args.model == "encoder":
            cfg = ck_cfg
        elif ck_cfg["preprocess"] != cfg["preprocess"] or ck_cfg["data"] != cfg["data"]:
            raise SystemExit("encoder checkpoint was trained with other data/preprocessing settings "
                             f"than {args.config}; retrain it with python -m src.xsubj.train")
    seed_everything(cfg["seed"])
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)
    cand_emb = bank.emb[bank.lookup(CANDIDATES.astype(str))]

    if args.model in ("encoder", "ensemble"):
        model, ck = load_checkpoint(resolve(args.checkpoint), bank.dim)
        model.to(get_device())
        lut = {s: i for i, s in enumerate(ck["subject_list"])}
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
        ranked = ranked_candidates(pred, cand_emb, args.col_norm).astype(np.int64)
        np.save(out / f"{subject}_ranked_candidate_list.npy", ranked)
        truth = d.video_id.astype(int) if d.video_id is not None else None
        print(f"{subject}: {check_and_score(ranked, truth)}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
