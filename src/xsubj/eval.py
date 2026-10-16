"""Leave-one-subject-out benchmark for the ridge baseline and the contrastive encoder.

    python -m src.xsubj.eval --config configs/xsubj.yaml                    # both models, all folds
    python -m src.xsubj.eval --config configs/xsubj.yaml --models ridge     # baseline only
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np

from src.xsubj.eeg.dataset import Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import average_repeats, permutation_test, retrieval_metrics
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.train import embed, train_split
from src.xsubj.utils import get_device, load_config, resolve, seed_everything
from src.xsubj.video.embeddings import load_video_bank


def evaluate(pred, test: Trials, bank, e: dict, seed: int) -> dict:
    res = {"trials": retrieval_metrics(pred, test.y, bank.emb, e["k_candidates"], seed),
           "permutation": permutation_test(pred, test.y, bank.emb, e["n_permutations"], seed)}
    if e["average_repeats"]:
        p, y, _ = average_repeats(pred, test.y, test.subject)
        if len(y) < len(test.y):
            res["repeats_averaged"] = retrieval_metrics(p, y, bank.emb, e["k_candidates"], seed)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--models", nargs="+", default=["ridge", "encoder"], choices=["ridge", "encoder"])
    ap.add_argument("--epochs", type=int, default=None, help="override train.epochs (quick runs)")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.epochs:
        cfg["train"]["epochs"] = args.epochs
    seed_everything(cfg["seed"])
    data = [d for d in load_preprocessed(cfg) if d.video_id is not None]
    bank = load_video_bank(cfg)
    fs, tmin = data[0].fs, data[0].tmin
    results = {m: {} for m in args.models}

    for held in data:
        print(f"\n=== holdout {held.subject} ===")
        others = [d for d in data if d is not held]
        if "ridge" in args.models:
            tr = Trials(others, bank, [])
            ridge = RidgeDecoder(cfg["ridge"], fs, tmin).fit(tr.X, bank.emb[tr.y])
            test = Trials([held], bank, [])
            results["ridge"][held.subject] = evaluate(ridge.predict(test.X), test, bank, cfg["eval"], cfg["seed"])
            print("ridge  ", _short(results["ridge"][held.subject]))
        if "encoder" in args.models:
            seed_everything(cfg["seed"])
            model, subject_list = train_split(cfg, data, bank, held.subject)
            test = Trials([held], bank, subject_list)
            pred = embed(model, test.X, test.subj, get_device())
            results["encoder"][held.subject] = evaluate(pred, test, bank, cfg["eval"], cfg["seed"])
            print("encoder", _short(results["encoder"][held.subject]))

    print("\n=== LOSO mean over subjects ===")
    for m, folds in results.items():
        keys = [k for k in next(iter(folds.values()))["trials"] if k.startswith(("top", "chance"))]
        mean = {k: float(np.mean([f["trials"][k] for f in folds.values()])) for k in keys}
        results[m]["_mean"] = mean
        print(f"{m:8s}", "  ".join(f"{k}={v:.3f}" for k, v in mean.items()))

    out = resolve("models/xsubj/results") / f"loso_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"config": cfg, "results": results}, indent=2))
    print(f"\nsaved {out}")


def _short(r: dict) -> str:
    t, p = r["trials"], r["permutation"]
    s = "  ".join(f"{k}={v:.3f}" for k, v in t.items() if k.startswith("top1"))
    s += f"  (perm p={p['p_value']:.4f})"
    if "repeats_averaged" in r:
        s += f"  | avg-repeats top1@all={r['repeats_averaged']['top1@all']:.3f}"
    return s


if __name__ == "__main__":
    main()
