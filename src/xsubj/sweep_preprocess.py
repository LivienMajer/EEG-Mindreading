"""Preprocessing sweep, scored with the ridge part of the submission (pooled + per-subject ridge,
equal weight, per-candidate normalisation), which trains in seconds.

Two criteria per variant:
  cv   5-fold cross-validation over *training videos* (no test data): normalised mean rank of
       held-out training trials among the held-out videos (0 = best, 0.5 = chance). Main criterion.
  dev  test splits of sub-01..04 among 102 candidates: top-1, top-5, mean rank. Confirmation only.

    python -m src.xsubj.sweep_preprocess                  # all variants
    python -m src.xsubj.sweep_preprocess --only filter    # variants whose name contains "filter"
    python -m src.xsubj.sweep_preprocess --combine "no filter" "demean trials"   # several changes at once
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import similarity
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.utils import load_config, resolve
from src.xsubj.video.embeddings import load_video_bank

VARIANTS = {
    "baseline (current)": {},
    "filter: none": {"l_freq": None, "h_freq": None},
    "filter: high-pass 0.1 Hz only": {"l_freq": 0.1, "h_freq": None},
    "filter: high-pass 0.5 Hz only": {"l_freq": 0.5, "h_freq": None},
    "filter: low-pass 40 Hz only": {"l_freq": None, "h_freq": 40.0},
    "filter: band-pass 0.1-40 Hz": {"l_freq": 0.1},
    "filter: band-pass 1-40 Hz": {"l_freq": 1.0},
    "filter: band-pass 0.5-30 Hz": {"h_freq": 30.0},
    "filter: band-pass 0.5-20 Hz": {"h_freq": 20.0},
    "filter: band-pass 0.5-12 Hz": {"h_freq": 12.0},
    "reference: no average reference": {"average_reference": False},
    "alignment: no Euclidean alignment": {"euclidean_alignment": False},
    "mean response: keep": {"remove_mean_response": False},
    "scaling: per channel": {"scaling": "channel"},
    "scaling: none": {"scaling": "none"},
    "clipping: none": {"clip_std": None},
    "clipping: 3 std": {"clip_std": 3.0},
    "trial offset: demean trials": {"demean_trials": True},
    "trial offset: detrend trials": {"detrend": True},
    "sampling: 200 Hz": {"resample": 200},
}


def ridge_mix(cfg, trn_parts, fs, tmin, emb):
    """Pooled ridge on all parts + one ridge per subject; returns predict(subject, X)."""
    pooled = RidgeDecoder(cfg["ridge"], fs, tmin).fit(np.concatenate([X for X, _ in trn_parts.values()]),
                                                       emb(np.concatenate([y for _, y in trn_parts.values()])))
    subj = {s: RidgeDecoder(cfg["ridge"], fs, tmin).fit(X, emb(y)) for s, (X, y) in trn_parts.items()}
    return lambda s, X: pooled.predict(X) + subj[s].predict(X)


def score(cfg, bank) -> dict:
    data = load_preprocessed(cfg)
    fs, tmin = data[0].fs, data[0].tmin
    emb = lambda ids: bank.emb[bank.lookup(np.asarray(ids).astype(str))]
    trn = {d.subject: (d.X, d.video_id.astype(int)) for d in data if d.split == "trn"}

    # cv over training videos
    videos = np.unique(np.concatenate([y for _, y in trn.values()]))
    folds = np.array_split(np.random.default_rng(0).permutation(videos), 5)
    cv = []
    for held in folds:
        held = set(held.tolist())
        parts, test = {}, {}
        for s, (X, y) in trn.items():
            m = np.array([v in held for v in y])
            parts[s], test[s] = (X[~m], y[~m]), (X[m], y[m])
        predict = ridge_mix(cfg, parts, fs, tmin, emb)
        for s, (X, y) in test.items():
            ids = np.unique(y)
            S = similarity(predict(s, X), emb(ids), True)
            col = np.searchsorted(ids, y)
            cv.append((S > S[np.arange(len(S)), col][:, None]).sum(1) / (len(ids) - 1))

    # dev test
    predict = ridge_mix(cfg, trn, fs, tmin, emb)
    cand = np.arange(1001, 1103)
    ranks = []
    for d in data:
        if d.split == "tst" and d.video_id is not None:
            S = similarity(predict(d.subject, d.X), emb(cand), True)
            t = d.video_id.astype(int) - 1001
            ranks.append((S > S[np.arange(len(S)), t][:, None]).sum(1))
    r = np.concatenate(ranks)
    return {"cv_rank": float(np.mean(np.concatenate(cv))), "top1": float(np.mean(r < 1)),
            "top5": float(np.mean(r < 5)), "mean_rank": float(r.mean() + 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of variant names")
    ap.add_argument("--combine", nargs="+", default=None, help="variant names to apply together")
    args = ap.parse_args()
    cfg = load_config(args.config)
    bank = load_video_bank(cfg)

    if args.combine:
        over = {}
        for name in args.combine:
            over.update(VARIANTS[name])
        todo = {"baseline (current)": {}, " + ".join(args.combine): over}
    else:
        todo = {n: o for n, o in VARIANTS.items()
                if not args.only or n == "baseline (current)" or any(x in n for x in args.only)}

    print(f"{'variant':38s} {'cv rank':>8} | {'top1':>6} {'top5':>6} {'rank':>5}   (cv: lower is better)")
    rows = []
    for name, over in todo.items():
        t = time.time()
        c = copy.deepcopy(cfg)
        c["preprocess"].update(over)
        s = score(c, bank)
        rows.append({"variant": name, "overrides": over, **s})
        print(f"{name:38s} {s['cv_rank']:8.4f} | {s['top1']:6.3f} {s['top5']:6.3f} {s['mean_rank']:5.1f}"
              f"   ({time.time() - t:.0f}s)", flush=True)
    out = resolve("models/xsubj/results") / f"sweep_preprocess_{time.strftime('%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
