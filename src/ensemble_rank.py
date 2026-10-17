"""Rank-fusion of multiple holdout/submission ranking files (mean rank / RRF).

Finds ranking dumps in work/holdout (or submission rankings), fuses them by
mean rank, reports top-1/top-5 on holdout, and can write a fused submission.

Usage:
  python -m src.ensemble_rank --rankings work/holdout/ranked_sub01_*.npy \
      --truth work/holdout/y_sub01_s0.npy
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np


def fuse_mean_rank(rankings: list[np.ndarray],
                   weights: list[float] | None = None) -> np.ndarray:
    """Fuse (n, k) ID-ranked lists -> single (n, k) ranking by (weighted) mean rank."""
    n, k = rankings[0].shape
    cand = sorted(set(int(v) for row in rankings[0] for v in row[:k]))
    pos = {v: j for j, v in enumerate(cand)}
    P = len(cand)
    w = np.ones(len(rankings)) if weights is None else np.asarray(weights, float)
    w = w / w.sum()
    avg = np.zeros((n, P))
    for r, wi in zip(rankings, w):
        for i in range(n):
            rv = np.full(P, k, dtype=np.float64)
            for p, v in enumerate(r[i]):
                rv[pos[int(v)]] = p
            avg[i] += wi * rv
    avg /= len(rankings)
    ens = np.empty_like(rankings[0])
    for i in range(n):
        ids = rankings[0][i]
        ens[i] = ids[np.argsort(avg[i][[pos[int(v)] for v in ids]])]
    return ens


def topk_acc(ranked: np.ndarray, y: np.ndarray):
    t1 = float((ranked[:, 0] == y).mean())
    t5 = float((ranked[:, :5] == y[:, None]).any(1).mean())
    return t1, t5


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--rankings", nargs="+", required=True,
                   help="glob pattern(s) or explicit paths of ranking .npy files")
    p.add_argument("--truth", required=True, help=".npy with per-trial video ids")
    p.add_argument("--out", default=None, help="optional output .npy for fused ranking")
    p.add_argument("--subject", default="01", help="subject tag for score logging")
    p.add_argument("--experiment", default="rank-ensemble",
                   help="experiment tag for score logging")
    p.add_argument("--weights", nargs="+", type=float, default=None,
                   help="fusion weights per ranking file (default: equal). "
                        "E.g. `--weights 3 1` down-weights the weak member.")
    args = p.parse_args()

    paths: list[str] = []
    for pat in args.rankings:
        paths.extend(sorted(glob.glob(pat)) or [pat])
    rankings = [np.load(f) for f in paths]
    y = np.load(args.truth)

    print(f"fusing {len(rankings)} rankings from {len(set(paths))} files")
    for f, r in zip(paths, rankings):
        t1, t5 = topk_acc(r, y)
        print(f"  {Path(f).name}: top1 {t1:.4f}  top5 {t5:.4f}")

    ens = fuse_mean_rank(rankings, args.weights)
    t1, t5 = topk_acc(ens, y)
    print(f"\nFUSED (mean rank): top1 {t1:.4f}  top5 {t5:.4f}")

    from .log_score import log_score
    log_score(experiment=args.experiment, subject=args.subject, embedding="+".join(
        Path(f).stem for f in sorted(set(paths))),
              top1=t1, top5=t5,
              note=f"mean-rank fusion of {len(rankings)} rankings")
    if args.out:
        np.save(args.out, ens)
        print(f"saved -> {args.out}")


if __name__ == "__main__":
    main()
