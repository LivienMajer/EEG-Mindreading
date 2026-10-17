"""Target-space comparison WITHOUT the test sets: contrastive_pipeline's holdout mode.

`contrastive_pipeline --holdout 102` hides 102 of a subject's TRAINING videos, trains on
the rest and ranks the hidden trials among the 102 hidden videos; no tst recording or
label is touched, so sub-05/06 can be included. For a given subject and --seed the
hidden videos are the same for every target space (the split depends only on the seed),
so targets are compared trial by trial.

This is the clean check of the member choice in docs/megafusion_rebuild.md: AlexNet fc6
and VideoMAE-l8 were picked with the RSA (training data) but their decoding scores were
first seen on the dev test sets.

Usage:
  for t in clip-vitb32 alexnet-fc6 videomae-l8; do for s in 01 02 03 04 05 06; do for k in 0 1 2; do
    python -m src.contrastive_pipeline --data-dir <root> --sub $s --emb-train work/emb_train__$t.npz \\
        --emb-test work/emb_test__$t.npz --holdout 102 --seed $k; done; done; done
  python -m src.holdout_compare --targets clip-vitb32 alexnet-fc6 videomae-l8
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

H = Path("work/holdout")


def ranks(target: str, sub: str, seed: int) -> np.ndarray:
    r = np.load(H / f"ranked_sub{sub}_emb_train__{target}_s{seed}_n{seed}.npy")
    y = np.load(H / f"y_sub{sub}_s{seed}.npy")
    return np.argmax(r == y[:, None], axis=1) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--targets", nargs="+", required=True, help="first = reference")
    p.add_argument("--subs", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    p.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    args = p.parse_args()

    R = {t: {(s, k): ranks(t, s, k) for s in args.subs for k in args.seeds} for t in args.targets}
    ref = args.targets[0]
    n = sum(len(v) for v in R[ref].values())
    print(f"holdout: 102 hidden training videos per subject and seed, {len(args.subs)} subjects x "
          f"{len(args.seeds)} seeds = {n} hidden trials (chance top-1 0.98%, top-5 4.9%)\n")
    print(f"{'target':14s} {'top-1':>6s} {'top-5':>6s} {'rank':>6s}   per subject top-1 (mean over seeds)")
    for t in args.targets:
        allr = np.concatenate([R[t][key] for key in sorted(R[t])])
        per = [np.mean(np.concatenate([R[t][(s, k)] for k in args.seeds]) == 1) * 100 for s in args.subs]
        print(f"{t:14s} {np.mean(allr == 1) * 100:5.1f}% {np.mean(allr <= 5) * 100:5.1f}% {allr.mean():6.1f}   "
              + "  ".join(f"{s}:{v:4.1f}" for s, v in zip(args.subs, per)))
    rb = np.concatenate([R[ref][key] for key in sorted(R[ref])])
    for t in args.targets[1:]:
        rt = np.concatenate([R[t][key] for key in sorted(R[t])])
        print(f"\n{t} vs {ref}, paired on the same {len(rt)} hidden trials:")
        for k in (1, 5):
            gain, loss = int(np.sum((rt <= k) & (rb > k))), int(np.sum((rb <= k) & (rt > k)))
            print(f"  top-{k}: +{gain} / -{loss} (exact McNemar p = {stats.binomtest(gain, gain + loss).pvalue:.2g})")
        print(f"  rank better in {np.sum(rt < rb)}, worse in {np.sum(rt > rb)} "
              f"(Wilcoxon p = {stats.wilcoxon(rt, rb).pvalue:.2g})")
        better = [s for s in args.subs
                  if np.mean(np.concatenate([R[t][(s, k)] for k in args.seeds])) <
                  np.mean(np.concatenate([R[ref][(s, k)] for k in args.seeds]))]
        print(f"  lower mean rank than {ref} in {len(better)}/{len(args.subs)} subjects: {', '.join(better)}")


if __name__ == "__main__":
    main()
