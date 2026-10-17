"""Would a new target space help the fusion? Candidate member vs the CLIP InfoNCE member.

The megafusion gains from members that solve DIFFERENT trials (two video spaces with
zero top-1 overlap gave +5pp). This scores candidates trained with
`src.prep_variants --variants A --emb <bank> --tag <tag>` on sub-01..04 tst:

  solo        top-1 / top-5 / mean rank of the candidate
  new hits    trials the candidate ranks first that the CLIP member does not
  fusion      weighted mean-rank fusion CLIP : candidate at 1:1 and 2:1 (fuse_mean_rank)
  paired      Wilcoxon on the true video's rank, fusion vs CLIP alone

Weights are not tuned beyond these two: a quick screen, not model selection. Members
that pass go into loo_generalization with the other megafusion members.

Usage:
  python -m src.member_check --tags _clap _loud _vmae8 _fc6
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from .ensemble_rank import fuse_mean_rank
from .prep_variants import OUT

SUBS = ["01", "02", "03", "04"]


def ranks(ranked: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.argmax(ranked == y[:, None], axis=1) + 1


def load(tag: str, seed: int):
    return [np.load(OUT / f"A{tag}_{s}_s{seed}.npy") for s in SUBS]


def describe(name: str, r: np.ndarray) -> str:
    return (f"{name:28s} top-1 {np.mean(r == 1) * 100:5.1f}%  top-5 {np.mean(r <= 5) * 100:5.1f}%  "
            f"mean rank {r.mean():5.1f}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--base", default="", help="tag of the reference member (default: CLIP)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    truth = [np.load(OUT / f"truth_{s}.npy") for s in SUBS]
    base = load(args.base, args.seed)
    rb = np.concatenate([ranks(b, y) for b, y in zip(base, truth)])
    print(f"sub-01..04 tst, {len(rb)} recordings, seed {args.seed}")
    print(describe("CLIP member (reference)", rb))
    for tag in args.tags:
        cand = load(tag, args.seed)
        rc = np.concatenate([ranks(c, y) for c, y in zip(cand, truth)])
        new = int(np.sum((rc == 1) & (rb != 1)))
        shared = int(np.sum((rc == 1) & (rb == 1)))
        print(f"\n{describe(tag.strip('_') + ' solo', rc)}")
        print(f"  top-1 hits: {np.sum(rc == 1)}, of which {new} new (CLIP misses them), {shared} shared")
        for w in ([1, 1], [2, 1]):
            rf = np.concatenate([ranks(fuse_mean_rank([b, c], w), y)
                                 for b, c, y in zip(base, cand, truth)])
            pw = stats.wilcoxon(rf, rb).pvalue if np.any(rf != rb) else 1.0
            print(f"  {describe(f'CLIP + {tag.strip(chr(95))} {w[0]}:{w[1]}', rf)}  "
                  f"(rank vs CLIP: better {np.sum(rf < rb)}, worse {np.sum(rf > rb)}, p = {pw:.3f})")


if __name__ == "__main__":
    main()
