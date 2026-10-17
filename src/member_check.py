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

With --rsa-summary (results/rsa/rsa_summary.csv from eeg_pipeline.rsa on
feature/eeg-preprocessing): Spearman correlation between each target space's RSA
brain-likeness (peak r) and the decoding accuracy of the member trained toward it.
With --seeds: each seed of the reference member and their mean-rank average.

Usage:
  python -m src.member_check --tags _clap _loud _vmae8 _fc6
  python -m src.member_check --tags _conv2 _fc6 _fc7 _vmae4 _vmae8 _vmae12 _clap _loud \
      --rsa-summary rsa_summary.csv
  python -m src.member_check --tags _fc6 --seeds 0 1 2
"""

from __future__ import annotations

import argparse

import numpy as np
from scipy import stats

from .ensemble_rank import fuse_mean_rank
from .prep_variants import OUT

SUBS = ["01", "02", "03", "04"]
TAG_TO_RSA = {"": "clip_vitb32_daniel", "_conv2": "alexnet_conv2", "_fc6": "alexnet_fc6",
              "_fc7": "alexnet_fc7", "_vmae4": "videomae_l4", "_vmae8": "videomae_l8",
              "_vmae12": "videomae_l12", "_clap": "clap_audio", "_loud": "low_loudness"}


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
    p.add_argument("--rsa-summary", default=None, help="results/rsa/rsa_summary.csv")
    p.add_argument("--seeds", nargs="+", type=int, default=None,
                   help="report these seeds of the reference member and their average")
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

    if args.seeds:
        R = {k: load(args.base, k) for k in args.seeds}
        print(f"\nseeds of the reference member{' ' + args.base if args.base else ' (CLIP)'}:")
        for k, Rk in R.items():
            print("  " + describe(f"seed {k}", np.concatenate([ranks(r, y) for r, y in zip(Rk, truth)])))
        avg = np.concatenate([ranks(fuse_mean_rank([R[k][i] for k in args.seeds]), y)
                              for i, y in enumerate(truth)])
        print("  " + describe(f"mean-rank average of {len(args.seeds)} seeds", avg))
        r0 = np.concatenate([ranks(r, y) for r, y in zip(R[args.seeds[0]], truth)])
        h1, h0 = int(np.sum((avg == 1) & (r0 != 1))), int(np.sum((r0 == 1) & (avg != 1)))
        print(f"  vs seed {args.seeds[0]}: rank better {np.sum(avg < r0)}, worse {np.sum(avg > r0)} "
              f"(Wilcoxon p = {stats.wilcoxon(avg, r0).pvalue:.4f}); top-1 only-average {h1} vs "
              f"only-seed {h0} (p = {stats.binomtest(h1, h1 + h0).pvalue:.3f})")

    if args.rsa_summary:
        import csv
        rsa = {r["model"]: float(r["peak_r"]) for r in csv.DictReader(open(args.rsa_summary))}
        tags = [args.base] + list(args.tags)
        rows = [(t, rsa[TAG_TO_RSA[t]],
                 np.concatenate([ranks(c, y) for c, y in zip(load(t, args.seed), truth)]))
                for t in tags if TAG_TO_RSA.get(t) in rsa]
        print("\nRSA brain-likeness (peak r) vs decoding of the member trained toward it:")
        for t, x, r in rows:
            print(f"  {TAG_TO_RSA[t]:20s} RSA {x:.4f}  top-1 {np.mean(r == 1) * 100:5.1f}%  "
                  f"top-5 {np.mean(r <= 5) * 100:5.1f}%")
        for label, keep in (("all", lambda t: True), ("without loudness (30-dim)", lambda t: t != "_loud")):
            sel = [(x, r) for t, x, r in rows if keep(t)]
            for k in (1, 5):
                s = stats.spearmanr([x for x, _ in sel], [np.mean(r <= k) for _, r in sel])
                print(f"  {label}, n = {len(sel)}: rho(RSA, top-{k}) = {s.statistic:.2f} (p = {s.pvalue:.3f})")


if __name__ == "__main__":
    main()
