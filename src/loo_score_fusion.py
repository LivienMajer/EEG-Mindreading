"""LOO: fuse similarity SCORES instead of ranks, with and without Sinkhorn re-ranking.

Same 11 members as the candidate megafusion (7 original, E-D from checkpoint 5e3154e,
+ fc6, VideoMAE-l8, Daniel-style fc6 and VideoMAE-l8), read from the score dumps
(<ranking>.scores.npy, src/score_dump.py). Each member's table is z-scored per recording
(over the 102 candidates; "row") or, in a second variant, also per candidate (over the 102
recordings; "row+col", as Daniel's per-candidate normalisation), then summed with weights.
Mean-rank fusion keeps only the order of each member's candidates; score fusion also keeps
how far apart they are, i.e. how sure the member is.

Fully nested, as src.loo_sinkhorn: for each held-out dev subject the weights (same grid as
src.loo_new_members) and then tau (Sinkhorn, or off) are chosen on the other three and
scored on the held-out one. Reported next to the mean-rank fusion (plain and Sinkhorn) on
the same 408 recordings, with paired tests.

    python -m src.loo_score_fusion
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

from .loo_new_members import FINAL, GRID_ALEX, GRID_NEW, SUBS
from .sinkhorn import mean_rank_matrix, sinkhorn

BASE = ["_fc6", "_vmae8", "_dfc6", "_dvmae8"]
TAUS_RANK = (None, 1, 2, 3, 5, 8, 12)               # mean-rank units
TAUS_Z = (None, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0)      # z units


def member_paths(s, tst, probe, ed, new, e="emb_train__clip-vitb32_s0"):
    others = [o for o in SUBS if o != s]
    return ([tst / f"infonce_{s}_{e}.npy", tst / f"infonce_{s}_emb_train__alexnet-f3_s0.npy",
             ed / f"ed_{s}_xsubj_s0.npy", tst / f"ridge_{s}_{e}.npy"]
            + [probe / f"ranked_tr{o}_te{s}_{e}.npy" for o in others]
            + [new / f"A{t}_{s}_s0.npy" for t in BASE])


def zrow(S):
    return (S - S.mean(1, keepdims=True)) / (S.std(1, keepdims=True) + 1e-12)


def zcol(S):
    return (S - S.mean(0, keepdims=True)) / (S.std(0, keepdims=True) + 1e-12)


def true_rank(score):
    """Rows are label-sorted and every subject's truth is 1001..1102, so recording i's true
    candidate is column i. Rank 1 = best; ties as a stable argsort (lower id first)."""
    d = np.diag(score)[:, None]
    earlier = np.tri(len(score), score.shape[1], -1, dtype=bool)     # columns j < i
    return (score > d).sum(1) + ((score == d) & earlier).sum(1) + 1


def apply_tau(score, tau):
    return score if tau is None else np.log(sinkhorn(score, tau) + 1e-300)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", default="work/prep_variants")
    p.add_argument("--tst", default="work/tst_ranked")
    p.add_argument("--probe", default="work/xsubj_probe")
    p.add_argument("--ed-dir", default="work/ed_5e3154e")
    args = p.parse_args()
    tst, probe, new, ed = Path(args.tst), Path(args.probe), Path(args.new_dir), Path(args.ed_dir)

    paths = {s: member_paths(s, tst, probe, ed, new) for s in SUBS}
    for s in SUBS:                                     # rows must be label-sorted (1001..1102)
        assert (np.load(tst / f"truth_{s}_s0.npy") == np.arange(1001, 1103)).all()
    R = {s: [np.load(f) for f in paths[s]] for s in SUBS}
    S = {s: [np.load(f.with_name(f.stem + ".scores.npy")) for f in paths[s]] for s in SUBS}
    for s in SUBS:                                     # the scores must give back the rankings
        for f, r, sc in zip(paths[s], R[s], S[s]):
            agree = (np.arange(1001, 1103)[np.argsort(-sc, axis=1, kind="stable")] == r).mean()
            assert agree > 0.99, (f, agree)
    tables = {"rank": {s: [-mean_rank_matrix([r], [1]) for r in R[s]] for s in SUBS},
              "z row": {s: [zrow(x) for x in S[s]] for s in SUBS},
              "z row+col": {s: [zrow(zcol(x)) for x in S[s]] for s in SUBS}}
    stack = {k: {s: np.stack(v[s]) for s in SUBS} for k, v in tables.items()}

    grid = [(a, w) for a in GRID_ALEX for w in itertools.product(GRID_NEW, repeat=len(BASE))]
    W = lambda a, w: np.array([FINAL[0], a, *FINAL[2:]] + list(w), float)

    def fused(kind, s, w):
        return np.tensordot(w / w.sum(), stack[kind][s], axes=1)

    def hits(kind, subs, w, tau=None):
        r = [true_rank(apply_tau(fused(kind, s, w), tau)) for s in subs]
        return np.mean([np.mean(x == 1) for x in r]), np.mean([np.mean(x <= 5) for x in r])

    results = {}
    for kind in stack:
        taus = TAUS_RANK if kind == "rank" else TAUS_Z
        plain, sink = [], []
        for h in SUBS:
            dev = [s for s in SUBS if s != h]
            w = W(*max(grid, key=lambda g: hits(kind, dev, W(*g))))
            tau = max(taus, key=lambda t: hits(kind, dev, w, t))
            plain.append(true_rank(fused(kind, h, w)))
            sink.append(true_rank(apply_tau(fused(kind, h, w), tau)))
            print(f"[{kind}] hold {h}: weights {':'.join(str(int(x)) for x in w)}, tau {tau}", flush=True)
        results[kind] = np.concatenate(plain)
        results[kind + " + Sinkhorn"] = np.concatenate(sink)

    print("\nLOO held-out, sub-01..04 (408 recordings):")
    for k, r in results.items():
        print(f"  {k:22s} top-1 {np.mean(r == 1):.4f}  top-5 {np.mean(r <= 5):.4f}  mean rank {r.mean():.2f}")
    print("\npaired vs mean-rank fusion (plain) and vs mean-rank + Sinkhorn:")
    for ref in ("rank", "rank + Sinkhorn"):
        for k, rb in results.items():
            if k == ref or k.startswith("rank"):
                continue
            ra, line = results[ref], []
            for kk in (1, 5):
                g, l = int(np.sum((rb <= kk) & (ra > kk))), int(np.sum((ra <= kk) & (rb > kk)))
                line.append(f"top-{kk} +{g}/-{l} p={stats.binomtest(g, g + l).pvalue if g + l else 1:.3f}")
            pw = stats.wilcoxon(rb, ra).pvalue if np.any(rb != ra) else 1.0
            line.append(f"rank better {np.sum(rb < ra)}/worse {np.sum(rb > ra)} p={pw:.2g}")
            print(f"  {k:22s} vs {ref:16s}: " + ";  ".join(line))


if __name__ == "__main__":
    main()
