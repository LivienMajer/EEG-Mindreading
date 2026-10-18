"""Does an extra member improve fusion B? (docs/megafusion_rebuild.md, sections 11 and 13)

Fusion B: clip 9, alexnet-f3 9, ridge 2, transfers 1 each, fc6 9, VideoMAE-l8 5, Daniel's
ensemble (_dnew) 60, Sinkhorn. For each tag in --tags, separately: the extra member's weight
(0 = off) and tau are chosen on three dev subjects and scored on the fourth; B's own weights
stay fixed. Paired against B (tau chosen the same way) on the same 408 recordings.

    python -m src.loo_fusion_b_extra --tags _late_fc6 _loud
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from .loo_new_members import SUBS, old_members
from .sinkhorn import mean_rank_matrix, rank

B_WEIGHTS = [9, 9, 2, 1, 1, 1, 9, 5, 60]
TAUS = (2, 3, 5)
EXTRA = (0, 3, 5, 9, 15, 30)


def true_rank(ranked, y):
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--new-dir", default="work/prep_variants")
    args = p.parse_args()
    tst, probe, ed, new = Path("work/tst_ranked"), Path("work/xsubj_probe"), Path("work/ed_5e3154e"), Path(args.new_dir)
    Y = {s: np.load(tst / f"truth_{s}_s0.npy") for s in SUBS}
    B = {s: [m for i, m in enumerate(old_members(s, tst, probe, "emb_train__clip-vitb32_s0", ed)) if i != 2]
         + [np.load(new / f"A{t}_{s}_s0.npy") for t in ("_fc6", "_vmae8", "_dnew")] for s in SUBS}

    def loo(M, extra_grid):
        cache = {}

        def r(s, w, tau):
            if (s, w, tau) not in cache:
                cache[(s, w, tau)] = true_rank(rank(-mean_rank_matrix(M[s], B_WEIGHTS + list(w)), tau), Y[s])
            return cache[(s, w, tau)]
        held, chosen = [], []
        for h in SUBS:
            dev = [s for s in SUBS if s != h]
            o = max(((w, t) for w in extra_grid for t in TAUS),
                    key=lambda o: (np.mean([np.mean(r(s, *o) == 1) for s in dev]),
                                   np.mean([np.mean(r(s, *o) <= 5) for s in dev])))
            held.append(r(h, *o))
            chosen.append(o)
        return np.concatenate(held), chosen

    rb, ch = loo(B, [()])
    print(f"B: top-1 {np.mean(rb == 1) * 100:.2f}  top-5 {np.mean(rb <= 5) * 100:.2f}  mean rank {rb.mean():.2f}"
          f"  (tau per fold {[c[1] for c in ch]})")
    for t in args.tags:
        M = {s: B[s] + [np.load(new / f"A{t}_{s}_s0.npy")] for s in SUBS}
        solo = np.concatenate([true_rank(M[s][-1], Y[s]) for s in SUBS])
        rx, ch = loo(M, [(w,) for w in EXTRA])
        print(f"B + {t.strip('_')}: top-1 {np.mean(rx == 1) * 100:.2f}  top-5 {np.mean(rx <= 5) * 100:.2f}"
              f"  mean rank {rx.mean():.2f}  (alone {np.mean(solo == 1) * 100:.1f} / {np.mean(solo <= 5) * 100:.1f}"
              f" / {solo.mean():.1f}; weight, tau per fold {[(c[0][0], c[1]) for c in ch]})")
        line = []
        for k in (1, 5):
            g, l = int(np.sum((rx <= k) & (rb > k))), int(np.sum((rb <= k) & (rx > k)))
            line.append(f"top-{k} +{g}/-{l} p={stats.binomtest(g, g + l).pvalue if g + l else 1:.3f}")
        pw = stats.wilcoxon(rx, rb).pvalue if np.any(rx != rb) else 1.0
        print("    " + "; ".join(line) + f"; rank better {np.sum(rx < rb)}/worse {np.sum(rx > rb)} p={pw:.2g}")


if __name__ == "__main__":
    main()
