"""Does one extra member improve the candidate megafusion? (generalises src.loo_clap_member)

Candidate: the 7 original members (E-D from checkpoint 5e3154e) + fc6, VideoMAE-l8,
Daniel-style fc6 and VideoMAE-l8 (docs/megafusion_rebuild.md, section 4). For each tag in
--tags, separately: LOO (weights chosen on 3 dev subjects, scored on the 4th) without and
with that member, then paired on the same 408 recordings (exact McNemar on top-1/top-5,
Wilcoxon on the true video's rank). Members: label-sorted dumps <new-dir>/A<tag>_XX_s0.npy,
e.g. from src.ridge_member.

    python -m src.loo_extra_member --tags _rconv2e _rclip9
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

from .loo_generalization import fuse
from .loo_new_members import FINAL, GRID_ALEX, GRID_NEW, SUBS, old_members, score

BASE = ["_fc6", "_vmae8", "_dfc6", "_dvmae8"]
GRID_EXTRA = (0, 1, 2, 3, 5, 9)


def true_rank(ranked, y):
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tags", nargs="+", required=True)
    p.add_argument("--new-dir", default="work/prep_variants")
    p.add_argument("--tst", default="work/tst_ranked")
    p.add_argument("--probe", default="work/xsubj_probe")
    p.add_argument("--ed-dir", default="work/ed_5e3154e")
    args = p.parse_args()
    tst, probe, new, ed = Path(args.tst), Path(args.probe), Path(args.new_dir), Path(args.ed_dir)
    truths = {s: np.load(tst / f"truth_{s}_s0.npy") for s in SUBS}
    base = {s: old_members(s, tst, probe, "emb_train__clip-vitb32_s0", ed)
            + [np.load(new / f"A{t}_{s}_s0.npy") for t in BASE] for s in SUBS}

    def loo(M, grids):
        grid = [(a, w) for a in GRID_ALEX for w in itertools.product(*grids)]
        W = lambda a, w: [FINAL[0], a, *FINAL[2:]] + list(w)
        ranks, chosen = [], []
        for h in SUBS:
            w = W(*max(grid, key=lambda g: score(M, truths, [s for s in SUBS if s != h], W(*g))))
            ranks.append(true_rank(fuse(M[h], w), truths[h]))
            chosen.append(w[-1])
        return np.concatenate(ranks), chosen

    ra, _ = loo(base, [GRID_NEW] * len(BASE))
    print(f"candidate: top-1 {np.mean(ra == 1):.4f}  top-5 {np.mean(ra <= 5):.4f}  mean rank {ra.mean():.2f}")
    for t in args.tags:
        M = {s: base[s] + [np.load(new / f"A{t}_{s}_s0.npy")] for s in SUBS}
        solo = np.concatenate([true_rank(M[s][-1], truths[s]) for s in SUBS])
        if np.mean(solo <= 5) < 2 * 5 / 102:
            raise SystemExit(f"{t}: near chance on its own, rows not label-sorted?")
        rb, w = loo(M, [GRID_NEW] * len(BASE) + [GRID_EXTRA])
        line = [f"+ {t.strip('_')}: top-1 {np.mean(rb == 1):.4f}  top-5 {np.mean(rb <= 5):.4f}"
                f"  mean rank {rb.mean():.2f}  (alone {np.mean(solo == 1):.4f} / {np.mean(solo <= 5):.4f}"
                f" / {solo.mean():.1f}; weight per fold {w})"]
        for k in (1, 5):
            g, l = int(np.sum((rb <= k) & (ra > k))), int(np.sum((ra <= k) & (rb > k)))
            line.append(f"top-{k} +{g}/-{l} p={stats.binomtest(g, g + l).pvalue if g + l else 1:.3f}")
        p_w = stats.wilcoxon(rb, ra).pvalue if np.any(rb != ra) else 1.0
        line.append(f"rank better {np.sum(rb < ra)} / worse {np.sum(rb > ra)} p={p_w:.2g}")
        print(line[0] + "\n    " + ";  ".join(line[1:]), flush=True)


if __name__ == "__main__":
    main()
