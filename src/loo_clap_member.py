"""Does Daniel's audio member (CLAP ridge) improve the candidate megafusion?

Candidate: the 7 original members (E-D from checkpoint 5e3154e) + fc6, VideoMAE-l8,
Daniel-style fc6 and VideoMAE-l8 (docs/megafusion_rebuild.md, section 4). Same LOO protocol
as src.loo_new_members (weights chosen on 3 dev subjects, scored on the 4th), once without
and once with the CLAP member, then a paired comparison of the two on the same 408 recordings
(exact McNemar on top-1/top-5 hits, Wilcoxon on the true video's rank).

The CLAP member comes from Daniel's branch (src.xsubj.audio_member, label-sorted dev dumps):
  for s in 01 02 03 04; do git show origin/feature/clip-lowlevel-forward-model:models/xsubj/members/clap_ridge/dev_sorted/clap_${s}_ridge_s0.npy \
      > work/prep_variants/A_dclap_${s}_s0.npy; done
  python -m src.loo_clap_member
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
GRID_CLAP = (0, 1, 2, 3, 5, 9)  # finer at the low end: Daniel's own audio weight is small (0.3)


def true_rank(ranked, y):
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", default="work/prep_variants")
    p.add_argument("--tst", default="work/tst_ranked")
    p.add_argument("--probe", default="work/xsubj_probe")
    p.add_argument("--ed-dir", default="work/ed_5e3154e")
    p.add_argument("--clap-tag", default="_dclap")
    args = p.parse_args()
    tst, probe, new, ed = Path(args.tst), Path(args.probe), Path(args.new_dir), Path(args.ed_dir)
    truths = {s: np.load(tst / f"truth_{s}_s0.npy") for s in SUBS}

    def run(tags, grids):
        M = {s: old_members(s, tst, probe, "emb_train__clip-vitb32_s0", ed)
             + [np.load(new / f"A{t}_{s}_s0.npy") for t in tags] for s in SUBS}
        grid = [(a, w) for a in GRID_ALEX for w in itertools.product(*grids)]
        W = lambda a, w: [FINAL[0], a, *FINAL[2:]] + list(w)
        ranks = []
        print(" + ".join(t.strip("_") for t in tags))
        for h in SUBS:
            dev = [s for s in SUBS if s != h]
            w = W(*max(grid, key=lambda g: score(M, truths, dev, W(*g))))
            ranks.append(true_rank(fuse(M[h], w), truths[h]))
            print(f"  hold {h}: weights {':'.join(map(str, w))} -> top-1 {np.mean(ranks[-1] == 1):.4f}"
                  f"  top-5 {np.mean(ranks[-1] <= 5):.4f}")
        r = np.concatenate(ranks)
        print(f"  LOO: top-1 {np.mean(r == 1):.4f}  top-5 {np.mean(r <= 5):.4f}  mean rank {r.mean():.2f}")
        return r, M

    ra, _ = run(BASE, [GRID_NEW] * len(BASE))
    rb, M = run(BASE + [args.clap_tag], [GRID_NEW] * len(BASE) + [GRID_CLAP])
    solo = np.concatenate([true_rank(M[s][-1], truths[s]) for s in SUBS])
    if np.mean(solo <= 5) < 2 * 5 / 102:
        raise SystemExit("CLAP member near chance on its own: rows not label-sorted?")
    print(f"CLAP member alone: top-1 {np.mean(solo == 1):.4f}  top-5 {np.mean(solo <= 5):.4f}"
          f"  mean rank {solo.mean():.2f}")
    print(f"with vs without CLAP, paired on {len(ra)} recordings:")
    for k in (1, 5):
        g, l = int(np.sum((rb <= k) & (ra > k))), int(np.sum((ra <= k) & (rb > k)))
        p_mc = stats.binomtest(g, g + l).pvalue if g + l else 1.0
        print(f"  top-{k}: gained {g}, lost {l} (exact McNemar p = {p_mc:.3f})")
    p_w = stats.wilcoxon(rb, ra).pvalue if np.any(rb != ra) else 1.0
    print(f"  rank better in {np.sum(rb < ra)}, worse in {np.sum(rb > ra)} (Wilcoxon p = {p_w:.2g})")


if __name__ == "__main__":
    main()
