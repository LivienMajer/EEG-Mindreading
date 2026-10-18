"""LOO check of Sinkhorn re-ranking (src/sinkhorn.py) on the candidate megafusion.

Candidate: 7 original members (E-D from checkpoint 5e3154e) + fc6, VideoMAE-l8, Daniel-style
fc6 and VideoMAE-l8 (docs/megafusion_rebuild.md, section 4). Fully nested: for each held-out
dev subject, the member weights are chosen on the other three exactly as in
src.loo_new_members, then tau (or "off") is chosen on the same three, and both are scored on
the held-out subject. Paired against the plain fusion on the same 408 recordings: exact
McNemar on top-1/top-5 hits, Wilcoxon on the true video's rank.

    python -m src.loo_sinkhorn
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np
from scipy import stats

from .loo_new_members import FINAL, GRID_ALEX, GRID_NEW, SUBS, old_members, score
from .sinkhorn import mean_rank_matrix, rank

BASE = ["_fc6", "_vmae8", "_dfc6", "_dvmae8"]
TAUS = (None, 1, 2, 3, 5, 8, 12)           # None = plain mean-rank fusion


def true_rank(ranked, y):
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", default="work/prep_variants")
    p.add_argument("--tst", default="work/tst_ranked")
    p.add_argument("--probe", default="work/xsubj_probe")
    p.add_argument("--ed-dir", default="work/ed_5e3154e")
    args = p.parse_args()
    tst, probe, new, ed = Path(args.tst), Path(args.probe), Path(args.new_dir), Path(args.ed_dir)
    truths = {s: np.load(tst / f"truth_{s}_s0.npy") for s in SUBS}
    M = {s: old_members(s, tst, probe, "emb_train__clip-vitb32_s0", ed)
         + [np.load(new / f"A{t}_{s}_s0.npy") for t in BASE] for s in SUBS}
    grid = [(a, w) for a in GRID_ALEX for w in itertools.product(GRID_NEW, repeat=len(BASE))]
    W = lambda a, w: [FINAL[0], a, *FINAL[2:]] + list(w)

    def ranks(s, w, tau):
        return true_rank(rank(-mean_rank_matrix(M[s], w), tau), truths[s])

    def mean_hits(subs, w, tau):
        r = [ranks(s, w, tau) for s in subs]
        return np.mean([np.mean(x == 1) for x in r]), np.mean([np.mean(x <= 5) for x in r])

    plain, sink = [], []
    for h in SUBS:
        dev = [s for s in SUBS if s != h]
        w = W(*max(grid, key=lambda g: score(M, truths, dev, W(*g))))
        tau = max(TAUS, key=lambda t: mean_hits(dev, w, t))
        plain.append(ranks(h, w, None))
        sink.append(ranks(h, w, tau))
        print(f"hold {h}: weights {':'.join(map(str, w))}, tau {tau} -> top-1 {np.mean(sink[-1] == 1):.4f}"
              f"  top-5 {np.mean(sink[-1] <= 5):.4f}  (plain {np.mean(plain[-1] == 1):.4f} / "
              f"{np.mean(plain[-1] <= 5):.4f})")
    ra, rb = np.concatenate(plain), np.concatenate(sink)
    for name, r in (("plain", ra), ("Sinkhorn", rb)):
        print(f"LOO {name}: top-1 {np.mean(r == 1):.4f}  top-5 {np.mean(r <= 5):.4f}  mean rank {r.mean():.2f}")
    print(f"paired on {len(ra)} recordings:")
    for k in (1, 5):
        g, l = int(np.sum((rb <= k) & (ra > k))), int(np.sum((ra <= k) & (rb > k)))
        print(f"  top-{k}: gained {g}, lost {l} (exact McNemar p = {stats.binomtest(g, g + l).pvalue:.4f})")
    print(f"  rank better in {np.sum(rb < ra)}, worse in {np.sum(rb > ra)} "
          f"(Wilcoxon p = {stats.wilcoxon(rb, ra).pvalue:.2g})")


if __name__ == "__main__":
    main()
