"""LOO weight selection for the megafusion WITH the new InfoNCE members (fc6, VideoMAE-l8).

Extends loo_generalization: same 7 members, files and fusion, plus members of the same
type as members 1-2 (per-subject InfoNCE EEGNet) trained toward other video banks
(src.prep_variants --emb ... --tag ...; dumps A<tag>_<sub>_s0.npy, label-sorted like
work/tst_ranked). The weights of the original members stay at the final 9:9:5:2:1:1:1
except member 2 (alexnet-f3), which may be dropped; the new members' weights are searched.

For each held-out dev subject, weights are chosen on the OTHER three (mean top-1, ties ->
top-5) and scored on the held-out one, exactly as in loo_generalization. Reported next to
the current final configuration (9:9:5:2:1:1:1, LOO 22.30 / 48.28).

Before fusing, each new member must score clearly above chance on its own: dumps in the
wrong row order (e.g. original tst order instead of label-sorted) fall to chance and fail.

Usage (from the repo root, with the members of loo_generalization in work/):
  python -m src.loo_new_members --new-dir work/prep_variants --tags _fc6 _vmae8
"""

from __future__ import annotations

import argparse
import itertools
from pathlib import Path

import numpy as np

from .loo_generalization import fuse, topk

SUBS = ["01", "02", "03", "04"]
FINAL = [9, 9, 5, 2, 1, 1, 1]
GRID_NEW = (0, 3, 5, 9)
GRID_ALEX = (9, 0)


def old_members(s: str, tst: Path, probe: Path, e: str) -> list[np.ndarray]:
    """The 7 megafusion members, in loo_generalization order."""
    others = [o for o in SUBS if o != s]
    m = [np.load(tst / f"infonce_{s}_emb_train__clip-vitb32_s0.npy"),
         np.load(tst / f"infonce_{s}_emb_train__alexnet-f3_s0.npy"),
         np.load(tst / f"ed_{s}_xsubj_s0.npy"),
         np.load(tst / f"ridge_{s}_{e}.npy")]
    return m + [np.load(probe / f"ranked_tr{o}_te{s}_{e}.npy") for o in others]


def score(M, truths, subs, weights):
    res = [topk(fuse(M[s], weights), truths[s]) for s in subs]
    return float(np.mean([r[0] for r in res])), float(np.mean([r[1] for r in res]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--new-dir", default="work/prep_variants")
    p.add_argument("--tags", nargs="+", default=["_fc6", "_vmae8"])
    p.add_argument("--tst", default="work/tst_ranked")
    p.add_argument("--probe", default="work/xsubj_probe")
    p.add_argument("--emb-tag", default="emb_train__clip-vitb32_s0")
    args = p.parse_args()
    tst, probe, new = Path(args.tst), Path(args.probe), Path(args.new_dir)

    truths = {s: np.load(tst / f"truth_{s}_s0.npy") for s in SUBS}
    M = {s: old_members(s, tst, probe, args.emb_tag)
            + [np.load(new / f"A{t}_{s}_s0.npy") for t in args.tags] for s in SUBS}
    # row-order check: every subject's label-sorted truth is 1001..1102, so comparing truth
    # files proves nothing; a member whose rows are misaligned falls to chance instead.
    for t, k in zip(args.tags, range(7, 7 + len(args.tags))):
        t5 = np.mean([topk(M[s][k], truths[s])[1] for s in SUBS])
        if t5 < 2 * 5 / 102:
            raise SystemExit(f"{t}: solo top-5 {t5:.3f} is near chance (0.049): rows not "
                             f"label-sorted like {tst}?")

    def weights(w_alex, w_new):
        return [FINAL[0], w_alex, *FINAL[2:]] + list(w_new)

    grid = [(a, w) for a in GRID_ALEX for w in itertools.product(GRID_NEW, repeat=len(args.tags))]
    names = "clip, alexnet-f3, ED, ridge, 3 transfers, " + ", ".join(t.strip("_") for t in args.tags)
    print(f"members: {names}")

    base = [topk(fuse(M[s], weights(9, [0] * len(args.tags))), truths[s]) for s in SUBS]
    print(f"current final 9:9:5:2:1:1:1 (new members off): top-1 {np.mean([b[0] for b in base]):.4f}"
          f"  top-5 {np.mean([b[1] for b in base]):.4f}")

    held = []
    for h in SUBS:
        dev = [s for s in SUBS if s != h]
        best = max(grid, key=lambda g: score(M, truths, dev, weights(*g)))
        w = weights(*best)
        t1, t5 = topk(fuse(M[h], w), truths[h])
        held.append((t1, t5))
        print(f"  hold {h}: weights {':'.join(map(str, w))} chosen on {dev} -> held top-1 {t1:.4f}"
              f"  top-5 {t5:.4f}")
    print(f"LOO held-out mean WITH new members: top-1 {np.mean([x[0] for x in held]):.4f}"
          f"  top-5 {np.mean([x[1] for x in held]):.4f}")
    oracle = max(grid, key=lambda g: score(M, truths, SUBS, weights(*g)))
    o1, o5 = score(M, truths, SUBS, weights(*oracle))
    print(f"(oracle on all 4, for reference only: {':'.join(map(str, weights(*oracle)))} -> "
          f"{o1:.4f} / {o5:.4f})")


if __name__ == "__main__":
    main()
