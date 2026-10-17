"""LOO test: does the E-A joint+finetune member ADD to the megafusion?

Same leave-one-dev-subject-out protocol as loo_generalization.py, but the
member list gains jointft_{sub} (own joint+ft ranking) and the grid gains
w_joint in {0, 0.5, 1, 2}. Everything else identical. If held-out beats the
8-member LOO (22.30/48.28), the joint member earns its place; if w_joint=0
keeps winning, it does not (still good failure analysis material).

Usage:  python -m src.loo_joint
"""

from __future__ import annotations

import numpy as np

from .log_score import log_score
from .loo_generalization import SUBS, fuse, topk

TST = "work/tst_ranked"
PROBE = "work/xsubj_probe"
E = "emb_train__clip-vitb32"
J = f"jointft_{{s}}_{E}_s0.npy"


def members(s: str) -> list[np.ndarray]:
    others = [o for o in SUBS if o != s]
    m = [
        np.load(f"{TST}/infonce_{s}_{E}_s0.npy"),
        np.load(f"{TST}/infonce_{s}_emb_train__alexnet-f3_s0.npy"),
        np.load(f"{TST}/ed_{s}_xsubj_s0.npy"),
        np.load(f"{TST}/ridge_{s}_{E}_s0.npy"),
    ]
    m += [np.load(f"{PROBE}/ranked_tr{o}_te{s}_{E}_s0.npy") for o in others]
    m += [np.load(f"{TST}/" + J.format(s=s))]
    return m


def main():
    truths = {s: np.load(f"{TST}/truth_{s}_s0.npy") for s in SUBS}
    M = {s: members(s) for s in SUBS}

    grid = [(w_ed, w_r, w_j)
            for w_ed in (3, 5, 7)
            for w_r in (1, 2)
            for w_j in (0, 0.5, 1, 2, 3, 5)]

    print("LOO weight-selection: megafusion + potentially jointft "
          "[clip 9, alexnet 9, ed w, ridge r, transfers 1 1 1, joint w_j]")
    held_scores = []
    for held in SUBS:
        dev = [d for d in SUBS if d != held]
        best, best_key = None, -1.0
        for w_ed, w_r, w_j in grid:
            weights = [9, 9, w_ed, w_r, 1, 1, 1, w_j]
            t1s = [topk(fuse(M[d], weights), truths[d])[0] for d in dev]
            key = float(np.mean(t1s))
            if key > best_key:
                best, best_key = (w_ed, w_r, w_j), key
        w_ed, w_r, w_j = best
        weights = [9, 9, w_ed, w_r, 1, 1, 1, w_j]
        t1, t5 = topk(fuse(M[held], weights), truths[held])
        held_scores.append((t1, t5))
        print(f"  hold {held}: selected w_ed={w_ed} w_ridge={w_r} "
              f"w_joint={w_j} (dev mean top1 {best_key:.4f})  ->  "
              f"held top1 {t1:.4f}  top5 {t5:.4f}")
        log_score(experiment="loo-heldout-joint", subject=held,
                  embedding="megafusion+jointft", top1=t1, top5=t5,
                  note=f"9:9:{w_ed}:{w_r}:1:1:1:{w_j}")

    t1m = float(np.mean([s[0] for s in held_scores]))
    t5m = float(np.mean([s[1] for s in held_scores]))
    print(f"\nLOO held-out mean (with joint member): top-1 {t1m:.4f}  "
          f"top-5 {t5m:.4f}")
    print("8-member LOO reference:                       top-1 0.2230  "
          "top-5 0.4828")
    log_score(experiment="loo-heldout-joint", subject="mean",
              embedding="megafusion+jointft", top1=t1m, top5=t5m,
              note="leave-one-dev-subject-out, 9-member grid incl w_joint")


if __name__ == "__main__":
    main()
