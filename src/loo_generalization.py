"""Leave-one-subject-out generalization test for the megafusion weights.

Motivation: the megafusion weights (clip 9 : alexnet 9 : E-D w : ridge r :
3 transfers 1:1:1) were selected ON the dev subjects we measure — an oracle
estimate. The final submission scores come from subjects that had no say in
the weight choice. This script quantifies the selection bias:

  For each held-out dev subject s:
    - weight grid search using ONLY the other 3 dev subjects (their fused
      top-1, mean over the 3)
    - apply the winning weights to s, record the score.

The member set mirrors the sub-05/06 submission situation exactly:
  own infonce clip + own infonce alexnet + own ridge (trained on the
  subject's own trn — legal for 05/06 too), the teammate cross-subject
  encoder (E-D; trained on all subjects' trn, as it will be for 05/06),
  and transfer rankings from the other subjects' models.

Usage:  python -m src.loo_generalization
        python -m src.loo_generalization --ed-dir work/ed_5e3154e   # other E-D checkpoint
"""

from __future__ import annotations

import numpy as np

from .log_score import log_score

SUBS = ["01", "02", "03", "04"]
E = "emb_train__clip-vitb32_s0"
TST = "work/tst_ranked"
PROBE = "work/xsubj_probe"

# member files per subject, fixed order for the weight vector:
# [clip 9, alexnet 9, ed w, ridge r, transfer, transfer, transfer]
ED_DIR = TST


def members(s: str) -> list[np.ndarray]:
    others = [o for o in SUBS if o != s]
    m = [
        np.load(f"{TST}/infonce_{s}_emb_train__clip-vitb32_s0.npy"),
        np.load(f"{TST}/infonce_{s}_emb_train__alexnet-f3_s0.npy"),
        np.load(f"{ED_DIR}/ed_{s}_xsubj_s0.npy"),
        np.load(f"{TST}/ridge_{s}_{E}.npy"),
    ]
    m += [np.load(f"{PROBE}/ranked_tr{o}_te{s}_{E}.npy") for o in others]
    return m


def fuse(rankings: list[np.ndarray], weights: list[float]) -> np.ndarray:
    """Vectorized weighted mean-rank fusion (same math as fuse_mean_rank)."""
    cand = np.sort(rankings[0][0])
    n, k = rankings[0].shape
    w = np.asarray(weights, float)
    w = w / w.sum()
    avg = np.zeros((n, len(cand)))
    for r, wi in zip(rankings, w):
        cols = np.searchsorted(cand, r)           # (n, k) candidate indices
        R = np.empty((n, len(cand)))
        R[np.arange(n)[:, None], cols] = np.arange(k)
        avg += wi * R
    return cand[np.argsort(avg, axis=1)]


def topk(ranked: np.ndarray, y: np.ndarray):
    return (float((ranked[:, 0] == y).mean()),
            float((ranked[:, :5] == y[:, None]).any(1).mean()))


def main():
    import argparse
    global ED_DIR
    ap = argparse.ArgumentParser()
    ap.add_argument("--ed-dir", default=TST, help="folder with ed_XX_xsubj_s0.npy")
    ED_DIR = ap.parse_args().ed_dir
    truths = {s: np.load(f"{TST}/truth_{s}_s0.npy") for s in SUBS}
    M = {s: members(s) for s in SUBS}

    grid = [(w_ed, w_r)
            for w_ed in (1, 3, 5, 7, 9)
            for w_r in (1, 2)]

    print("LOO weight-selection for megafusion [clip 9, alexnet 9, ed w, "
          "ridge r, transfers 1 1 1]")
    held_scores = []
    for held in SUBS:
        dev = [d for d in SUBS if d != held]
        best, best_key = None, (-1.0, -1.0)
        for w_ed, w_r in grid:
            weights = [9, 9, w_ed, w_r, 1, 1, 1]
            t1s = [topk(fuse(M[d], weights), truths[d])[0] for d in dev]
            key = (float(np.mean(t1s)),)
            if key > best_key[:1]:
                best, best_key = (w_ed, w_r), (float(np.mean(t1s)),)
        w_ed, w_r = best
        weights = [9, 9, w_ed, w_r, 1, 1, 1]
        t1, t5 = topk(fuse(M[held], weights), truths[held])
        held_scores.append((t1, t5))
        print(f"  hold {held}: selected w_ed={w_ed} w_ridge={w_r} (dev mean "
              f"top1 {best_key[0]:.4f})  ->  held top1 {t1:.4f}  top5 {t5:.4f}")
        log_score(experiment="loo-heldout", subject=held,
                   embedding="megafusion", top1=t1, top5=t5,
                   note=f"weights picked on other 3 dev subjects: "
                        f"9:9:{w_ed}:{w_r}:1:1:1")

    t1m = float(np.mean([s[0] for s in held_scores]))
    t5m = float(np.mean([s[1] for s in held_scores]))
    print(f"\nLOO held-out mean: top-1 {t1m:.4f}  top-5 {t5m:.4f}")
    print("oracle (weights chosen on all 4):  top-1 0.2157  top-5 0.4902")
    log_score(experiment="loo-heldout", subject="mean", embedding="megafusion",
              top1=t1m, top5=t5m,
              note="leave-one-dev-subject-out; weights per fold from other 3")


if __name__ == "__main__":
    main()
