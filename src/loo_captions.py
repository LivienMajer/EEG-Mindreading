"""LOO test: does the BOLD caption-INFONCE member ADD to the 9-member megafusion?

E-T4-lin result (docs/text_anchor_captions.md): blending CLIP image +
BOLD caption text embeddings (same ViT-B/32, alpha=0.35) keeps clip-level
decodability (dev mean 10.78/33.82 vs clip 11.03/33.09) while
decorrelating from the clip member (top-1 agreement 0.57, rank corr
0.063). This module tests that blend space as the 10th member.

Protocol identical to src/loo_joint.py; the member list gains
infonce_{s}_emb_train__capblend035_s0.npy (subject's own trn -> blend
space, label-sorted dev rows) and the grid gains w_cap in {0, 0.5, 1, 2}.
If held-out beats the 9-member LOO (23.04/48.28), the blend member
earns a slot; if w_cap=0 keeps winning, it does not (failure-analysis
material for the report either way).

Usage:  python -m src.loo_captions [--data-dir <root>]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .log_score import log_score
from .loo_generalization import SUBS, fuse, topk

TST = "work/tst_ranked"
E = "emb_train__clip-vitb32"
CAP = "emb_train__capblend100"
DATA_ROOT = Path("provided_material/eeg-mind-reading-challenge/"
                 "eeg-mind-reading-challenge")


def members(s: str) -> list[np.ndarray]:
    others = [o for o in SUBS if o != s]
    m = [
        np.load(f"{TST}/infonce_{s}_{E}_s0.npy"),
        np.load(f"{TST}/infonce_{s}_emb_train__alexnet-f3_s0.npy"),
        np.load(f"{TST}/ed_{s}_xsubj_s0.npy"),
        np.load(f"{TST}/ridge_{s}_{E}_s0.npy"),
    ]
    m += [np.load(f"work/xsubj_probe/ranked_tr{o}_te{s}_{E}_s0.npy")
          for o in others]
    m += [np.load(f"{TST}/jointft_{s}_{E}_s0.npy")]
    m += [np.load(f"{TST}/infonce_{s}_{CAP}_s0.npy")]
    return m


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", type=Path, default=DATA_ROOT)
    args = p.parse_args()

    truths = {s: np.load(f"{TST}/truth_{s}_s0.npy") for s in SUBS}
    M = {s: members(s) for s in SUBS}

    grid = [(w_ed, w_r, w_j, w_c)
            for w_ed in (3, 5, 7)
            for w_r in (1, 2)
            for w_j in (0, 0.5, 1, 2, 3)
            for w_c in (0, 0.5, 1, 2)]

    print("LOO weight-selection: megafusion + jointft + caption-ridge "
          "[clip 9, alexnet 9, ed w, ridge r, transfers 1 1 1, joint w_j, "
          "caption w_c]")
    held_scores = []
    for held in SUBS:
        dev = [d for d in SUBS if d != held]
        best, best_key = None, -1.0
        for w_ed, w_r, w_j, w_c in grid:
            weights = [9, 9, w_ed, w_r, 1, 1, 1, w_j, w_c]
            t1s = [topk(fuse(M[d], weights), truths[d])[0] for d in dev]
            key = float(np.mean(t1s))
            if key > best_key:
                best, best_key = (w_ed, w_r, w_j, w_c), key
        w_ed, w_r, w_j, w_c = best
        weights = [9, 9, w_ed, w_r, 1, 1, 1, w_j, w_c]
        t1, t5 = topk(fuse(M[held], weights), truths[held])
        held_scores.append((t1, t5))
        print(f"  hold {held}: selected w_ed={w_ed} w_ridge={w_r} "
              f"w_joint={w_j} w_cap={w_c} (dev mean top1 {best_key:.4f}) "
              f"->  held top1 {t1:.4f}  top5 {t5:.4f}")
        log_score(experiment="loo-heldout-caption", subject=held,
                  embedding="megafusion+jointft+capridge", top1=t1, top5=t5,
                  note=f"9:9:{w_ed}:{w_r}:1:1:1:{w_j}:{w_c}")

    t1m = float(np.mean([s[0] for s in held_scores]))
    t5m = float(np.mean([s[1] for s in held_scores]))
    print(f"\nLOO held-out mean (with caption member): top-1 {t1m:.4f}  "
          f"top-5 {t5m:.4f}")
    print("9-member LOO reference:                      top-1 0.2304  "
          "top-5 0.4828")
    log_score(experiment="loo-heldout-caption", subject="mean",
              embedding="megafusion+jointft+capridge", top1=t1m, top5=t5m,
              note="leave-one-dev-subject-out, 10-member grid incl w_cap")


if __name__ == "__main__":
    main()
