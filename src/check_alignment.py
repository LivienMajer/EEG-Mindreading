"""Row-order sanity check for unlabeled members (sub-05/06), where no score is possible.

Two members built independently but from the same recordings agree far more than chance
on which videos they rank top-5 for each recording. A member whose rows are in another
order (e.g. label-sorted instead of tst.npy order) agrees only at chance level, i.e. like
the same pair compared with its rows shifted. Each member is compared with references
from the original megafusion.

Usage:
  python -m src.check_alignment --subs 05 06 --members work/prep_variants/A_fc6_{s}_s0.npy \
      work/ed_member_5e3154e/ranked_sub{s}.npy
"""

from __future__ import annotations

import argparse

import numpy as np

REFS = {"InfoNCE CLIP": "work/tst_ranked/infonce_{s}_emb_train__clip-vitb32_s0.npy",
        "E-D (86fac43)": "work/ed_member/ranked_sub{s}.npy"}


def top5_overlap(a: np.ndarray, b: np.ndarray, shift: int = 0) -> float:
    n = len(a)
    return float(np.mean([len(set(a[i, :5]) & set(b[(i + shift) % n, :5])) > 0 for i in range(n)]))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--subs", nargs="+", default=["05", "06"])
    p.add_argument("--members", nargs="+", required=True, help="paths with {s} for the subject")
    p.add_argument("--min-margin", type=float, default=0.15,
                   help="fail unless the overlap beats the shifted baseline by this much "
                        "for at least one reference")
    args = p.parse_args()
    bad = []
    for s in args.subs:
        refs = {n: np.load(path.format(s=s)) for n, path in REFS.items()}
        print(f"sub-{s}: recordings whose top-5 lists overlap (aligned / rows shifted by 37)")
        for path in args.members:
            a = np.load(path.format(s=s))
            cells, margins = [], []
            for n, b in refs.items():
                ov, sh = top5_overlap(a, b), top5_overlap(a, b, 37)
                cells.append(f"{n}: {ov * 100:3.0f}% / {sh * 100:3.0f}%")
                margins.append(ov - sh)
            # aligned members clearly agree with at least one reference; different model
            # families can agree only moderately, a misaligned member agrees with none
            if max(margins) < args.min_margin:
                bad.append((s, path))
            print(f"  {path.format(s=s):55s} " + " | ".join(cells))
    if bad:
        raise SystemExit(f"possible row-order problem: {bad}")
    print("all members aligned with the references")


if __name__ == "__main__":
    main()
