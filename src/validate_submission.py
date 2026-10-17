"""Validate submission files against the challenge format rules.

Checks (per submission file):
  1. loads as int array of shape (102, 102)
  2. every row is a permutation of video ids 1001..1102
  3. ROW ORDER sanity: rows must correspond 1:1 to tst.npy rows in ORIGINAL
     file order. tst.npy is a plain (102, 17, 1500) array -- its row order
     cannot be verified post-hoc from the file itself. For dev subjects with
     labels (tst_stim/) we compute top-1/top-5 of the submission: if rows
     were label-sorted (argsort) instead of kept in original order, accuracy
     collapses to chance (~1%/5%) and we FAIL loudly.

Usage:
  python -m src.validate_submission work\\sub-05_ranked_candidate_list.npy \
      provided_material\\eeg-mind-reading-challenge\\eeg-mind-reading-challenge 05
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .challenge_data import load_test_subject

VALID = np.arange(1001, 1103)


def check(path: Path, data_dir: Path, sub: str) -> bool:
    a = np.load(path)
    ok = True
    if not np.issubdtype(a.dtype, np.integer):
        print(f"FAIL dtype: {a.dtype} (must be int)")
        ok = False
    if a.shape != (102, 102):
        print(f"FAIL shape: {a.shape} (must be (102, 102))")
        return False
    bad = [i for i, row in enumerate(a)
           if set(row.tolist()) != set(VALID.tolist())]
    if bad:
        print(f"FAIL rows not a permutation of 1001..1102: {bad[:5]}")
        ok = False
    print(f"format checks: {'OK' if ok else 'FAILED'}")

    # Row-order sanity via labels (dev subjects only)
    tst = load_test_subject(data_dir, sub)
    if tst.y is not None:
        t1 = float((a[:, 0] == tst.y).mean())
        t5 = float((a[:, :5] == tst.y[:, None]).any(1).mean())
        verdict = ("OK (way above chance)" if t1 > 0.02 else
                   "SUSPECT -- row order may be shuffled/argsorted!")
        print(f"row-order sanity: top-1 {t1:.4f}  top-5 {t5:.4f}  [{verdict}]")
        if t1 <= 0.02:
            ok = False
    else:
        print("no labels (sub-05/06): row order NOT verifiable here. "
              "Ensure the generating pipeline ranked tst.X in file order "
              "(no np.argsort(tst.y) before saving).")
    print("ALL OK" if ok else "VALIDATION FAILED")
    return ok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("submission", type=Path)
    p.add_argument("data_dir", type=Path)
    p.add_argument("sub", help="subject id, e.g. 05")
    args = p.parse_args()
    check(args.submission, args.data_dir, args.sub)


if __name__ == "__main__":
    main()
