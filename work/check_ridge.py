"""Check: does make_submission.ridge_member reproduce the dev ridge dump
bit-exactly? (sub-01; loads refs BEFORE the call, restores label-sorted files
afterwards -- ridge_member itself dumps original-order for 05/06-style use)"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.challenge_data import load_test_subject  # noqa: E402

ROOT = Path("provided_material/eeg-mind-reading-challenge/eeg-mind-reading-challenge")
import src.make_submission as ms  # noqa: E402

ms.root = ROOT

ref = np.load(f"work/tst_ranked/ridge_01_{ms.CLIP}_s0.npy")
truth = np.load("work/tst_ranked/truth_01_s0.npy")

ranked = ms.ridge_member("01", ms.CLIP)   # overwrites the dump in original order
tst = load_test_subject(ROOT, "01")
np.save(f"work/tst_ranked/ridge_01_{ms.CLIP}_s0.npy",
        ranked[np.argsort(tst.y)])        # restore label-sorted dev convention

mine_sorted = ranked[np.argsort(tst.y)]
same_0 = np.array_equal(mine_sorted, ref)

# also validate the row-order claim itself: label-sorted must equal the file
# we just restored, and truth must be unchanged
same_1 = np.array_equal(np.load(f"work/tst_ranked/ridge_01_{ms.CLIP}_s0.npy"),
                        ref)
assert np.array_equal(np.load("work/tst_ranked/truth_01_s0.npy"), truth)
print("ridge_member == bench dump (label-sorted):", same_0)
print("dev dump restored intact:", same_1)
assert same_0 and same_1
print("OK: ridge_member is bit-exact vs bench_embeddings")
