"""Proper check: load the reference dump BEFORE running transfer_member, then
verify bit-exactness. Also restores the tr01_te02 dev dump that an earlier
buggy check overwrote (label-sorted = original_order[argsort(y)] -- same
deterministic checkpoint + rank_candidates, so this is a faithful restore)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.challenge_data import load_test_subject  # noqa: E402

ROOT = Path("provided_material/eeg-mind-reading-challenge/eeg-mind-reading-challenge")
import src.make_submission as ms  # noqa: E402

ms.root = ROOT

# 1) restore tr01_te02 dev dump (currently original order from the same
#    checkpoint run; probe convention was ranked[argsort(y)])
tst02 = load_test_subject(ROOT, "02")
cur = np.load(f"work/xsubj_probe/ranked_tr01_te02_{ms.CLIP}_s0.npy")
restored = cur[np.argsort(tst02.y)]
np.save(f"work/xsubj_probe/ranked_tr01_te02_{ms.CLIP}_s0.npy", restored)
print("restored ranked_tr01_te02 to label-sorted convention")

# 2) bit-exactness on a PAIR WHOSE FILE WE DID NOT TOUCH: load ref FIRST
ref = np.load(f"work/xsubj_probe/ranked_tr01_te03_{ms.CLIP}_s0.npy")
ranked = ms.transfer_member("03", "01", ms.CLIP)   # overwrites te03 file, ref already in memory
# after this call the file is original-order; restore dev dump as well
tst03 = load_test_subject(ROOT, "03")
np.save(f"work/xsubj_probe/ranked_tr01_te03_{ms.CLIP}_s0.npy",
        ranked[np.argsort(tst03.y)])

mine_sorted = ranked[np.argsort(tst03.y)]
same = np.array_equal(mine_sorted, ref)
print("checkpoint reload == probe dump (tr01->te03):", same)
assert same, "MISMATCH"
print("OK: transfer_member machinery is bit-exact; te03 file restored")
