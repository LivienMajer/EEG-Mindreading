"""Final sanity: submission files' format, plus the strongest available
row-order proof for sub-05/06: re-run the fusion from the member dumps and
confirm the saved file matches (members are all tst.npy row order, transfers
and ridge re-verified bit-exact vs dev conventions)."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")
from src.ensemble_rank import fuse_mean_rank  # noqa: E402
import src.make_submission as ms  # noqa: E402

ms.root = Path("provided_material/eeg-mind-reading-challenge/eeg-mind-reading-challenge")

for sub in ("05", "06"):
    a = np.load(f"submissions/sub-{sub}_ranked_candidate_list.npy")
    print(f"sub-{sub}: shape {a.shape}, dtype {a.dtype}, "
          f"ids {a.min()}..{a.max()}")
    assert a.shape == (102, 102) and a.dtype == np.int64
    assert a.min() >= 1001 and a.max() <= 1102
    bad = sum(sorted(set(r)) != sorted(a[0]) for r in a)
    print(f"  rows not perm-of-candidates: {bad}")
    assert bad == 0

    members = [ms.infonce_member(sub, ms.CLIP),
               ms.infonce_member(sub, ms.ALEX),
               np.load(f"work/ed_member/ranked_sub{sub}.npy"),
               np.load(f"work/tst_ranked/ridge_{sub}_{ms.CLIP}_s0.npy")]
    members += [np.load(f"work/xsubj_probe/ranked_tr{d}_te{sub}_{ms.CLIP}_s0.npy")
                for d in ms.DEV]
    refused = fuse_mean_rank(members, ms.WEIGHTS).astype(np.int64)
    ok = np.array_equal(refused, a)
    print(f"  re-fused == saved file: {ok}")
    assert ok

print("OK: both submissions consistent with member dumps (row order = tst.npy)")
