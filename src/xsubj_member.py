"""Turn a src.xsubj.submit output folder into megafusion member dumps.

src.xsubj.submit writes sub-XX_ranked_candidate_list.npy for sub-01..06 in ORIGINAL
tst.npy row order. The fusion scripts (loo_new_members, member_check) expect dev dumps
LABEL-SORTED (np.argsort(tst.y)) as work/prep_variants/A<tag>_XX_s0.npy; sub-05/06 stay
in original order (submission convention, as for the other members).

Usage:
  python -m src.xsubj.submit --config configs/xsubj_fc6.yaml --out work/xsubj_fc6
  python -m src.xsubj_member --src work/xsubj_fc6 --tag _dfc6 --data-dir <root>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .challenge_data import load_test_subject
from .prep_variants import OUT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="folder written by src.xsubj.submit")
    p.add_argument("--tag", required=True, help="member tag, e.g. _dfc6")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--subs", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    args = p.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for s in args.subs:
        ranked = np.load(Path(args.src) / f"sub-{s}_ranked_candidate_list.npy")
        y = load_test_subject(args.data_dir, s).y
        if y is not None:
            ranked = ranked[np.argsort(y)]
            t1 = float((ranked[:, 0] == y[np.argsort(y)]).mean())
            note = f"label-sorted, top-1 {t1:.4f}"
        else:
            note = "original order (no labels)"
        np.save(OUT / f"A{args.tag}_{s}_s0.npy", ranked.astype(np.int64))
        print(f"sub-{s}: {note} -> {OUT / f'A{args.tag}_{s}_s0.npy'}")


if __name__ == "__main__":
    main()
