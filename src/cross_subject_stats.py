"""Quick facts about cross-subject video overlap in the training splits.

Answers: how many EEG observations does the average train video have across
subjects? Prints per-subject trial counts and pairwise overlap with sub-01,
plus the union statistics that motivated docs/cross_subject_idea.md.

Usage:
  python -m src.cross_subject_stats provided_material\\eeg-mind-reading-challenge\\eeg-mind-reading-challenge
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .challenge_data import N_CANDIDATES, load_train_subject


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", type=Path)
    p.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04", "05", "06"])
    args = p.parse_args()

    per_subject_sets = {}
    total = 0
    for sub in args.subjects:
        t = load_train_subject(args.data_dir, sub)
        vids = set(t.video_ids.tolist())
        per_subject_sets[sub] = vids
        total += len(vids)
        ref = per_subject_sets[args.subjects[0]]
        print(f"sub-{sub}: {len(vids)} trn trials (unique videos {len(vids)}), "
              f"overlap with sub-{args.subjects[0]}: {len(vids & ref)}")

    union = set().union(*per_subject_sets.values())
    counts = np.zeros(len(union), dtype=int)
    idx = {v: i for i, v in enumerate(sorted(union))}
    for vids in per_subject_sets.values():
        for v in vids:
            counts[idx[v]] += 1
    print(f"\nunion of train videos across subjects: {len(union)} "
          f"(vs {N_CANDIDATES} test candidates)")
    print(f"EEG observations per train video: mean {counts.mean():.2f}, "
          f"min {counts.min()}, max {counts.max()}")
    print(f"videos with >= 2 subject observations: "
          f"{(counts >= 2).mean():.1%}  -> joint training pool "
          f"~{total} trials (vs ~950 per subject)")


if __name__ == "__main__":
    main()
