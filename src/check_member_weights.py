"""Check that the saved member weights (src/member_weights.py) reproduce the fusion members.

For every checkpoint in models/fusion/: reload it, rank the subject's test recordings from
the raw tst EEG, and compare with the member's ranking file (dev subjects: label-sorted rows,
as written by the member scripts; sub-05/06: original order). Uses no labels except to put
the dev rows in the same order as the files being compared.

    python -m src.check_member_weights --data-dir <challenge root>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from .bench_embeddings import corr_match, load_pooled, predict_eeg, video_matrix_pooled
from .challenge_data import TEST_VIDEO_IDS, load_test_subject
from .contrastive_pipeline import eeg_window, rank_candidates
from .member_weights import WEIGHTS_DIR, load_infonce
from .prep_variants import prep_split, whiten


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    args = p.parse_args()
    import joblib

    tst_cache, n_ok, bad = {}, 0, []

    def tst(sub):
        if sub not in tst_cache:
            tst_cache[sub] = load_test_subject(args.data_dir, sub)
        return tst_cache[sub]

    for path in sorted((WEIGHTS_DIR / "infonce").glob("*.pt")):
        name = path.stem                     # = the ranking file name
        sub = name.split("_")[1] if name.startswith("infonce_") else name.split("_")[-2]
        ref = Path("work/tst_ranked" if name.startswith("infonce_") else "work/prep_variants") / f"{name}.npy"
        model, ck = load_infonce(path)
        d = tst(sub)
        if ck["prep"] is None:
            E = eeg_window(d.X)
        else:
            cfg = {k: ck["prep"][k] for k in ("lowpass", "remove_mean", "align")}
            E = prep_split(d.X, **cfg)
            if cfg["align"] == "train":
                E = whiten(E, ck["prep"]["R"])
        ranked = rank_candidates(model, E, ck["X_cand"])
        if d.y is not None:
            ranked = ranked[np.argsort(d.y)]
        same = (ranked == np.load(ref)).all()
        n_ok += same
        if not same:
            bad.append(name)
        print(f"{name}: {'reproduces' if same else 'DIFFERS from'} {ref}")

    for path in sorted((WEIGHTS_DIR / "ridge").glob("*.joblib")):
        b = joblib.load(path)
        sub = path.stem.split("_")[1]
        ref = Path("work/tst_ranked") / f"{path.stem}_s0.npy"
        d = tst(sub)
        w0, w1 = b["window"]
        E = decimate(d.X, 5, axis=-1, ftype="fir", zero_phase=True)[:, :, w0:w1]
        if d.y is not None:
            E = E[np.argsort(d.y)]
        emb_tst = load_pooled(f"work/{b['emb_tag'].replace('train', 'test')}.npz")
        pred = predict_eeg(b, video_matrix_pooled(emb_tst, TEST_VIDEO_IDS))
        ranked = TEST_VIDEO_IDS[np.argsort(-corr_match(E, pred), axis=1)]
        same = (ranked == np.load(ref)).all()
        n_ok += same
        if not same:
            bad.append(path.stem)
        print(f"{path.stem}: {'reproduces' if same else 'DIFFERS from'} {ref}")

    print(f"\n{n_ok} checkpoints reproduce their member exactly" + (f"; differ: {bad}" if bad else ""))
    if bad:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
