"""Write sub-05/06 submissions via the LOO-validated megafusion.

Members per held-out subject (all legal: only trn data is used for training;
tst is only ever ranked, never trained on):
  1. own InfoNCE (clip)    w=9  -- trained on the subject's own trn
  2. own InfoNCE (alexnet) w=9  -- idem
  3. E-D teammate encoder  w=5  -- frozen ckpt, joint-trained on all 6 subjects' trn
  4. own ridge (clip)      w=2  -- ridge video-features->EEG on the subject's trn
  5-8. transfers: the FOUR dev-subject InfoNCE models, w=1 each

Fusion weights 9:9:5:2:1:1:1:1. Dev LOO selected 9:9:5:2:1:1:1 (3 transfers,
LOO held-out 22.30/48.28); the 4th transfer gets the same unit weight.

Row order: ORIGINAL tst.npy recording order (no label sorting -- there are
no labels for sub-05/06). Every member here produces rows in that same
convention:
  - infonce 05/06 dumps: written by contrastive_pipeline no-labels path
  - ED rankings: work/ed_member/ranked_subXX.npy (original order)
  - ridge: computed here, no argsort(tst.y)
  - transfers: rank_candidates follows input order; no label sorting here

Usage:  python -m src.make_submission \
            --data-dir provided_material/eeg-mind-reading-challenge/eeg-mind-reading-challenge
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from .bench_embeddings import (corr_match, fit_forward, load_pooled,
                               predict_eeg, video_matrix_pooled)
from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject
from .contrastive_pipeline import build_target_space, eeg_window, rank_candidates
from .ensemble_rank import fuse_mean_rank

CLIP = "emb_train__clip-vitb32"
ALEX = "emb_train__alexnet-f3"
DEV = ["01", "02", "03", "04"]
# dev LOO selected 9:9:5:2:1:1:1 (3 transfers); the 4th transfer for the
# submission subjects gets the same unit weight -> 9:9:5:2:1:1:1:1.
WEIGHTS = [9, 9, 5, 2, 1, 1, 1, 1]

# ---------------------------------------------------------------------------
# members
# ---------------------------------------------------------------------------

def infonce_member(sub: str, emb_tag: str) -> np.ndarray:
    return np.load(f"work/tst_ranked/infonce_{sub}_{emb_tag}_s0.npy")


def transfer_member(te_sub: str, tr_sub: str, emb_tag: str) -> np.ndarray:
    """Rank te_sub's tst with the dev checkpoint trained on tr_sub's trn.

    Same target space + preprocessing as cross_subject_probe (the saved
    model is the probe's). Rows in ORIGINAL order (probe dumped label-sorted
    versions for dev -- we re-rank here without any sorting).
    """
    import torch

    from .eeg_encoder import build_eegnet

    ck_path = Path(f"work/xsubj_probe/model_sub{tr_sub}_{emb_tag}_s0.pt")
    if not ck_path.exists():
        raise SystemExit(f"missing checkpoint {ck_path} -- run cross_subject_probe first")
    ck = torch.load(ck_path, map_location="cpu", weights_only=False)
    model = build_eegnet(ck["n_channels"], ck["n_times"],
                         embed_dim=ck["embed_dim"], k1=16, k2=8)
    model.load_state_dict(ck["state_dict"])

    emb_trn = load_pooled(f"work/{emb_tag}.npz")
    emb_tst = load_pooled(f"work/{emb_tag.replace('train', 'test')}.npz")
    _, X_cand, _ = build_target_space(emb_trn, emb_tst, n_components=256)

    tst = load_test_subject(root, te_sub)
    E_tst = eeg_window(tst.X)
    ranked = rank_candidates(model, E_tst, X_cand)
    np.save(f"work/xsubj_probe/ranked_tr{tr_sub}_te{te_sub}_{emb_tag}_s0.npy",
            ranked)
    return ranked


def ridge_member(sub: str, emb_tag: str) -> np.ndarray:
    """Ridge forward model on the subject's own trn, rank its tst.

    Replicates bench_embeddings EXACTLY (decimate WITHOUT z-scoring, window
    (5,35), corr_match) so the member matches the dev ridge dumps -- except
    rows stay in ORIGINAL tst.npy order (no labels to sort by).
    """
    emb_trn = load_pooled(f"work/{emb_tag}.npz")
    emb_tst = load_pooled(f"work/{emb_tag.replace('train', 'test')}.npz")
    trn = load_train_subject(root, sub)
    E_trn = decimate(trn.X, 5, axis=-1, ftype="fir", zero_phase=True)
    E_trn = E_trn[:, :, 5:35]
    X_feat = video_matrix_pooled(emb_trn, trn.video_ids)
    bundle = fit_forward(X_feat, E_trn)

    tst = load_test_subject(root, sub)
    E_tst = decimate(tst.X, 5, axis=-1, ftype="fir", zero_phase=True)
    E_tst = E_tst[:, :, 5:35]
    X_cand = video_matrix_pooled(emb_tst, TEST_VIDEO_IDS)
    pred = predict_eeg(bundle, X_cand)          # (102, 17, t)
    S = corr_match(E_tst, pred)                 # (n_tst, n_cand)
    ranked = TEST_VIDEO_IDS[np.argsort(-S, axis=1)].astype(np.int64)
    np.save(f"work/tst_ranked/ridge_{sub}_{emb_tag}_s0.npy", ranked)
    return ranked

# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

root: Path = Path(".")


def main():
    global root
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--subjects", nargs="+", default=["05", "06"])
    args = ap.parse_args()
    root = Path(args.data_dir)

    out_dir = Path("submissions")
    out_dir.mkdir(exist_ok=True)

    for sub in args.subjects:
        members = [
            infonce_member(sub, CLIP),
            infonce_member(sub, ALEX),
            np.load(f"work/ed_member/ranked_sub{sub}.npy"),
            ridge_member(sub, CLIP),
        ]
        members += [transfer_member(sub, d, CLIP) for d in DEV]
        fused = fuse_mean_rank(members, WEIGHTS)
        path = out_dir / f"sub-{sub}_ranked_candidate_list.npy"
        np.save(path, fused.astype(np.int64))
        print(f"sub-{sub}: fused 8 members with weights {WEIGHTS} -> {path}")


if __name__ == "__main__":
    main()
