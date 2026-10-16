"""Benchmark any frame-embedding .npz against the dev subjects.

Same forward-model + correlation matching as the organizer baseline, but the
video space is swappable: any .npz with keys f"{video_id:05d}_f{k:02d}" from
src.extract_embeddings works. Frame embeddings are z-scored, PCA-reduced,
then a Ridge predicts the (windowed) EEG; ranking = correlation of measured
vs predicted EEG.

Also supports frame-level matching at eval time:
  - frame_agg='mean'   : average frame embeddings (organizer style)
  - frame_agg='max'    : per-frame EEG prediction, best frame wins
  - frame_agg='concat' : flatten frames (captures temporal structure)

Usage:
  python -m src.bench_embeddings --data-dir <challenge root> \
      --emb work/emb_test__clip-vitb32.npz --emb-train work/emb_train__clip-vitb32.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import decimate
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject

# from snr_probe study: 50-350 ms @ 100 Hz wins; 50-600 as organizer reference
WINDOWS = {"snr": (5, 35), "org": (5, 60)}
KEY_TEMPLATE = "{vid:05d}_f{k:02d}"


def video_matrix(emb, video_ids, agg="mean", n_frames=8):
    """(n_videos, D) or (n_videos, n_frames*D) for agg='concat'."""
    key = KEY_TEMPLATE.format
    rows = []
    missing = []
    for vid in video_ids:
        feats = [emb[key(vid=vid, k=k)] for k in range(n_frames)
                 if key(vid=vid, k=k) in emb]
        if not feats:
            missing.append(vid)
            continue
        if agg == "mean":
            rows.append(np.mean(feats, axis=0))
        elif agg == "concat":
            rows.append(np.concatenate(feats))
        else:
            raise ValueError(agg)
    if missing:
        raise KeyError(f"{len(missing)} videos missing in embeddings, "
                      f"e.g. {missing[:5]} -- extract them first")
    return np.stack(rows).astype(np.float32)


def fit_forward(X_feat, X_eeg_win):
    """Ridge forward model video-features -> windowed EEG (organizer style).

    X_eeg_win: (n_trials, 17, t_win) -- already windowed, flattened as target.
    """
    sc = StandardScaler().fit(X_feat)
    X_z = sc.transform(X_feat)
    # n_samples (<=~5800) << n_features (up to 140k+): covariance_eigh is
    # exact, fast and memory-safe here; svd_solver='full' would be O(n_feat^3).
    pca = PCA(n_components=0.95, svd_solver="covariance_eigh").fit(X_z)
    X_p = pca.transform(X_z)
    sc2 = StandardScaler().fit(X_p)
    model = RidgeCV(alphas=np.logspace(-3, 6, 20))
    model.fit(sc2.transform(X_p), X_eeg_win.reshape(len(X_eeg_win), -1))
    return {"sc": sc, "pca": pca, "sc2": sc2, "model": model}


def predict_eeg(bundle, X_feat):
    X = bundle["sc"].transform(X_feat)
    X = bundle["pca"].transform(X)
    X = bundle["sc2"].transform(X)
    pred = bundle["model"].predict(X)
    return pred.reshape(len(X_feat), 17, -1)


def corr_match(M, P):
    """(n_measured, n_predicted) Pearson correlation matrix.

    M and P: (n, 17, t) arrays with the SAME temporal window already applied.
    """
    m = M.reshape(len(M), -1)
    p = P.reshape(len(P), -1)
    mz = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-8)
    pz = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-8)
    return mz @ pz.T / m.shape[1]


def topk(ranked, truth, k):
    return float((ranked[:, :k] == truth[:, None]).any(axis=1).mean()
                 if k > 1 else (ranked[:, 0] == truth).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--emb-train", required=True, help="npz with train-video embeddings")
    p.add_argument("--emb-test", required=True, help="npz with test-video embeddings")
    p.add_argument("--agg", default="mean", choices=["mean", "concat"])
    p.add_argument("--window", default="snr", choices=list(WINDOWS))
    p.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04"])
    args = p.parse_args()

    root = Path(args.data_dir)
    emb_trn = dict(np.load(args.emb_train))
    emb_tst = dict(np.load(args.emb_test))
    t_win = WINDOWS[args.window]

    # candidate matrix is shared across subjects: rows = TEST_VIDEO_IDS sorted
    X_cand = video_matrix(emb_tst, TEST_VIDEO_IDS, agg=args.agg)

    results = {}
    for sub in args.subjects:
        trn = load_train_subject(root, sub)
        # EEG: decimate, take same window (precomputed on 100 Hz grid)
        E = decimate(trn.X, 5, axis=-1, ftype="fir", zero_phase=True)
        E_full = E[:, :, t_win[0]:t_win[1]]  # windowed EEG target
        X_trn = video_matrix(emb_trn, trn.video_ids, agg=args.agg)

        bundle = fit_forward(X_trn, E_full)
        pred_cand = predict_eeg(bundle, X_cand)  # (102, 17, t_win)

        tst = load_test_subject(root, sub)
        E_tst = decimate(tst.X, 5, axis=-1, ftype="fir", zero_phase=True)
        E_tst_win = E_tst[:, :, t_win[0]:t_win[1]]  # same window as training

        order = np.argsort(tst.y)  # eval: sorted by gt id
        S = corr_match(E_tst_win[order], pred_cand)
        ranked = TEST_VIDEO_IDS[np.argsort(-S, axis=1)]
        truth = tst.y[order]
        results[sub] = (topk(ranked, truth, 1), topk(ranked, truth, 5))

    print(f"\n=== {Path(args.emb_test).name} | agg={args.agg} | window={args.window} ===")
    for sub, (t1, t5) in results.items():
        print(f"  sub-{sub}: top-1 {t1:.4f}  top-5 {t5:.4f}   (chance 0.0098/0.049)")
    t1s = [v[0] for v in results.values()]
    t5s = [v[1] for v in results.values()]
    print(f"  MEAN : top-1 {np.mean(t1s):.4f}  top-5 {np.mean(t5s):.4f}")
    return results


if __name__ == "__main__":
    main()
