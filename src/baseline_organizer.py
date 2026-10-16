"""Reproduction of the organizer baseline, as a clean script instead of notebooks.

Pipeline (from 01_preprocess_and_train_model.ipynb / 02_predict_movies.ipynb):
  1. Decimate EEG 500 -> 100 Hz (FIR, zero-phase)
  2. Extract N_FRAMES=8 frames per video, embed with AlexNet (net2brain)
     at 'features.3', average over frames  -> (n_videos, 139968)
  3. StandardScaler -> PCA(0.95) -> StandardScaler on image features
  4. Forward model: RidgeCV(image features -> flattened EEG)
     (yes FORWARD: video features PREDICT EEG; matching is correlation
      between predicted and measured EEG)
  5. Matching: z-score both, correlation in the 50-600 ms window,
     rank videos by correlation

This requires: pip install net2brain opencv-python joblib

Usage:
  python -m src.baseline_organizer --data-dir <challenge root> --sub 01
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

DEC, SFREQ, N_FRAMES = 5, 500, 8
LAYER = "features.3"
# Empirical d' probe across all 9 subject pairs: 50-350 ms beats the
# organizer's 50-600 ms window by ~30% relative (cross-subject identifiability).
# TODO: validate both windows in the full pipeline before final submission.
T_WIN = (5, 35)  # samples at 100 Hz = 50-350 ms post-onset


def decimate_eeg(X: np.ndarray) -> np.ndarray:
    return decimate(X, DEC, axis=-1, ftype="fir", zero_phase=True).astype(np.float32)


def extract_frame_features(video_dir: str | Path, video_ids: np.ndarray,
                            frames_dir: str | Path, feat_dir: str | Path) -> np.ndarray:
    """Dump N_FRAMES per video, then extract AlexNet features via net2brain.

    Returns (n_videos, D) with features averaged over frames.
    """
    import cv2

    video_dir, frames_dir, feat_dir = (Path(p) for p in
                                       (video_dir, frames_dir, feat_dir))
    frames_dir.mkdir(parents=True, exist_ok=True)

    vid_paths = {int(p.stem): p for p in video_dir.glob("*.mp4")}
    missing = [v for v in video_ids if v not in vid_paths]
    if missing:
        raise FileNotFoundError(f"{len(missing)} videos not found, e.g. {missing[:3]}")

    for sid in video_ids:
        cap = cv2.VideoCapture(str(vid_paths[int(sid)]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        picks = np.linspace(0, max(total - 1, 0), N_FRAMES).astype(int)
        for k, fno in enumerate(picks):
            path = frames_dir / f"{int(sid):05d}_f{k:02d}.jpg"
            if path.exists():
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(fno))
            ok, frame = cap.read()
            if ok:
                cv2.imwrite(str(path), frame)
        cap.release()

    from net2brain.feature_extraction import FeatureExtractor
    fx = FeatureExtractor(model="AlexNet", netset="Standard", device="cpu")
    fx.extract(data_path=str(frames_dir), layers_to_extract=[LAYER],
               save_path=str(feat_dir))

    z = np.load(feat_dir / f"consolidated_{LAYER}.npz")
    rows = []
    for sid in video_ids:
        keys = [k for k in z.files if k.startswith(f"{int(sid):05d}_")]
        assert keys, f"no features for stimulus {sid}"
        rows.append(np.mean([z[k].flatten() for k in keys], axis=0))
    return np.stack(rows).astype(np.float32)


def fit_forward_model(X_feat: np.ndarray, X_eeg_100: np.ndarray):
    """Video features -> EEG regression. Returns everything needed for predict."""
    sc = StandardScaler().fit(X_feat)
    X_z = sc.transform(X_feat)
    pca = PCA(n_components=0.95, svd_solver="full").fit(X_z)
    X_p = pca.transform(X_z)
    sc2 = StandardScaler().fit(X_p)
    X_s = sc2.transform(X_p)
    model = RidgeCV(alphas=np.logspace(-3, 6, 20))
    model.fit(X_s, X_eeg_100.reshape(len(X_eeg_100), -1))
    return {"sc": sc, "pca": pca, "sc2": sc2, "model": model}


def predict_eeg(bundle, X_feat: np.ndarray) -> np.ndarray:
    X_s = bundle["sc2"].transform(bundle["pca"].transform(bundle["sc"].transform(X_feat)))
    pred = bundle["model"].predict(X_s)
    n = len(X_feat)
    return pred.reshape(n, -1, 300)  # (n, 17, 300) at 100 Hz


def similarity(M: np.ndarray, P: np.ndarray, t_win=T_WIN) -> np.ndarray:
    """Correlation between measured EEG rows and predicted-EEG columns.

    M: (n_a, 17, 300) measured; P: (n_b, 17, 300) predicted.
    Returns (n_a, n_b) z-scored correlation over the time window.
    """
    lo, hi = t_win
    m = M[:, :, lo:hi].reshape(len(M), -1)
    p = P[:, :, lo:hi].reshape(len(P), -1)
    mz = (m - m.mean(1, keepdims=True)) / (m.std(1, keepdims=True) + 1e-8)
    pz = (p - p.mean(1, keepdims=True)) / (p.std(1, keepdims=True) + 1e-8)
    return mz @ pz.T / m.shape[1]


def rank_videos(S: np.ndarray, candidate_ids=TEST_VIDEO_IDS) -> np.ndarray:
    """(n, n_cand) correlation -> (n, 102) ranked video IDs, best first."""
    order = np.argsort(-S, axis=1)
    return candidate_ids[order]


def topk_accuracy(ranked: np.ndarray, truth_ids: np.ndarray, k: int = 1) -> float:
    return float((ranked[:, :k] == truth_ids[:, None]).any(axis=1).mean()
                 if k > 1 else (ranked[:, 0] == truth_ids).mean())


def main():
    import joblib

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="challenge root (with trn/, tst/, stimuli_*)")
    p.add_argument("--sub", required=True, help="subject number, e.g. 01")
    p.add_argument("--work", default="work", help="cache dir for frames/features")
    p.add_argument("--skip-train", action="store_true", help="only score an existing model")
    args = p.parse_args()

    root, work = Path(args.data_dir), Path(args.work)
    sub = f"{int(args.sub):02d}"

    trn = load_train_subject(root, sub)
    X_eeg = decimate_eeg(trn.X)

    if not args.skip_train and not (work / f"encoder_sub-{sub}.joblib").exists():
        X_feat = extract_frame_features(
            root / "stimuli_train", trn.video_ids,
            work / "frames_train", work / "features_train")
        bundle = fit_forward_model(X_feat, X_eeg)
        joblib.dump(bundle, work / f"encoder_sub-{sub}.joblib", compress=3)
    bundle = joblib.load(work / f"encoder_sub-{sub}.joblib")

    tst = load_test_subject(root, sub)
    X_tst_eeg = decimate_eeg(tst.X)
    X_tst_feat = extract_frame_features(
        root / "stimuli_test", TEST_VIDEO_IDS,
        work / "frames_test", work / "features_test")

    pred = predict_eeg(bundle, X_tst_feat)  # rows follow sorted candidate order

    if tst.y is not None:
        # evaluation: rows sorted by ground truth id (as in organizer notebook)
        order = np.argsort(tst.y)
        S = similarity(X_tst_eeg[order], pred)  # pred rows already id-sorted
        ranked = rank_videos(S)
        truth = tst.y[order]
        print(f"sub-{sub}  top-1 {topk_accuracy(ranked, truth):.4f} "
              f"(chance {1 / 102:.4f}) | top-5 {topk_accuracy(ranked, truth, 5):.4f} "
              f"(chance {5 / 102:.4f})")
    else:
        # submission: rows must stay in ORIGINAL tst.npy order
        S = similarity(X_tst_eeg, pred)
        ranked = rank_videos(S)
        out = work / f"sub-{sub}_ranked_candidate_list.npy"
        np.save(out, ranked.astype(int))
        print(f"saved {out} with shape {ranked.shape}")


if __name__ == "__main__":
    main()
