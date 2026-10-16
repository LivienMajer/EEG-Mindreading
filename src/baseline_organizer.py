"""Organizer baseline as a script, with swappable video features.

Pipeline (from 01_preprocess_and_train_model.ipynb / 02_predict_movies.ipynb):
  1. Decimate EEG 500 -> 100 Hz (FIR, zero-phase)
  2. Video features, one vector per video. Feature sets (--features):
       alexnet   organizer original: 8 frames -> AlexNet 'features.3'
                 (net2brain), averaged -> (n_videos, 139968)
       clip      OpenCLIP ViT-B-32, mean over 8 frames  (ids + emb .npz)
       lowlevel  thumbnail / colour / contrast / edges / motion (ids + emb .npz)
     Default: clip + lowlevel.
  3. Per feature set: StandardScaler -> PCA(0.95) -> StandardScaler, then
     the sets are concatenated
  4. Forward model: RidgeCV(video features -> EEG in the matching window)
     (yes FORWARD: video features PREDICT EEG; matching is correlation
      between predicted and measured EEG)
  5. Matching: z-score both, correlation in the 50-600 ms window,
     rank videos by correlation

Dev result (sub-01..04, 102 candidates, mean): clip + lowlevel top-1 0.056,
top-5 0.157, mean rank 28.9 vs clip alone 0.034 / 0.172 / 30.6 and the
organizer's AlexNet 0.039 / 0.118 (sub-01 only). 50-350 ms was worse than
50-600 ms for every feature set (e.g. clip + lowlevel mean rank 30.8).

The clip / lowlevel .npz files come from
  python -m src.xsubj.video.clip_features --stim <root>/stimuli_train <root>/stimuli_test \
      --out models/xsubj/features/clip_vitb32
  python -m src.xsubj.video.lowlevel --stim <root>/stimuli_train <root>/stimuli_test \
      --out models/xsubj/features/lowlevel.npz
alexnet additionally requires: pip install net2brain

Usage:
  python -m src.baseline_organizer --data-dir <challenge root> --sub 01 02 03 04 05 06
  python -m src.baseline_organizer --data-dir <challenge root> --sub 01 --features alexnet
Labeled subjects (01-04) are scored; unlabeled ones (05/06) get a submission file.
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
# The snr_probe d' favoured 50-350 ms, but in the full forward-model pipeline
# 50-600 ms ranks better for every feature set (see module docstring).
T_WIN = (5, 60)  # samples at 100 Hz = 50-600 ms post-onset
FEATURE_FILES = {"clip": "models/xsubj/features/clip_vitb32.npz",
                 "lowlevel": "models/xsubj/features/lowlevel.npz"}


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


def load_npz_features(path: str | Path, video_ids: np.ndarray) -> np.ndarray:
    """(n_videos, D) rows of an ids + emb .npz, in video_ids order."""
    z = np.load(path, allow_pickle=False)
    index = {int(i): k for k, i in enumerate(z["ids"])}
    missing = [v for v in video_ids if int(v) not in index]
    if missing:
        raise KeyError(f"{len(missing)} videos missing in {path}, e.g. {missing[:3]}")
    return z["emb"][[index[int(v)] for v in video_ids]].astype(np.float32)


def video_features(names: list[str], root: Path, work: Path, video_ids: np.ndarray,
                   split: str) -> list[np.ndarray]:
    """One (n_videos, D) block per feature set; split is 'train' or 'test'."""
    blocks = []
    for name in names:
        if name == "alexnet":
            blocks.append(extract_frame_features(
                root / f"stimuli_{split}", video_ids,
                work / f"frames_{split}", work / f"features_{split}"))
        else:
            blocks.append(load_npz_features(FEATURE_FILES[name], video_ids))
    return blocks


def fit_forward_model(X_feats: list[np.ndarray], X_eeg_win: np.ndarray):
    """Video features -> windowed EEG regression. Returns everything needed for predict.

    X_feats: one block per feature set, each reduced by its own
    scaler -> PCA -> scaler so a large set (AlexNet) can't drown a small one.
    X_eeg_win: (n_trials, 17, t_win) EEG already cut to the matching window.
    """
    chains, reduced = [], []
    for X in X_feats:
        sc = StandardScaler().fit(X)
        pca = PCA(n_components=0.95, svd_solver="full").fit(sc.transform(X))
        sc2 = StandardScaler().fit(pca.transform(sc.transform(X)))
        chains.append((sc, pca, sc2))
        reduced.append(sc2.transform(pca.transform(sc.transform(X))))
    model = RidgeCV(alphas=np.logspace(-3, 6, 20))
    model.fit(np.hstack(reduced), X_eeg_win.reshape(len(X_eeg_win), -1))
    return {"chains": chains, "model": model, "t_win": X_eeg_win.shape[-1]}


def predict_eeg(bundle, X_feats: list[np.ndarray]) -> np.ndarray:
    X_s = np.hstack([sc2.transform(pca.transform(sc.transform(X)))
                     for (sc, pca, sc2), X in zip(bundle["chains"], X_feats)])
    pred = bundle["model"].predict(X_s)
    return pred.reshape(len(X_s), -1, bundle["t_win"])  # (n, 17, t_win) at 100 Hz


def similarity(M: np.ndarray, P: np.ndarray) -> np.ndarray:
    """Correlation between measured EEG rows and predicted-EEG columns.

    M: (n_a, 17, t) measured; P: (n_b, 17, t) predicted, same window.
    Returns (n_a, n_b) z-scored correlation over the time window.
    """
    m = M.reshape(len(M), -1)
    p = P.reshape(len(P), -1)
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


def check_submission(ranked: np.ndarray) -> None:
    assert ranked.shape == (102, 102) and ranked.dtype.kind == "i", (ranked.shape, ranked.dtype)
    assert (np.sort(ranked, axis=1) == TEST_VIDEO_IDS).all(), \
        "every row must be a permutation of 1001..1102"


def main():
    import joblib

    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True, help="challenge root (with trn/, tst/, stimuli_*)")
    p.add_argument("--sub", nargs="+", required=True, help="subject numbers, e.g. 01 05 06")
    p.add_argument("--features", nargs="+", default=["clip", "lowlevel"],
                   choices=["alexnet", *FEATURE_FILES])
    p.add_argument("--work", default="work", help="cache dir for frames/features/models")
    p.add_argument("--out", default="submissions", help="dir for the ranked candidate lists")
    p.add_argument("--skip-train", action="store_true", help="only score an existing model")
    args = p.parse_args()

    root, work, out = Path(args.data_dir), Path(args.work), Path(args.out)
    work.mkdir(parents=True, exist_ok=True)
    lo, hi = T_WIN
    tag = "+".join(args.features)
    X_cand = video_features(args.features, root, work, TEST_VIDEO_IDS, "test")

    scores = []
    for sub in (f"{int(s):02d}" for s in args.sub):
        model_path = work / f"encoder_sub-{sub}_{tag}_win{lo}-{hi}.joblib"
        if not args.skip_train:
            trn = load_train_subject(root, sub)
            X_eeg = decimate_eeg(trn.X)[:, :, lo:hi]
            X_feat = video_features(args.features, root, work, trn.video_ids, "train")
            joblib.dump(fit_forward_model(X_feat, X_eeg), model_path, compress=3)
        bundle = joblib.load(model_path)

        tst = load_test_subject(root, sub)
        X_tst_eeg = decimate_eeg(tst.X)[:, :, lo:hi]
        pred = predict_eeg(bundle, X_cand)  # rows follow sorted candidate order

        # rows stay in ORIGINAL tst.npy order, for scoring and submission alike
        ranked = rank_videos(similarity(X_tst_eeg, pred)).astype(np.int64)
        check_submission(ranked)
        if tst.y is not None:
            rank = np.argmax(ranked == tst.y[:, None], axis=1) + 1
            scores.append((topk_accuracy(ranked, tst.y), topk_accuracy(ranked, tst.y, 5), rank.mean()))
            print(f"sub-{sub}  top-1 {scores[-1][0]:.4f} | top-5 {scores[-1][1]:.4f} | "
                  f"mean rank {scores[-1][2]:.1f}   (alpha {bundle['model'].alpha_:.3g})")
        else:
            out.mkdir(parents=True, exist_ok=True)
            path = out / f"sub-{sub}_ranked_candidate_list.npy"
            np.save(path, ranked)
            print(f"sub-{sub}  saved {path} with shape {ranked.shape}")

    if scores:
        t1, t5, mr = np.mean(scores, axis=0)
        print(f"MEAN [{tag}, {lo * 10}-{hi * 10} ms] over {len(scores)} labeled subjects: "
              f"top-1 {t1:.4f} | top-5 {t5:.4f} | mean rank {mr:.1f}   "
              f"(chance {1 / 102:.4f} / {5 / 102:.4f} / 51.5)")


if __name__ == "__main__":
    main()
