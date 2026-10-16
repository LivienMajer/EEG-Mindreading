"""Evaluation harness: candidate-set matching accuracy.

The competition metric: given EEG trials and a set of candidate videos,
fraction of trials where the correct video is ranked first.
"""

from __future__ import annotations

import argparse

import numpy as np


def match_accuracy(eeg_embs: np.ndarray, vid_embs: np.ndarray,
                   y: np.ndarray, frame_level: bool = True) -> float:
    """Top-1 candidate-match accuracy.

    eeg_embs: (n_trials, D)
    vid_embs: (n_candidates, D) if frame_level=False,
              or list/array of (n_candidates, n_frames, D) if True
    y: (n_trials,) indices into candidates
    """
    eeg = eeg_embs / (np.linalg.norm(eeg_embs, axis=-1, keepdims=True) + 1e-8)

    if frame_level and vid_embs.ndim == 3:
        # cosine sim to every frame, take the max per candidate
        v = vid_embs / (np.linalg.norm(vid_embs, axis=-1, keepdims=True) + 1e-8)
        sims = np.einsum("td,cfd->tc", eeg, v).max(axis=-1)
    else:
        v = vid_embs / (np.linalg.norm(vid_embs, axis=-1, keepdims=True) + 1e-8)
        sims = eeg @ v.T

    return float((sims.argmax(axis=1) == y).mean())


def leave_one_subject_out_acc(X, y, subjects, model_fn, vid_embs, **kw):
    """Cross-subject validation - mimics the held-out test subjects.

    model_fn(X_train, y_train) -> anything with .transform(X) -> embeddings.
    """
    accs = {}
    for s in np.unique(subjects):
        tr, te = subjects != s, subjects == s
        model = model_fn(X[tr], y[tr])
        accs[s] = match_accuracy(model.transform(X[te]), vid_embs, y[te], **kw)
        print(f"subject {s}: acc {accs[s]:.3f}")
    mean = float(np.mean(list(accs.values())))
    print(f"LOSOCV mean: {mean:.3f}")
    return accs, mean


def baseline_centroid(X: np.ndarray, y: np.ndarray):
    """Nearest-centroid baseline: average EEG per video, match by correlation."""
    class Centroid:
        def __init__(self, X, y):
            self.centroids = {lab: X[y == lab].mean(axis=0) for lab in np.unique(y)}

        def transform(self, X):
            # similarity = 1 - correlation distance per video
            sims = np.stack([
                np.array([np.corrcoef(t, c)[0, 1] for t in X])
                for c in self.centroids.values()], axis=1)
            return sims @ np.eye(sims.shape[1])  # passthrough

    return Centroid(X, y)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Run evaluation")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--data", default="data/train")
    p.add_argument("--model", default="models/eeg_encoder.pt")
    p.add_argument("--video-embeddings", default="data/video_embeddings.npz")
    args = p.parse_args()

    import yaml
    cfg = yaml.safe_load(open(args.config))

    from .load_data import load_all
    from .preprocess import bandpass, notch, reject_artifacts, log_bandpower
    from .eeg_encoder import LinearProjection

    data = load_all(args.data, sfreq=cfg["data"]["sfreq"])
    X = notch(bandpass(data.X, cfg["data"]["sfreq"], **cfg["filter"]),
              cfg["data"]["sfreq"])
    keep = reject_artifacts(X, cfg["data"]["artifact_threshold_uv"])
    X, y, subj = X[keep], data.y[keep], data.subjects[keep]

    vid_embs = dict(np.load(args.video_embeddings))
    vid_ids = sorted(vid_embs.keys())
    V_per_frame = np.stack([vid_embs[v] for v in vid_ids])  # (C, F, D)
    lab_to_idx = {lab: i for i, lab in enumerate(vid_ids)}

    feats = log_bandpower(X, cfg["data"]["sfreq"], cfg["features"]["bands"])
    X_feat = feats.reshape(len(X), -1)
    y_idx = np.array([lab_to_idx.get(int(v), -1) for v in y])
    keep2 = y_idx >= 0
    X_feat, y_idx, subj2 = X_feat[keep2], y_idx[keep2], subj[keep2]

    def model_fn(Xtr, ytr):
        targets = V_per_frame[ytr].mean(axis=1)  # trial -> mean frame embedding
        return LinearProjection(alpha=cfg["model"]["ridge_alpha"]).fit(Xtr, targets)

    leave_one_subject_out_acc(X_feat, y_idx, subj2, model_fn, V_per_frame)
