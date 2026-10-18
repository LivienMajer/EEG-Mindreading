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
from sklearn.linear_model import RidgeCV
from sklearn.preprocessing import StandardScaler

from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject
from .member_weights import save_ridge
from .score_dump import save_scores

# from snr_probe study: 50-350 ms @ 100 Hz wins; 50-600 as organizer reference
WINDOWS = {"snr": (5, 35), "org": (5, 60)}
KEY_TEMPLATE = "{vid:05d}_f{k:02d}"


def load_pooled(path, cache: bool = True) -> dict:
    """Memory-safe embedding loader -> {int video_id: (D,) mean-of-frames}.

    Frame-level npz files (keys '00001_f00'..) are HUGE for high-dim models
    (alexnet-f3: 8000 frames x 140k dims = 4.5 GB); materializing them with
    dict(np.load()) swaps out 16 GB machines. This streams one frame batch at
    a time, pools per video, and (cache=True) saves the result as
    '<stem>_pooled.npz' for instant reloads. Video-level npz files (keys are
    plain ids) pass through unchanged.
    """
    path = Path(path)
    cache_path = path.with_name(path.stem + "_pooled.npz")
    if cache and cache_path.exists():
        z = np.load(cache_path)
        return {int(k): z[k] for k in z.files}
    z = np.load(path)
    keys = z.files
    out = {}
    if "_" in keys[0]:  # frame-level -> pool
        groups: dict[int, list] = {}
        for k in keys:
            groups.setdefault(int(k.split("_")[0]), []).append(k)
        for vid, ks in groups.items():
            out[vid] = np.mean([z[k] for k in ks], axis=0)
    else:  # already video-level
        for k in keys:
            out[int(k)] = z[k]
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache_path, **{f"{v:05d}": a for v, a in out.items()})
    return out


def video_matrix_pooled(emb: dict, video_ids, agg: str = "mean"):
    """emb: {int vid: (D,) pooled embedding} -> (n_videos, D) float32."""
    missing = [v for v in video_ids if int(v) not in emb]
    if missing:
        raise KeyError(f"{len(missing)} videos missing in embeddings, "
                       f"e.g. {missing[:5]}")
    return np.stack([emb[int(v)] for v in video_ids]).astype(np.float32)


class HighDimPCA:
    """PCA for n_samples << n_features (e.g. 900 trials x 140k features).

    sklearn's covariance_eigh/`full` solvers build or SVD the (p x p) matrix,
    which is impossible at this scale. Instead we eigendecompose the small
    (n x n) Gram matrix -- mathematically identical PCA.

    Interface follows sklearn: fit(X) -> self, transform(X) -> projection.
    """

    def __init__(self, variance_ratio=0.95, n_components=None):
        """n_components (if set) overrides variance_ratio."""
        self.variance_ratio = variance_ratio
        self.n_components = n_components

    def fit(self, X):
        self.mean_ = X.mean(axis=0)
        Xc = X - self.mean_
        G = Xc @ Xc.T / (len(Xc) - 1)          # (n, n) Gram
        evals, evecs = np.linalg.eigh(G)       # ascending order
        evals = np.maximum(evals[::-1], 0.0)   # descending, clip numerical noise
        if self.n_components is not None:
            n_keep = min(self.n_components, len(evals))
        else:
            total = evals.sum()
            n_keep = int(np.searchsorted(np.cumsum(evals) / total,
                                         self.variance_ratio) + 1)
            n_keep = min(n_keep, len(evals) - 1)
        # components in feature space: rows of Xc.T @ evec, normalized so that
        # components_ are unit-norm and orthonormal (dual PCA).
        ek = evecs[:, ::-1][:, :n_keep]        # columns follow descending evals
        comps = Xc.T @ ek
        comps /= np.linalg.norm(comps, axis=0, keepdims=True) + 1e-12
        self.components_ = comps.T              # (n_keep, p)
        return self

    def transform(self, X):
        return (X - self.mean_) @ self.components_.T


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
    pca = HighDimPCA(variance_ratio=0.95).fit(X_z)
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
    p.add_argument("--agg", default="mean", choices=["mean"])
    p.add_argument("--window", default="snr", choices=list(WINDOWS))
    p.add_argument("--subjects", nargs="+", default=["01", "02", "03", "04"])
    args = p.parse_args()

    root = Path(args.data_dir)
    emb_trn = load_pooled(args.emb_train)
    emb_tst = load_pooled(args.emb_test)
    t_win = WINDOWS[args.window]

    # candidate matrix is shared across subjects: rows = TEST_VIDEO_IDS sorted
    X_cand = video_matrix_pooled(emb_tst, TEST_VIDEO_IDS, agg=args.agg)

    results = {}
    for sub in args.subjects:
        trn = load_train_subject(root, sub)
        # EEG: decimate, take same window (precomputed on 100 Hz grid)
        E = decimate(trn.X, 5, axis=-1, ftype="fir", zero_phase=True)
        E_full = E[:, :, t_win[0]:t_win[1]]  # windowed EEG target
        X_trn = video_matrix_pooled(emb_trn, trn.video_ids, agg=args.agg)

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
        # dump for rank fusion (same row convention as contrastive_pipeline tst path)
        Path("work/tst_ranked").mkdir(parents=True, exist_ok=True)
        emb_tag = Path(args.emb_train).stem.replace("_pooled", "")
        np.save(f"work/tst_ranked/ridge_{sub}_{emb_tag}_s0.npy", ranked)
        save_scores(f"work/tst_ranked/ridge_{sub}_{emb_tag}_s0.npy", S)
        save_ridge(sub, emb_tag, bundle, t_win)
        np.save(f"work/tst_ranked/truth_{sub}_s0.npy", truth)

    print(f"\n=== {Path(args.emb_test).name} | agg={args.agg} | window={args.window} ===")
    for sub, (t1, t5) in results.items():
        print(f"  sub-{sub}: top-1 {t1:.4f}  top-5 {t5:.4f}   (chance 0.0098/0.049)")
    t1s = [v[0] for v in results.values()]
    t5s = [v[1] for v in results.values()]
    print(f"  MEAN : top-1 {np.mean(t1s):.4f}  top-5 {np.mean(t5s):.4f}")
    from .log_score import log_score
    emb_tag = Path(args.emb_train).stem.replace("_pooled", "")
    for sub, (t1, t5) in results.items():
        log_score(experiment="ridge-tst", subject=sub, embedding=emb_tag,
                  window=args.window, top1=t1, top5=t5, note="")
    log_score(experiment="ridge-tst", subject="MEAN", embedding=emb_tag,
              window=args.window, top1=float(np.mean(t1s)),
              top5=float(np.mean(t5s)), note="")
    return results


if __name__ == "__main__":
    # run the imported copy, so saved ridge weights refer to src.bench_embeddings.HighDimPCA
    # and not to __main__ (they could not be reloaded otherwise)
    from src.bench_embeddings import main as _main
    _main()
