"""EEG preprocessing variants for the per-subject InfoNCE EEGNet (fusion members 1-2).

The InfoNCE members run on nearly raw EEG (per-trial z-score, 50-350 ms). The
teammate cross-subject encoder (E-D) uses filtered, mean-removed, Euclidean-
aligned EEG. This tests whether those steps also help the InfoNCE members.
Model, target space, training and ranking are contrastive_pipeline's,
unchanged; only the EEG input differs:

  A  baseline: per-trial z-score -> 100 Hz -> 50-350 ms (= eeg_window)
  B  A + Euclidean alignment (per subject and split)
  C  A + removal of the split's mean response (QC: test mean is inverted)
  D  40 Hz low-pass, then C, then Euclidean alignment
  E  A + Euclidean alignment with the TRAIN split's covariance applied to both
     splits (tests whether B fails because test trials, ~2x averaged, have a
     different covariance than single training trials)

All split statistics are label-free (allowed on test). Protocol as the tst
path of contrastive_pipeline: train on the subject's trn, rank its 102 tst
recordings; rankings are dumped label-sorted for paired comparison/fusion.
Subjects without tst labels (05/06) are dumped in ORIGINAL tst.npy row order
(submission convention) and not scored.

Usage:
  python -m src.prep_variants --data-dir <challenge root> --emb work/clip_vitb32_daniel.npz \
      --variants A B --subs 01 02 03 04
  python -m src.prep_variants --compare A B C D       # paired tests on the dumps
  python -m src.prep_variants --data-dir <root> --variants A --emb work/rsa_features/clap_audio.npz       --tag _clap                                     # same member, other target space
"""

from __future__ import annotations

import argparse
import itertools
import time
from pathlib import Path

import numpy as np
from scipy import signal, stats

from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject, zscore
from .contrastive_pipeline import T_WIN, build_target_space, rank_candidates, train_infonce
from .member_weights import save_infonce
from .score_dump import save_scores

VARIANTS = {
    "A": dict(lowpass=False, remove_mean=False, align=False),
    "B": dict(lowpass=False, remove_mean=False, align=True),
    "C": dict(lowpass=False, remove_mean=True, align=False),
    "D": dict(lowpass=True, remove_mean=True, align=True),
    "E": dict(lowpass=False, remove_mean=False, align="train"),
}
OUT = Path("work/prep_variants")


# ---------------------------------------------------------------------------
# EEG input per variant
# ---------------------------------------------------------------------------

def mean_cov(X: np.ndarray) -> np.ndarray:
    return np.mean([x @ x.T / x.shape[-1] for x in X], axis=0)


def whiten(X: np.ndarray, R: np.ndarray) -> np.ndarray:
    """R^-1/2 X for every trial."""
    w, V = np.linalg.eigh(R)
    w = np.maximum(w, w.max() * 1e-6)
    return np.einsum("cd,ndt->nct", (V / np.sqrt(w)) @ V.T, X).astype(np.float32)


def euclidean_alignment(X: np.ndarray) -> np.ndarray:
    """He & Wu (2019): whiten by the split's mean spatial covariance."""
    return whiten(X, mean_cov(X))


def prep_split(X: np.ndarray, lowpass: bool, remove_mean: bool, align: bool,
               t_win=T_WIN, dec: int = 5) -> np.ndarray:
    """(n, 17, 1500) @500 Hz -> (n, 17, w) @100 Hz; statistics from this split only.
    align="train" is left to the caller, which needs both splits."""
    if lowpass:
        sos = signal.butter(4, 40.0, btype="lowpass", fs=500.0, output="sos")
        X = signal.sosfiltfilt(sos, X, axis=-1)
    E = signal.decimate(zscore(X, axis=-1), dec, axis=-1, ftype="fir", zero_phase=True)
    if remove_mean:
        E = E - E.mean(axis=0, keepdims=True)
    E = E[:, :, t_win[0]:t_win[1]]
    return euclidean_alignment(E) if align is True else E.astype(np.float32)


# ---------------------------------------------------------------------------
# video side: {int video id: (d,)} from an ids/emb npz (e.g. src.xsubj CLIP bank)
# ---------------------------------------------------------------------------

def load_bank(path) -> tuple[dict, dict]:
    z = np.load(path, allow_pickle=False)
    emb = {int(i): e for i, e in zip(z["ids"], z["emb"].astype(np.float32))}
    test = set(TEST_VIDEO_IDS.tolist())
    return ({v: e for v, e in emb.items() if v not in test},
            {v: e for v, e in emb.items() if v in test})


# ---------------------------------------------------------------------------
# run / compare
# ---------------------------------------------------------------------------

def run(args):
    root = Path(args.data_dir)
    emb_trn, emb_tst = load_bank(args.emb)
    X_trn_p, X_cand, _ = build_target_space(emb_trn, emb_tst, n_components=256)
    row = {v: i for i, v in enumerate(sorted(emb_trn))}
    OUT.mkdir(parents=True, exist_ok=True)

    for sub in args.subs:
        trn = load_train_subject(root, sub)
        tst = load_test_subject(root, sub)
        labeled = tst.y is not None
        order = np.argsort(tst.y) if labeled else np.arange(len(tst.X))
        if labeled:
            np.save(OUT / f"truth_{sub}.npy", tst.y[order])
        V = X_trn_p[[row[int(v)] for v in trn.video_ids]]
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        for name in args.variants:
            t0 = time.time()
            cfg = VARIANTS[name]
            E_trn, E_tst = prep_split(trn.X, **cfg), prep_split(tst.X, **cfg)
            if cfg["align"] == "train":
                R = mean_cov(E_trn)
                E_trn, E_tst = whiten(E_trn, R), whiten(E_tst, R)
            model = train_infonce(E_trn, V, trn.video_ids, sorted(emb_trn),
                                  embed_dim=256, net_seed=args.seed, batch_size=256,
                                  epochs=args.epochs, lr=3e-4, temperature=0.1,
                                  seed=args.seed, log_every=50)
            ranked, S = rank_candidates(model, E_tst, X_cand, return_scores=True)
            save_infonce(f"{name}{args.tag}_{sub}_s{args.seed}", model, E_tst, X_cand, 256,
                         prep={**cfg, **({"R": R} if cfg["align"] == "train" else {})}, emb=str(args.emb))
            ranked = ranked[order]
            np.save(OUT / f"{name}{args.tag}_{sub}_s{args.seed}.npy", ranked)
            save_scores(OUT / f"{name}{args.tag}_{sub}_s{args.seed}.npy", S[order])
            if not labeled:
                print(f"sub-{sub} variant {name}: no labels, ranking dumped in original "
                      f"order ({time.time() - t0:.0f} s)", flush=True)
                continue
            t1 = float((ranked[:, 0] == tst.y[order]).mean())
            t5 = float((ranked[:, :5] == tst.y[order][:, None]).any(1).mean())
            print(f"sub-{sub} variant {name}: top-1 {t1:.4f}  top-5 {t5:.4f}  "
                  f"({time.time() - t0:.0f} s)", flush=True)
            from .log_score import log_score
            log_score(experiment="prep-variant", subject=sub, embedding=Path(args.emb).stem,
                      window=f"{T_WIN}", n_components=256, embed_dim=256,
                      epochs=args.epochs, batch_size=256, temperature=0.1, lr=3e-4,
                      seed=args.seed, holdout=0, top1=t1, top5=t5,
                      note=f"variant {name}: {cfg}")


def true_rank(ranked: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.argmax(ranked == y[:, None], axis=1) + 1          # 1 = best


def compare(args):
    subs = sorted(p.stem.split("_")[1] for p in OUT.glob("truth_*.npy"))
    res = {}
    for name in args.compare:
        files = [OUT / f"{name}_{s}_s{args.seed}.npy" for s in subs]
        if not all(f.exists() for f in files):
            print(f"variant {name}: missing dumps, skipped")
            continue
        res[name] = np.concatenate([true_rank(np.load(f), np.load(OUT / f"truth_{s}.npy"))
                                    for f, s in zip(files, subs)])
    print(f"subjects {subs}, {len(next(iter(res.values())))} test recordings, seed {args.seed}")
    for name, r in res.items():
        print(f"  {name}: top-1 {np.mean(r == 1):.4f}  top-5 {np.mean(r <= 5):.4f}  "
              f"mean rank {r.mean():.1f}")
    for a, b in itertools.combinations(res, 2):
        ra, rb = res[a], res[b]
        w = stats.wilcoxon(ra, rb).pvalue if np.any(ra != rb) else 1.0
        hit_a, hit_b = int(np.sum((ra == 1) & (rb != 1))), int(np.sum((rb == 1) & (ra != 1)))
        mc = stats.binomtest(hit_a, hit_a + hit_b).pvalue if hit_a + hit_b else 1.0
        print(f"  {a} vs {b}: rank better in {np.sum(ra < rb)} vs {np.sum(rb < ra)} "
              f"(Wilcoxon p={w:.3f}); top-1 only-{a} {hit_a} vs only-{b} {hit_b} (p={mc:.3f})")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir")
    p.add_argument("--emb", default="work/clip_vitb32_daniel.npz",
                   help="video bank: npz with ids + emb (any feature set, e.g. "
                        "eeg_pipeline.rsa_features output)")
    p.add_argument("--tag", default="", help="suffix for dump names, e.g. _clap")
    p.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    p.add_argument("--subs", nargs="+", default=["01", "02", "03", "04"])
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--compare", nargs="+", default=None, metavar="VARIANT")
    args = p.parse_args()
    if args.compare:
        compare(args)
    else:
        if not args.data_dir:
            p.error("--data-dir is required for training")
        run(args)


if __name__ == "__main__":
    main()
