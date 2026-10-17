"""Contrastive alignment pipeline adapted to the actual challenge (branch:
contrastive-alignment). Owns: InfoNCE / shared-embedding-space learning.

Protocol (mirrors the challenge task EXACTLY):
  - Train the EEG encoder on a subject's TRAIN trials (videos 1..1000).
  - Evaluate on that subject's TEST recordings: rank all 102 candidates
    (videos 1001..1102) by cosine similarity -> top-1 / top-5 accuracy.
  - Model selection uses ONLY dev subjects 01-04 (their tst labels exist).

Baseline to beat (ridge forward model, same dev subjects):
  alexnet-f3, window 50-350 ms  ->  see results in README findings log.

Data notes (measured, see src/snr_probe.py):
  - NO filtering (raw decimated data wins); window 50-350 ms @100 Hz.
  - Per-trial z-scoring per channel (unsupervised, allowed on test).

Usage on THIS CPU machine:
  python -m src.contrastive_pipeline --data-dir <challenge root> --sub 01 \
      --emb-train work/emb_train__alexnet-f3.npz \
      --emb-test work/emb_test__alexnet-f3.npz
Until GPU teammates deliver CLIP/DINOv2 embeddings, alexnet-f3 npz files
(extracted already, in work/) are the drop-in development target space.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from .bench_embeddings import (KEY_TEMPLATE, HighDimPCA, load_pooled,
                               video_matrix_pooled)
from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject

# snr-window result: 50-350 ms @100 Hz
T_WIN = (5, 35)
DEV_SUBJECTS = ["01", "02", "03", "04"]


# ---------------------------------------------------------------------------
# EEG features
# ---------------------------------------------------------------------------

def eeg_window(X: np.ndarray, t_win=T_WIN, dec: int = 5) -> np.ndarray:
    """(n, 17, 1500) @500Hz -> decimated, windowed, per-trial z-scored (n, 17, w)."""
    from .challenge_data import zscore
    Z = zscore(X, axis=-1)                 # per-chunk channel z-scoring
    E = decimate(Z, dec, axis=-1, ftype="fir", zero_phase=True)
    return E[:, :, t_win[0]:t_win[1]]


# ---------------------------------------------------------------------------
# Target space preparation (PCA-reduced video embeddings)
# ---------------------------------------------------------------------------

def build_target_space(emb_train, emb_test, n_components=256, agg="mean"):
    """Returns (X_trn_feat, X_cand, pca) with candidates in the SAME pca space.

    PCA is fit on TRAIN video embeddings only (unsupervised wrt test videos,
    no label leakage): test-video features are just transformed.
    """
    V_ids = sorted(emb_train)
    X_trn = video_matrix_pooled(emb_train, V_ids)
    X_cand_full = video_matrix_pooled(emb_test, TEST_VIDEO_IDS)

    # joint scaling: fit scaler on train embeddings
    mu, sd = X_trn.mean(0), X_trn.std(0) + 1e-8
    pca = HighDimPCA(n_components=n_components).fit((X_trn - mu) / sd)
    X_trn_p = pca.transform((X_trn - mu) / sd)
    X_cand = pca.transform((X_cand_full - mu) / sd)
    return X_trn_p, X_cand, {"mu": mu, "sd": sd, "pca": pca}


def sorted_train_ids(emb):
    return sorted({int(k.split("_")[0]) for k in emb})


# ---------------------------------------------------------------------------
# InfoNCE training (CPU-friendly)
# ---------------------------------------------------------------------------

def train_infonce(Xe, V, trn_ids, V_ids, embed_dim=256, batch_size=256,
                  epochs=200, lr=3e-4, temperature=0.1, seed=0,
                  log_every=20, device="cpu", subject_token=True,
                  net_seed=None):
    """Train EEGNet to align EEG trials with (PCAReduced) video embeddings.

    Xe:   (n_trials, 17, t_win) EEG train input
    V:    (n_train_videos, d) video embedding targets for each trial
          (rows follow trn_ids / V_ids ordering)
    subject_token: adds a learned per-subject bias bank (identity only).
    """
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    from .eeg_encoder import build_eegnet

    torch.manual_seed(net_seed if net_seed is not None else seed)
    rng = np.random.default_rng(net_seed if net_seed is not None else seed)

    n_trials = len(Xe)
    d_emb = V.shape[1]
    model = build_eegnet(Xe.shape[1], Xe.shape[2], embed_dim=embed_dim,
                         k1=16, k2=8).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    # trials point at their video's row in V
    vid_to_row = {int(v): i for i, v in enumerate(V_ids)}

    ds = TensorDataset(torch.from_numpy(Xe.astype(np.float32)),
                       torch.from_numpy(V.astype(np.float32)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in range(1, epochs + 1):
        model.train()
        tot, hits, n = 0.0, 0, 0
        for xb, vb in dl:
            xb, vb = xb.to(device), vb.to(device)
            ze = model(xb)
            ze = ze / (ze.norm(dim=-1, keepdim=True) + 1e-8)

            # shuffled video batch as negatives (video set >> batch size)
            # vb rows correspond 1:1 to xb rows (same trial order!)
            loss, ok = infonce(ze, vb, temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
            hits += ok.sum().item(); n += len(xb)
        sched.step()
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"  epoch {epoch:3d} | loss {tot / n:.4f} "
                  f"| batch top-1 {hits / n:.3f}")
    return model


def infonce(ze, zv, temperature=0.1):
    """Symmetric InfoNCE; ze, zv L2-normalized (B, d)."""
    import torch
    import torch.nn.functional as F
    logits = ze @ zv.T / temperature
    labels = torch.arange(len(ze), device=ze.device)
    loss = (F.cross_entropy(logits, labels) +
            F.cross_entropy(logits.T, labels)) / 2
    return loss, logits.argmax(1) == labels


def rank_candidates(model, E_tst_win, X_cand, device="cpu", batch_size=512,
                    cand_ids=None):
    """(n, 17, w) test EEG -> (n, n_cand) similarity -> candidate ranking.

    Returns ranked candidate video IDs (n, n_cand); rows follow INPUT order.
    cand_ids: 1-D array of candidate video IDs (default: TEST_VIDEO_IDS).
    """
    import torch
    if cand_ids is None:
        cand_ids = TEST_VIDEO_IDS
    model.eval()
    with torch.no_grad():
        Z = []
        for i in range(0, len(E_tst_win), batch_size):
            xb = torch.from_numpy(
                E_tst_win[i:i + batch_size].astype(np.float32)).to(device)
            z = model(xb)
            Z.append((z / (z.norm(dim=-1, keepdim=True) + 1e-8)).cpu().numpy())
        Z = np.concatenate(Z)
    Zc = X_cand / (np.linalg.norm(X_cand, axis=1, keepdims=True) + 1e-8)
    S = Z @ Zc.T                                   # (n, n_cand) cosine sim
    return cand_ids[np.argsort(-S, axis=1)]


# ---------------------------------------------------------------------------
# main: dev-subject protocol
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--sub", required=True)
    p.add_argument("--emb-train", required=True)
    p.add_argument("--emb-test", required=True)
    p.add_argument("--n-components", type=int, default=256,
                   help="PCA dims of target space")
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--net-seed", type=int, default=-1,
                   help="network init/shuffle seed; -1 = use --seed (legacy). "
                        "Split the object of study: --seed controls the "
                        "holdout split AND net unless overridden here.")
    p.add_argument("--holdout", type=int, default=0,
                   help="if >0: hold out N train videos as PSEUDO candidates "
                        "and evaluate ranking them (validation WITHOUT "
                        "touching tst labels). 0 = train on all, eval on tst.")
    args = p.parse_args()

    root = Path(args.data_dir)
    emb_trn = load_pooled(args.emb_train)
    emb_tst = load_pooled(args.emb_test)

    trn = load_train_subject(root, args.sub)
    E_full = eeg_window(trn.X)                    # (n_trials, 17, 30)

    rng = np.random.default_rng(args.seed)
    if args.holdout > 0:
        # ---- pseudo-candidate validation (Exp #2) ------------------------
        # hold out N of THIS subject's train videos; train on the rest,
        # then rank the held-out videos' trials against all N candidates.
        # Uses ONLY train data -> tst labels stay untouched.
        vids = np.array(sorted(set(trn.video_ids.tolist())))
        pick = rng.permutation(len(vids))[:args.holdout]
        hold_ids = vids[pick]
        is_hold = np.isin(trn.video_ids, hold_ids)
        E_fit, v_fit = E_full[~is_hold], trn.video_ids[~is_hold]
        E_hold, v_hold = E_full[is_hold], trn.video_ids[is_hold]
        print(f"holdout: training on {len(E_fit)} trials, "
              f"ranking {len(E_hold)} held-out trials over "
              f"{args.holdout} candidates")
    else:
        E_fit, v_fit = E_full, trn.video_ids
        hold_ids = None

    # target space: PCA on train-video embeddings of the FIT videos only
    if hold_ids is not None:
        emb_fit = {v: a for v, a in emb_trn.items() if v not in set(hold_ids.tolist())}
    else:
        emb_fit = emb_trn
    X_trn_p, X_cand, space = build_target_space(emb_fit, emb_tst,
                                                 n_components=args.n_components)

    # target row per trial: trial -> its video's row in X_trn_p
    V_ids = sorted(emb_fit)
    row = {v: i for i, v in enumerate(V_ids)}
    V = X_trn_p[[row[int(v)] for v in v_fit]]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)

    print(f"sub-{args.sub}: {len(E_fit)} trials, target dim {V.shape[1]}, "
          f"window {T_WIN} @{100} Hz")
    net_seed = args.net_seed if args.net_seed >= 0 else args.seed
    model = train_infonce(E_fit, V, v_fit, V_ids,
                          embed_dim=args.embed_dim,
                          net_seed=net_seed,
                          batch_size=args.batch_size,
                          epochs=args.epochs, lr=args.lr,
                          temperature=args.temperature, seed=args.seed)

    if hold_ids is not None:
        # rank held-out trials against the held-out videos' embeddings,
        # transformed through the SAME PCA that was fit on the fit videos
        C_raw = video_matrix_pooled(emb_trn, hold_ids)
        C = space["pca"].transform((C_raw - space["mu"]) / space["sd"])
        C = C / (np.linalg.norm(C, axis=1, keepdims=True) + 1e-8)
        ranked = rank_candidates(model, E_hold, C, cand_ids=hold_ids)
        y = v_hold
        r = ranked
        t1 = float((r[:, 0] == y).mean())
        t5 = float((r[:, :5] == y[:, None]).any(1).mean())
        # dump per-trial rankings for cross-run overlap analysis
        Path("work/holdout").mkdir(parents=True, exist_ok=True)
        emb_tag = Path(args.emb_train).stem.replace("_pooled", "")
        np.save(f"work/holdout/ranked_sub{args.sub}_{emb_tag}_s{args.seed}_n{net_seed}.npy", r)
        np.save(f"work/holdout/y_sub{args.sub}_s{args.seed}.npy", y)
        print(f"\nsub-{args.sub} HOLDOUT RESULT: top-1 {t1:.4f}  "
              f"top-5 {t5:.4f}  (chance {1/len(hold_ids):.4f}/"
              f"{min(5, len(hold_ids))/len(hold_ids):.4f}, "
              f"{len(hold_ids)} candidates)")
        from .log_score import log_score
        emb_tag = Path(args.emb_train).stem.replace("_pooled", "")
        log_score(experiment="infonce-holdout", subject=args.sub,
                  embedding=emb_tag, window=f"{T_WIN}",
                  n_components=args.n_components, embed_dim=args.embed_dim,
                  epochs=args.epochs, batch_size=args.batch_size,
                  temperature=args.temperature, lr=args.lr,
                  seed=args.seed, holdout=args.holdout,
                  top1=t1, top5=t5, note="")
        return

    tst = load_test_subject(root, args.sub)
    E_tst = eeg_window(tst.X)
    ranked = rank_candidates(model, E_tst, X_cand)

    if tst.y is not None:
        order = np.argsort(tst.y)                 # eval order (label-sorted)
        ranked_e = ranked[order]
        t1 = float((ranked_e[:, 0] == tst.y[order]).mean())
        t5 = float((ranked_e[:, :5] == tst.y[order][:, None]).any(1).mean())
        print(f"\nsub-{args.sub} RESULT: top-1 {t1:.4f}  top-5 {t5:.4f} "
              f"(chance 0.0098/0.049)")
        from .log_score import log_score
        emb_tag = Path(args.emb_train).stem.replace("_pooled", "")
        log_score(experiment="infonce", subject=args.sub,
                  embedding=emb_tag, window=f"{T_WIN}",
                  n_components=args.n_components, embed_dim=args.embed_dim,
                  epochs=args.epochs, batch_size=args.batch_size,
                  temperature=args.temperature, lr=args.lr,
                  seed=args.seed, holdout=0, top1=t1, top5=t5, note="")
        Path("work/tst_ranked").mkdir(parents=True, exist_ok=True)
        np.save(f"work/ranked_sub{args.sub}_infonce.npy", ranked_e)
        np.save(f"work/tst_ranked/infonce_{args.sub}_{emb_tag}_s{args.seed}.npy",
                ranked_e)
        np.save(f"work/tst_ranked/truth_{args.sub}_s0.npy", tst.y[order])
    else:
        print("no labels (sub-05/06): submission file written")
        np.save(f"work/sub-{args.sub}_ranked_candidate_list.npy", ranked)


if __name__ == "__main__":
    main()
