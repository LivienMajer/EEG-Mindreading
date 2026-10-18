"""Interim submission: xsubj default ensemble + the megafusion's per-subject InfoNCE members as votes.

    S = S_xsubj + sum_k w_k * colnorm(S_k)      (optionally Sinkhorn-normalised, then ranked)

S_xsubj is the weighted sum of src.xsubj.submit's per-candidate normalised votes (weights from
configs/xsubj.yaml, read from `submit --dump-scores`). The four InfoNCE members (per-subject
EEGNet, 50-350 ms, PCA-256 target, src/contrastive_pipeline.py on experiment/eegnet-preprocessing)
come from their saved dumps in models/xsubj/members/infonce_dumps/:
  clip, alexf3   similarity tables S (dev rows label-sorted, sub-05/06 rows in original order)
  fc6, vmae8     rankings only (same row convention); turned into scores as minus the rank
These dumps are an interim stand-in: src/xsubj/infonce_member.py retrains the members with saved
checkpoints so that submit.py can compute the votes itself (submit.infonce_votes).

Weights (fc6 0.5, VideoMAE-l8 1.0, AlexNet-f3 0.5, CLIP 0.5) were chosen on sub-01..04 only (grid
0/0.5/1/1.5 per member, criterion mean of top-1 and top-5; leave-one-dev-subject-out estimate
36.3% / 67.2% / 6.4 vs 28.7% / 65.9% / 7.2 without). Sinkhorn tau 0.5 was chosen the same way.
  without Sinkhorn  21.1% / 52.5% / 9.7      with Sinkhorn tau 0.5  23.0% / 52.9% / 9.5
  (xsubj default    20.1% / 48.0% / 11.2)

    python -m src.xsubj.submit --dump-scores work/xsubj_scores --out work/xsubj_default
    python -m src.xsubj.fuse_infonce                       # -> submissions_infonce/
    python -m src.xsubj.fuse_infonce --sinkhorn 0.5        # -> submissions_infonce_sinkhorn/
"""
from __future__ import annotations

import argparse

import numpy as np

from src.xsubj.submit import CANDIDATES, check_and_score, ranked_candidates
from src.xsubj.utils import load_config, resolve

MEMBERS = {"fc6": 0.5, "vmae8": 1.0, "alexf3": 0.5, "clip": 0.5}
DEV = ["sub-01", "sub-02", "sub-03", "sub-04"]


def col_norm(S: np.ndarray) -> np.ndarray:
    return (S - S.mean(0)) / (S.std(0) + 1e-8)


def rank_to_score(ranked: np.ndarray) -> np.ndarray:
    """(n, 102) ranked video ids -> (n, 102) score = minus the rank, columns 1001..1102."""
    R = np.empty(ranked.shape)
    R[np.arange(len(ranked))[:, None], np.searchsorted(CANDIDATES, ranked)] = np.arange(ranked.shape[1])
    return -R


def sinkhorn(S: np.ndarray, tau: float, n_iter: int = 200) -> np.ndarray:
    """log of the doubly normalised exp(S / tau): every recording and every video sums to 1."""
    P = np.exp((S - S.max()) / tau)
    for _ in range(n_iter):
        P /= P.sum(1, keepdims=True)
        P /= P.sum(0, keepdims=True)
    return np.log(np.maximum(P, 1e-300))


def member_vote(name: str, subject: str, labels: np.ndarray | None) -> np.ndarray:
    folder = resolve("models/xsubj/members/infonce_dumps")
    if name in ("fc6", "vmae8"):
        S = rank_to_score(np.load(folder / f"{name}_ranked_{subject}.npy"))
    else:
        S = np.load(folder / f"{name}_{subject}.npz")["S"].astype(np.float64)
    if labels is not None:          # dev dumps are label-sorted: row i = video 1001 + i
        S = S[labels - CANDIDATES[0]]
    return col_norm(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--scores", default="work/xsubj_scores", help="output of submit --dump-scores")
    ap.add_argument("--sinkhorn", type=float, default=None, help="tau; off by default")
    ap.add_argument("--subjects", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config)["submit"]
    w = {"image": cfg.get("image_weight", 1.0)} | {v["name"]: v["weight"] for v in cfg.get("extra_votes", [])}
    out = resolve(args.out or ("submissions_infonce_sinkhorn" if args.sinkhorn else "submissions_infonce"))
    out.mkdir(parents=True, exist_ok=True)
    root = resolve(load_config(args.config)["data"]["challenge_dir"])
    for subject in args.subjects:
        stim = root / "tst_stim" / f"{subject}_tst_stim.npy"
        labels = np.load(stim).astype(int) if subject in DEV else None   # dev only: undo label sorting
        votes = np.load(resolve(args.scores) / f"{subject}.npz")
        S = sum(w[k] * votes[k] for k in w)
        S = S + sum(wk * member_vote(k, subject, labels) for k, wk in MEMBERS.items())
        if args.sinkhorn:
            S = sinkhorn(S, args.sinkhorn)
        ranked = ranked_candidates(S).astype(np.int64)
        np.save(out / f"{subject}_ranked_candidate_list.npy", ranked)
        truth, note = labels, ""
        print(f"{subject}: {check_and_score(ranked, truth)}{note}")
    print(f"written to {out}")


if __name__ == "__main__":
    main()
