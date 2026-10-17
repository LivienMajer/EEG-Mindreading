"""E-A: ONE joint InfoNCE encoder trained on ALL subjects' trn (the
'no-fusion simplification': fold the cross-subject idea into a single model
instead of fusing 8 rankings).

Motivation (docs/cross_subject_idea.md, E-C sonde said GO): a single subject
has ~950 trials; pooled over 6 subjects there are ~5400. One model with a
learned subject token gets 6x data, and the joint encoder may generalize
better than per-subject models. This is our own joint model (the teammate's
E-D encoder is the same family, but different preprocessing/architecture --
so if both are strong, they are complementary evidence).

Protocol per dev subject s (tst unseen during training, trn legal):
  1. joint model trained on pooled trn of ALL 6 subjects (incl. 05/06 trn)
  2. optional finetune: +30 epochs on s's own trn only (lower lr)
  3. rank s's tst -> top-1/top-5 (label-sorted convention, dumped for fusion)

Reference points (dev mean, seed 0): solo clip 11.03/33.09,
LOO-validated megafusion 22.30/48.28.

Reads legal data only: trn of all subjects (05/06 trn is allowed), tst is
ranked, never trained on. Row conventions: dev dumps label-sorted,
sub-05/06 dumps in ORIGINAL tst.npy order (no labels exist).

Usage:
  python -m src.joint_infonce --data-dir <challenge root> \
      --emb-train work/emb_train__clip-vitb32.npz \
      --emb-test work/emb_test__clip-vitb32.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .bench_embeddings import load_pooled
from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject
from .contrastive_pipeline import build_target_space, eeg_window, infonce
from .log_score import log_score

ALL_SUBS = ["01", "02", "03", "04", "05", "06"]
DEV_SUBS = ["01", "02", "03", "04"]
K1, K2 = 16, 8          # same kernel sizes as the solo runs (window is 30 samples)


class JointEEGNet(nn.Module):
    """EEGNet backbone (identical to build_eegnet) + subject token added to
    the 256-d hidden layer, so each subject gets a learned bias bank."""

    def __init__(self, n_channels: int, n_times: int, n_subjects: int,
                 embed_dim: int = 256, subj_dim: int = 256, dropout: float = 0.5):
        super().__init__()
        self.firstconv = nn.Sequential(
            nn.Conv2d(1, 8, (1, K1), padding=(0, K1 // 2), bias=False),
            nn.BatchNorm2d(8))
        self.depthwise = nn.Sequential(
            nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(dropout))
        self.sepconv = nn.Sequential(
            nn.ZeroPad2d((K2 // 2 - 1, K2 // 2, 0, 1)),
            nn.Conv2d(16, 16, (1, K2), bias=False),
            nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(dropout))
        self.proj = nn.LazyLinear(256)
        self.subj = nn.Embedding(n_subjects, subj_dim)
        self.out = nn.Linear(256, embed_dim)
        self.act = nn.ELU()

    def forward(self, x, subj):                 # x: (B, n_ch, n_times)
        z = x.unsqueeze(1)                       # (B, 1, n_ch, n_times)
        z = self.firstconv(z)
        z = self.depthwise(z)
        z = self.sepconv(z)
        h = self.act(self.proj(z.flatten(1)) + self.subj(subj))
        return self.out(h)


def train_loop(model, E, V, subj, *, epochs, lr, temperature, batch_size,
               seed, log_every, tag):
    """Mirror of contrastive_pipeline.train_infonce, + subject ids."""
    torch.manual_seed(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    ds = TensorDataset(torch.from_numpy(E.astype(np.float32)),
                       torch.from_numpy(V.astype(np.float32)),
                       torch.from_numpy(subj.astype(np.int64)))
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)
    for epoch in range(1, epochs + 1):
        model.train()
        tot = hits = n = 0
        for xb, vb, sb in dl:
            ze = model(xb, sb)
            ze = ze / (ze.norm(dim=-1, keepdim=True) + 1e-8)
            loss, ok = infonce(ze, vb, temperature)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(xb)
            hits += ok.sum().item(); n += len(xb)
        sched.step()
        if log_every and (epoch % log_every == 0 or epoch == 1):
            print(f"  [{tag}] epoch {epoch:3d} | loss {tot / n:.4f} "
                  f"| batch top-1 {hits / n:.3f}", flush=True)


@torch.no_grad()
def rank_subj(model, E_tst, subj_id, X_cand, batch_size=512):
    """(n, 17, w) tst EEG -> (n, n_cand) ranked candidate IDs (input order)."""
    model.eval()
    Z = []
    for i in range(0, len(E_tst), batch_size):
        xb = torch.from_numpy(E_tst[i:i + batch_size].astype(np.float32))
        sb = torch.full((len(xb),), subj_id, dtype=torch.int64)
        z = model(xb, sb)
        Z.append((z / (z.norm(dim=-1, keepdim=True) + 1e-8)).numpy())
    Z = np.concatenate(Z)
    Zc = X_cand / (np.linalg.norm(X_cand, axis=1, keepdims=True) + 1e-8)
    S = Z @ Zc.T
    return TEST_VIDEO_IDS[np.argsort(-S, axis=1)]


def evaluate(model, tst, subj_id, X_cand, emb_tag, name, sub,
             dump=True):
    """Rank a tst set; score if labels exist; dump in the right convention."""
    E_tst = eeg_window(tst.X)
    ranked = rank_subj(model, E_tst, subj_id, X_cand)
    Path("work/tst_ranked").mkdir(parents=True, exist_ok=True)
    if tst.y is not None:
        order = np.argsort(tst.y)
        ranked_e = ranked[order]
        t1 = float((ranked_e[:, 0] == tst.y[order]).mean())
        t5 = float((ranked_e[:, :5] == tst.y[order][:, None]).any(1).mean())
        print(f"  {name} sub-{sub}: top-1 {t1:.4f}  top-5 {t5:.4f}",
              flush=True)
        log_score(experiment=name, subject=sub, embedding=emb_tag,
                  top1=t1, top5=t5, note="E-A joint")
        if dump:
            np.save(f"work/tst_ranked/{name}_{sub}_{emb_tag}_s0.npy", ranked_e)
        return t1, t5
    if dump:
        # sub-05/06: ORIGINAL tst.npy row order (submission convention)
        np.save(f"work/tst_ranked/{name}_{sub}_{emb_tag}_s0.npy", ranked)
    print(f"  {name} sub-{sub}: no labels, ranking dumped (original order)",
          flush=True)
    return None, None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--emb-train", required=True)
    p.add_argument("--emb-test", required=True)
    p.add_argument("--pool-subs", nargs="+", default=ALL_SUBS)
    p.add_argument("--epochs", type=int, default=100,
                   help="joint epochs (5.7x data per epoch vs solo runs)")
    p.add_argument("--finetune-epochs", type=int, default=30)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--finetune-lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    args = p.parse_args()

    root = Path(args.data_dir)
    emb_tag = Path(args.emb_train).stem.replace("_pooled", "")

    emb_trn = load_pooled(args.emb_train)
    emb_tst = load_pooled(args.emb_test)
    X_trn_p, X_cand, _ = build_target_space(emb_trn, emb_tst, n_components=256)
    V_ids = sorted(emb_trn)
    row = {v: i for i, v in enumerate(V_ids)}

    # ---------------- pooled trn of ALL subjects --------------------------
    print(f"pooling trn of {args.pool_subs} ...", flush=True)
    E_parts, V_parts, S_parts = [], [], []
    subj_idx = {}
    for i, sub in enumerate(args.pool_subs):
        trn = load_train_subject(root, sub)
        E = eeg_window(trn.X)
        V = X_trn_p[[row[int(v)] for v in trn.video_ids]]
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        E_parts.append(E); V_parts.append(V)
        S_parts.append(np.full(len(E), i, dtype=np.int64))
        subj_idx[sub] = i
        print(f"  sub-{sub}: {len(E)} trials", flush=True)
    E_all = np.concatenate(E_parts)
    V_all = np.concatenate(V_parts)
    S_all = np.concatenate(S_parts)
    print(f"pooled: {len(E_all)} trials, {len(args.pool_subs)} subjects",
          flush=True)

    # ---------------- joint training --------------------------------------
    torch.manual_seed(args.seed)
    model = JointEEGNet(E_all.shape[1], E_all.shape[2],
                        n_subjects=len(args.pool_subs),
                        embed_dim=args.embed_dim)
    # init LazyLinear
    model(torch.zeros(1, E_all.shape[1], E_all.shape[2]),
          torch.zeros(1, dtype=torch.long))

    print(f"\n=== joint training: {args.epochs} epochs, "
          f"{len(E_all)} trials ===", flush=True)
    train_loop(model, E_all, V_all, S_all, epochs=args.epochs, lr=args.lr,
               temperature=args.temperature, batch_size=args.batch_size,
               seed=args.seed, log_every=args.log_every, tag="joint")

    ck_dir = Path("work/joint_model"); ck_dir.mkdir(exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "n_channels": E_all.shape[1], "n_times": E_all.shape[2],
                "n_subjects": len(args.pool_subs),
                "embed_dim": args.embed_dim,
                "subj_dim": 256,
                "pool_subs": args.pool_subs,
                "emb_tag": emb_tag},
               ck_dir / f"joint_{emb_tag}_s{args.seed}.pt")
    print("checkpoint saved", flush=True)

    # ---------------- evaluation: joint-only on all 6 ----------------------
    print("\n=== joint-only rankings ===", flush=True)
    base = model.state_dict()   # keep the joint weights for finetune restarts
    for sub in args.pool_subs:
        tst = load_test_subject(root, sub, with_labels=sub in DEV_SUBS)
        evaluate(model, tst, subj_idx[sub], X_cand, emb_tag, "joint", sub)

    # ---------------- finetune per subject (restart from joint) ------------
    if args.finetune_epochs > 0:
        print(f"\n=== joint + finetune ({args.finetune_epochs} epochs on own "
              f"trn, lr {args.finetune_lr}) ===", flush=True)
        for sub in args.pool_subs:
            ft = JointEEGNet(E_all.shape[1], E_all.shape[2],
                            n_subjects=len(args.pool_subs),
                            embed_dim=args.embed_dim)
            ft(torch.zeros(1, E_all.shape[1], E_all.shape[2]),
               torch.zeros(1, dtype=torch.long))
            ft.load_state_dict(base)
            i = subj_idx[sub]
            mask = S_all == i
            train_loop(ft, E_all[mask], V_all[mask], S_all[mask],
                       epochs=args.finetune_epochs, lr=args.finetune_lr,
                       temperature=args.temperature,
                       batch_size=args.batch_size, seed=args.seed,
                       log_every=0, tag=f"ft{sub}")
            tst = load_test_subject(root, sub, with_labels=sub in DEV_SUBS)
            evaluate(ft, tst, i, X_cand, emb_tag, "jointft", sub)
            del ft

    print("\ndone.", flush=True)


if __name__ == "__main__":
    main()
