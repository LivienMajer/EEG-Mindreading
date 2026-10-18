"""Per-subject InfoNCE EEGNet votes (the megafusion members), trained and saved for src.xsubj.submit.

One small EEGNet per subject maps a nearly raw EEG trial (per-trial channel z-score, 100 Hz,
50-350 ms) into a PCA-256 space of a video model, trained with symmetric InfoNCE against the
subject's own training videos (port of src/contrastive_pipeline.py, branch
experiment/eegnet-preprocessing; same hyperparameters). Unlike the CLIP encoders of
src.xsubj.train there is no subject layer and no cross-subject training: every subject gets its
own network, so the vote errs differently from the xsubj encoders and ridges.

The PCA is fitted on the training videos only; the 102 candidates are projected through it and
stored in the checkpoint, so inference needs neither the (large) feature files nor refitting.

    python -m src.xsubj.infonce_member --target fc6 --emb models/xsubj/features/rsa/alexnet_fc6.npz
    python -m src.xsubj.infonce_member --target vmae8 --emb models/xsubj/features/rsa/videomae_l8.npz --seeds 0 1 2

Checkpoints: models/xsubj/checkpoints/infonce_<target>/sub-XX_s<seed>.pt
Use in the submission: `submit.infonce_votes` in configs/xsubj.yaml.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
from scipy.signal import decimate

from src.xsubj.utils import resolve

T_WIN = (5, 35)             # samples at 100 Hz: 50-350 ms
N_COMPONENTS = 256
HPARAMS = dict(embed_dim=256, batch_size=256, epochs=200, lr=3e-4, temperature=0.1)
CANDIDATES = np.arange(1001, 1103)


# ---------------------------------------------------------------------------- data

def eeg_window(X: np.ndarray) -> np.ndarray:
    """(n, 17, 1500) at 500 Hz -> per-trial channel z-score, 100 Hz, 50-350 ms: (n, 17, 30)."""
    Z = (X - X.mean(-1, keepdims=True)) / (X.std(-1, keepdims=True) + 1e-8)
    E = decimate(Z.astype(np.float32), 5, axis=-1, ftype="fir", zero_phase=True)
    return E[:, :, T_WIN[0]:T_WIN[1]].astype(np.float32)


def load_train(root: Path, subject: str):
    d = np.load(root / "trn" / f"{subject}_trn.npz")
    keys = sorted(d.files, key=int)
    return np.array([int(k) for k in keys]), np.stack([d[k] for k in keys]).astype(np.float32)


def load_test(root: Path, subject: str) -> np.ndarray:
    return np.load(root / "tst" / f"{subject}_tst.npy").astype(np.float32)   # original order


def load_emb(path) -> dict[int, np.ndarray]:
    """{video id: (d,)} from an ids/emb npz (models/xsubj/features) or a per-video-key npz
    (keys '00001', as written by src.bench_embeddings.load_pooled)."""
    z = np.load(resolve(path))
    if "ids" in z.files:
        return {int(i): e.astype(np.float32) for i, e in zip(z["ids"], z["emb"])}
    return {int(k): z[k].astype(np.float32) for k in z.files}


# ---------------------------------------------------------------------------- target space

def high_dim_pca(X: np.ndarray, k: int):
    """Dual (Gram-matrix) PCA for n << p; identical to sklearn PCA (src.bench_embeddings.HighDimPCA)."""
    mean = X.mean(0)
    Xc = X - mean
    evals, evecs = np.linalg.eigh(Xc @ Xc.T / (len(Xc) - 1))
    evals, evecs = np.maximum(evals[::-1], 0), evecs[:, ::-1]
    k = min(k, len(evals))
    comps = Xc.T @ evecs[:, :k]
    comps /= np.linalg.norm(comps, axis=0, keepdims=True) + 1e-12
    return mean, comps.T                              # (d,), (k, d)


def target_space(emb: dict[int, np.ndarray]):
    """Standardise + PCA on the 1000 training videos; candidates projected through the same map."""
    trn_ids = sorted(v for v in emb if v not in set(CANDIDATES.tolist()))
    X = np.stack([emb[v] for v in trn_ids])
    C = np.stack([emb[int(v)] for v in CANDIDATES])
    mu, sd = X.mean(0), X.std(0) + 1e-8
    mean, comps = high_dim_pca((X - mu) / sd, N_COMPONENTS)
    proj = lambda A: ((A - mu) / sd - mean) @ comps.T
    return trn_ids, proj(X).astype(np.float32), proj(C).astype(np.float32)


# ---------------------------------------------------------------------------- model

def build_eegnet(n_channels: int, n_times: int, embed_dim: int, dropout: float = 0.5, k1: int = 16, k2: int = 8):
    """EEGNet-style CNN + projection head (src/eeg_encoder.build_eegnet, k1/k2 for 30 samples)."""
    import torch
    import torch.nn as nn

    class EEGNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.firstconv = nn.Sequential(nn.Conv2d(1, 8, (1, k1), padding=(0, k1 // 2), bias=False),
                                           nn.BatchNorm2d(8))
            self.depthwise = nn.Sequential(nn.Conv2d(8, 16, (n_channels, 1), groups=8, bias=False),
                                           nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(dropout))
            self.sepconv = nn.Sequential(nn.ZeroPad2d((k2 // 2 - 1, k2 // 2, 0, 1)),
                                         nn.Conv2d(16, 16, (1, k2), bias=False),
                                         nn.BatchNorm2d(16), nn.ELU(), nn.Dropout(dropout))
            self.head = nn.Sequential(nn.Flatten(), nn.LazyLinear(256), nn.ELU(), nn.Linear(256, embed_dim))

        def forward(self, x):
            return self.head(self.sepconv(self.depthwise(self.firstconv(x.unsqueeze(1)))))

    net = EEGNet()
    with torch.no_grad():
        net(torch.zeros(1, n_channels, n_times))     # materialise the LazyLinear
    return net


def train(E: np.ndarray, V: np.ndarray, seed: int, epochs: int, batch_size: int, lr: float,
          temperature: float, embed_dim: int, device: str = "cpu"):
    """Symmetric InfoNCE between unit EEG embeddings and the trials' unit video targets."""
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset

    torch.manual_seed(seed)
    np.random.seed(seed)
    model = build_eegnet(E.shape[1], E.shape[2], embed_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    dl = DataLoader(TensorDataset(torch.from_numpy(E), torch.from_numpy(V)),
                    batch_size=batch_size, shuffle=True, drop_last=True)
    for _ in range(epochs):
        model.train()
        for xb, vb in dl:
            xb, vb = xb.to(device), vb.to(device)
            ze = model(xb)
            ze = ze / (ze.norm(dim=-1, keepdim=True) + 1e-8)
            logits = ze @ vb.T / temperature
            labels = torch.arange(len(ze), device=device)
            loss = (F.cross_entropy(logits, labels) + F.cross_entropy(logits.T, labels)) / 2
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()
    return model.cpu().eval()


# ---------------------------------------------------------------------------- inference

def similarity(path, X_tst: np.ndarray) -> np.ndarray:
    """Cosine similarity (recordings x 102 candidates) of one saved member on raw test EEG."""
    import torch
    ck = torch.load(path, map_location="cpu", weights_only=False)
    E = eeg_window(X_tst)
    model = build_eegnet(E.shape[1], E.shape[2], ck["hparams"]["embed_dim"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model(torch.from_numpy(E)).numpy()
    z /= np.linalg.norm(z, axis=1, keepdims=True) + 1e-8
    C = ck["cand"] / (np.linalg.norm(ck["cand"], axis=1, keepdims=True) + 1e-8)
    return z @ C.T


def vote(target: str, subject: str, seeds, X_tst: np.ndarray, col_norm: bool = True) -> np.ndarray:
    """Seed-averaged, per-candidate normalised similarity of one target space for one subject."""
    folder = resolve(f"models/xsubj/checkpoints/infonce_{target}")
    sims = []
    for s in seeds:
        S = similarity(folder / f"{subject}_s{s}.pt", X_tst)
        if col_norm:
            S = (S - S.mean(0)) / (S.std(0) + 1e-8)
        sims.append(S)
    return np.mean(sims, 0)


# ---------------------------------------------------------------------------- main

def main():
    import torch
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--target", required=True, help="name of the target space (checkpoint folder suffix)")
    ap.add_argument("--emb", required=True, nargs="+",
                    help="video features: one ids/emb npz, or train + test per-video-key npz files")
    ap.add_argument("--data-dir", default="data/eeg-mind-reading-challenge")
    ap.add_argument("--subjects", nargs="+", default=[f"sub-0{i}" for i in range(1, 7)])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0])
    args = ap.parse_args()

    from src.xsubj.utils import get_device
    device = get_device()
    emb = {}
    for p in args.emb:
        emb.update(load_emb(p))
    trn_ids, Vtrn, Vcand = target_space(emb)
    row = {v: i for i, v in enumerate(trn_ids)}
    root = resolve(args.data_dir)
    out = resolve(f"models/xsubj/checkpoints/infonce_{args.target}")
    out.mkdir(parents=True, exist_ok=True)
    for subject in args.subjects:
        vids, X = load_train(root, subject)
        E = eeg_window(X)
        V = Vtrn[[row[int(v)] for v in vids]]
        V = (V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)).astype(np.float32)
        for seed in args.seeds:
            t0 = time.time()
            model = train(E, V, seed, device=device, **HPARAMS)
            path = out / f"{subject}_s{seed}.pt"
            torch.save({"state_dict": model.state_dict(), "cand": Vcand, "cand_ids": CANDIDATES,
                        "hparams": HPARAMS, "t_win": T_WIN, "target": args.target,
                        "emb": [str(p) for p in args.emb], "seed": seed, "subject": subject}, path)
            print(f"{subject} seed {seed}: {len(E)} trials, {time.time() - t0:.0f} s -> {path}", flush=True)


if __name__ == "__main__":
    main()
