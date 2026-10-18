"""Save (and reload) the weights of the fusion members that are trained inside the fusion scripts.

The rules ask for "all model weights required to reproduce the results". Daniel's encoders,
his ridges and the transfer models were already saved; these were not:
  - per-subject InfoNCE EEGNets (members 1-2: src.contrastive_pipeline; members 9-10:
    src.prep_variants) -> models/fusion/infonce/<ranking file name>.pt
  - per-subject ridge forward models (member 4: src.bench_embeddings for sub-01..04,
    src.make_submission for sub-05/06) -> models/fusion/ridge/ridge_<sub>_<emb tag>.joblib

An InfoNCE checkpoint holds the network weights and everything inference needs besides the
test EEG: the network shape, the EEG preprocessing (`prep`: None = contrastive_pipeline's
eeg_window, else the src.prep_variants variant) and the candidate videos in the network's
target space (X_cand, so the PCA is not refitted). src.check_member_weights reloads every
checkpoint and confirms that it reproduces the member's ranking file exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

WEIGHTS_DIR = Path("models/fusion")


def save_infonce(name: str, model, E: np.ndarray, X_cand: np.ndarray, embed_dim: int,
                 *, prep: dict | None, emb: str) -> Path:
    """name: the member's ranking file name without .npy; E: (n, 17, t) network input."""
    import torch
    path = WEIGHTS_DIR / "infonce" / f"{name}.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "n_channels": E.shape[1], "n_times": E.shape[2],
                "embed_dim": embed_dim, "k1": 16, "k2": 8, "prep": prep, "emb": emb,
                "X_cand": np.asarray(X_cand, np.float32)}, path)
    return path


def load_infonce(path):
    """-> (network in eval mode, checkpoint dict)"""
    import torch

    from .eeg_encoder import build_eegnet
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_eegnet(ck["n_channels"], ck["n_times"], embed_dim=ck["embed_dim"],
                         k1=ck["k1"], k2=ck["k2"])
    model.load_state_dict(ck["state_dict"])
    return model.eval(), ck


def save_ridge(sub: str, emb_tag: str, bundle: dict, window) -> Path:
    import joblib
    path = WEIGHTS_DIR / "ridge" / f"ridge_{sub}_{emb_tag}.joblib"
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({**bundle, "window": tuple(window), "emb_tag": emb_tag}, path)
    return path
