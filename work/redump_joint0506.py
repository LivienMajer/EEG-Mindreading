"""Re-dump the sub-05/06 joint/jointft rankings in ORIGINAL tst.npy row
order from the saved joint checkpoint (the main run consumed the teammates'
pseudo tst_stim labels and dumped fake-argsorted rows).

Replay determinism: the jointft finetune in the main run started from the
same joint weights (= checkpoint) and train_loop() resets torch.manual_seed
at entry -> identical trajectory. Rankings never depend on labels.
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, ".")
from src.bench_embeddings import load_pooled  # noqa: E402
from src.challenge_data import load_test_subject, load_train_subject  # noqa: E402
from src.contrastive_pipeline import build_target_space, eeg_window  # noqa: E402
from src.joint_infonce import JointEEGNet, rank_subj, train_loop  # noqa: E402

ROOT = Path("provided_material/eeg-mind-reading-challenge/eeg-mind-reading-challenge")
EMB = "emb_train__clip-vitb32"
SEED = 0

emb_trn = load_pooled(f"work/{EMB}.npz")
emb_tst = load_pooled(f"work/emb_test__clip-vitb32.npz")
X_trn_p, X_cand, _ = build_target_space(emb_trn, emb_tst, n_components=256)
V_ids = sorted(emb_trn)
row = {v: i for i, v in enumerate(V_ids)}

ck = torch.load(f"work/joint_model/joint_{EMB}_s{SEED}.pt",
                map_location="cpu", weights_only=False)
pool_subs = ck["pool_subs"]
subj_idx = {s: i for i, s in enumerate(pool_subs)}

E_parts, V_parts, S_parts = [], [], []
for sub in pool_subs:
    trn = load_train_subject(ROOT, sub)
    E = eeg_window(trn.X)
    V = X_trn_p[[row[int(v)] for v in trn.video_ids]]
    V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
    E_parts.append(E); V_parts.append(V)
    S_parts.append(np.full(len(E), subj_idx[sub], dtype=np.int64))
E_all = np.concatenate(E_parts)
V_all = np.concatenate(V_parts)
S_all = np.concatenate(S_parts)

model = JointEEGNet(ck["n_channels"], ck["n_times"], ck["n_subjects"],
                    ck["embed_dim"])
model(torch.zeros(1, ck["n_channels"], ck["n_times"]),
      torch.zeros(1, dtype=torch.long))
model.load_state_dict(ck["state_dict"])

for sub in ("05", "06"):
    tst = load_test_subject(ROOT, sub, with_labels=False)
    assert tst.y is None, f"sub-{sub}: pseudo labels leaked into loader!"
    i = subj_idx[sub]

    ranked = rank_subj(model, eeg_window(tst.X), i, X_cand)
    np.save(f"work/tst_ranked/joint_{sub}_{EMB}_s{SEED}.npy", ranked)
    print(f"sub-{sub}: joint   re-dumped (original order)", flush=True)

    ft = JointEEGNet(ck["n_channels"], ck["n_times"], ck["n_subjects"],
                     ck["embed_dim"])
    ft(torch.zeros(1, ck["n_channels"], ck["n_times"]),
       torch.zeros(1, dtype=torch.long))
    ft.load_state_dict(ck["state_dict"])
    mask = S_all == i
    train_loop(ft, E_all[mask], V_all[mask], S_all[mask], epochs=30,
               lr=1e-4, temperature=0.1, batch_size=256, seed=SEED,
               log_every=0, tag=f"ft{sub}")
    ranked = rank_subj(ft, eeg_window(tst.X), i, X_cand)
    np.save(f"work/tst_ranked/jointft_{sub}_{EMB}_s{SEED}.npy", ranked)
    print(f"sub-{sub}: jointft re-dumped (original order)", flush=True)
    del ft

print("OK: joint/jointft 05/06 dumps restored in original row order")
