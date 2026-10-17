"""E-C probe: cross-subject ranking WITHOUT new machinery — does a model
trained on subject Y's TRAIN data rank subject X's TEST trials above chance?

This is the cheap go/no-go gate for the cross-subject idea (see
docs/cross_subject_idea.md, Experiment E-C). It measures how much of the
learned EEG->video mapping is subject-general vs idiosyncratic:

  - diagonal (X == Y): sanity, must reproduce our tst numbers (~11% top-1)
  - off-diagonal: pure transfer. Above chance => joint training (E-A/E-B) has
    signal to build on; transferred rankings can also serve as diverse fusion
    members for the 3 subjects we have no per-subject model for.

No checkpoints were saved by the tst runs, so the probe RETRAINS each train
subject once (identical protocol/seed as the tst runs) and then ranks ALL
dev subjects' tst sets with the frozen per-subject model.

Cost: 4 trainings x ~13 min (CPU) + 16 cheap inferences. Results are logged
per (train, test) pair to scores.csv and dumped as ranking files for fusion.

Usage:
  python -m src.cross_subject_probe --data-dir <challenge root> \
      --emb-train work/emb_train__clip-vitb32.npz \
      --emb-test work/emb_test__clip-vitb32.npz
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .bench_embeddings import load_pooled
from .challenge_data import TEST_VIDEO_IDS, load_test_subject, load_train_subject
from .contrastive_pipeline import (build_target_space, eeg_window,
                                   rank_candidates, train_infonce)
from .log_score import log_score

# dev subject ids (DEV_SUBJECTS is local to contrastive_pipeline; redefined
# here to avoid odd import games)
DEV_SUBJECTS = ("01", "02", "03", "04")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", required=True)
    p.add_argument("--emb-train", required=True)
    p.add_argument("--emb-test", required=True)
    p.add_argument("--n-components", type=int, default=256)
    p.add_argument("--embed-dim", type=int, default=256)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--train-subs", nargs="+", default=DEV_SUBJECTS,
                   help="subjects whose models to train (default: 01-04)")
    p.add_argument("--test-subs", nargs="+", default=DEV_SUBJECTS,
                   help="tst sets to rank with each model (default: 01-04)")
    args = p.parse_args()

    root = Path(args.data_dir)
    emb_tag = Path(args.emb_train).stem.replace("_pooled", "")

    emb_trn = load_pooled(args.emb_train)
    emb_tst = load_pooled(args.emb_test)

    # target space: identical to the tst path (PCA on ALL train-video
    # embeddings of the embedding space, shared across subjects)
    X_trn_p, X_cand, space = build_target_space(
        emb_trn, emb_tst, n_components=args.n_components)
    V_ids = sorted(emb_trn)
    row = {v: i for i, v in enumerate(V_ids)}

    E_tst, y_tst = {}, {}
    for ts in args.test_subs:
        tst = load_test_subject(root, ts)
        E_tst[ts] = eeg_window(tst.X)
        y_tst[ts] = tst.y

    for tr in args.train_subs:
        trn = load_train_subject(root, tr)
        E_full = eeg_window(trn.X)
        V = X_trn_p[[row[int(v)] for v in trn.video_ids]]
        V = V / (np.linalg.norm(V, axis=1, keepdims=True) + 1e-8)
        print(f"\n=== training model on sub-{tr} "
              f"({len(E_full)} trials) ===")
        model = train_infonce(E_full, V, trn.video_ids, V_ids,
                              embed_dim=args.embed_dim, net_seed=args.seed,
                              batch_size=args.batch_size, epochs=args.epochs,
                              lr=args.lr, temperature=args.temperature,
                              seed=args.seed, log_every=50)

        # save checkpoint: reusable for sub-05/06 transfer ranking
        # (their tst has no labels, but their EEG can legitimately be ranked)
        Path("work/xsubj_probe").mkdir(parents=True, exist_ok=True)
        import torch
        torch.save({"state_dict": model.state_dict(),
                    "n_channels": E_full.shape[1], "n_times": E_full.shape[2],
                    "embed_dim": args.embed_dim, "emb_tag": emb_tag,
                    "target_space": "fit on all train-video embeddings, "
                                    "n_components=" + str(args.n_components),
                    "subject": tr},
                   f"work/xsubj_probe/model_sub{tr}_{emb_tag}_s{args.seed}.pt")

        for ts in args.test_subs:
            ranked = rank_candidates(model, E_tst[ts], X_cand)
            if y_tst[ts] is None:
                print(f"  model sub-{tr} -> tst sub-{ts} [no labels]: "
                      f"ranking dumped only")
                np.save(f"work/xsubj_probe/ranked_tr{tr}_te{ts}_{emb_tag}_s{args.seed}.npy",
                        ranked)   # original row order, ready for submission
                continue
            order = np.argsort(y_tst[ts])
            ranked_e = ranked[order]
            t1 = float((ranked_e[:, 0] == y_tst[ts][order]).mean())
            t5 = float((ranked_e[:, :5] == y_tst[ts][order][:, None]).any(1).mean())
            tag = "diag" if ts == tr else "transfer"
            print(f"  model sub-{tr} -> tst sub-{ts} [{tag}]: "
                  f"top-1 {t1:.4f}  top-5 {t5:.4f}")
            log_score(experiment="xsubj-probe", subject=f"tr{tr}-te{ts}",
                      embedding=emb_tag, seed=args.seed, holdout=0,
                      top1=t1, top5=t5, note=tag)
            np.save(f"work/xsubj_probe/ranked_tr{tr}_te{ts}_{emb_tag}_s{args.seed}.npy",
                    ranked_e)


if __name__ == "__main__":
    main()
