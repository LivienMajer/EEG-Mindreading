"""E-D member: teammate cross-subject encoder (frozen) as a fusion member.

Runs ONLY the tst splits through the teammates' EXACT inference path
(same preprocess_subject code, same video bank, same EEGEncoder checkpoint,
same similarity + stable argsort ranking as src.xsubj.submit --model encoder),
but reads the challenge data from OUR repo layout and skips their full-dataset
preprocessing cache (which would reprocess all 12 subject/split combinations).

Outputs per subject, in ORIGINAL tst.npy row order (no label sorting):
    work/ed_member/ranked_subXX.npy   (102, 102) int64, video ids 1001..1102

With --dev-sorted-dir, the labeled subjects (sub-01..04) are also written LABEL-SORTED
(np.argsort(tst.y)) as <dir>/ed_XX_xsubj_s0.npy, the file loo_generalization reads.

Usage:
  python -m src.ed_member --data-dir provided_material/eeg-mind-reading-challenge/eeg-mind-reading-challenge
  python -m src.ed_member --data-dir <root> --checkpoint <model.pt> --out work/ed_member \
      --dev-sorted-dir work/tst_ranked
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .challenge_data import TEST_VIDEO_IDS, load_test_subject
from .log_score import log_score
from .score_dump import save_scores


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--checkpoint", default="models/xsubj/checkpoints/all/model.pt")
    ap.add_argument("--out", default="work/ed_member", help="original-order dumps")
    ap.add_argument("--dev-sorted-dir", default=None,
                    help="also write label-sorted dev dumps ed_XX_xsubj_s0.npy here")
    ap.add_argument("--subjects", nargs="+",
                    default=["01", "02", "03", "04", "05", "06"])
    args = ap.parse_args()

    import torch
    from src.xsubj.eeg.adapter import SubjectData
    from src.xsubj.eeg.preprocess import preprocess_subject
    from src.xsubj.metrics import similarity
    from src.xsubj.nets.encoders import EEGEncoder
    from src.xsubj.train import embed
    from src.xsubj.video.embeddings import VideoBank

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    n_subj = (ck["state_dict"]["subject_layer.W"].shape[0]
              if cfg["model"]["subject_layer"] else 0)

    bank = VideoBank(*_load_bank(cfg["data"]["video_emb"]),
                     cfg["data"].get("video_standardize", False))
    model = EEGEncoder(ck["n_channels"], ck["n_times"], bank.dim, n_subj,
                       cfg["model"], fs=ck["fs"], tmin=cfg["preprocess"]["tmin"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    lut = {s: i for i, s in enumerate(ck["subject_list"])}

    cand = bank.emb[bank.lookup(TEST_VIDEO_IDS.astype(str))]

    root = Path(args.data_dir)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.dev_sorted_dir:
        Path(args.dev_sorted_dir).mkdir(parents=True, exist_ok=True)

    for sub in args.subjects:
        tst = load_test_subject(root, sub)
        X = tst.X.astype(np.float32)
        d = SubjectData(f"sub-{sub}", X, None, 500.0, 0.0,
                        [f"ch{i}" for i in range(X.shape[1])], split="tst")
        d = preprocess_subject(d, cfg["preprocess"])
        subj_ids = np.full(len(d.X), lut[f"sub-{sub}"], dtype=np.int64)
        pred = embed(model, d.X, subj_ids, "cpu")
        pred = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-8)
        sim = similarity(pred, cand, col_norm=False)
        ranked = TEST_VIDEO_IDS[np.argsort(-sim, axis=1, kind="stable")].astype(np.int64)
        np.save(out / f"ranked_sub{sub}.npy", ranked)
        save_scores(out / f"ranked_sub{sub}.npy", sim)

        if tst.y is not None and args.dev_sorted_dir:
            np.save(Path(args.dev_sorted_dir) / f"ed_{sub}_xsubj_s0.npy", ranked[np.argsort(tst.y)])
            save_scores(Path(args.dev_sorted_dir) / f"ed_{sub}_xsubj_s0.npy", sim[np.argsort(tst.y)])
        if tst.y is not None:
            t1 = float((ranked[:, 0] == tst.y).mean())
            t5 = float((ranked[:, :5] == tst.y[:, None]).any(1).mean())
            print(f"sub-{sub} [E-D solo]: top-1 {t1:.4f}  top-5 {t5:.4f}")
            log_score(experiment="ed-encoder-solo", subject=sub,
                      embedding="xsubj-shallow-clip", seed=0, holdout=0,
                      top1=t1, top5=t5, note="teammate frozen encoder, orig order")
        else:
            print(f"sub-{sub} [E-D]: no labels, ranking dumped")


def _load_bank(path):
    z = np.load(path, allow_pickle=False)
    return z["ids"].astype(str), z["emb"].astype(np.float32)


if __name__ == "__main__":
    main()
