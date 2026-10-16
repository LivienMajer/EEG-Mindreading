"""Rank candidate videos for each EEG trial with a trained encoder.

    python -m src.xsubj.predict --checkpoint models/xsubj/checkpoints/all/model.pt --subjects sub-05 sub-06
    python -m src.xsubj.predict --checkpoint ... --candidates candidates.json

candidates.json (optional): {"<subject>": [[vid, vid, ...], ...]} with one candidate list
per trial. Without it, every video in the embedding bank is a candidate.
Writes models/xsubj/predictions/<run>.csv; if the trials are labeled, also prints retrieval accuracy.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import retrieval_metrics
from src.xsubj.train import embed, load_checkpoint
from src.xsubj.utils import get_device, load_config, resolve
from src.xsubj.video.embeddings import load_video_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="data config for the test set (default: checkpoint's)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--top", type=int, default=5)
    args = ap.parse_args()

    device = get_device()
    ck_cfg = load_config(args.config) if args.config else None
    import torch
    cfg = ck_cfg or torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"]
    bank = load_video_bank(cfg)
    model, ck = load_checkpoint(args.checkpoint, bank.dim)
    model.to(device)
    cand = json.loads(Path(args.candidates).read_text()) if args.candidates else {}
    lut = {s: i for i, s in enumerate(ck["subject_list"])}

    out = resolve("models/xsubj/predictions") / f"{Path(args.checkpoint).parent.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial", "prediction", f"top{args.top}", "true"])
        for d in load_preprocessed(cfg, args.subjects):
            subj = np.full(len(d.X), lut.get(d.subject, -1))
            pred = embed(model, d.X, subj, device)
            for i, z in enumerate(pred):
                ids = np.array(cand[d.subject][i]) if d.subject in cand else bank.ids
                order = np.argsort(-(bank.emb[bank.lookup(ids)] @ z))
                w.writerow([d.subject, i, ids[order[0]], " ".join(ids[order[: args.top]]),
                            d.video_id[i] if d.video_id is not None else ""])
            if d.video_id is not None:
                m = retrieval_metrics(pred, bank.lookup(d.video_id), bank.emb, cfg["eval"]["k_candidates"])
                seen = "seen" if d.subject in lut else "unseen"
                print(d.subject, f"({seen})", "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k.startswith("top1")))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
