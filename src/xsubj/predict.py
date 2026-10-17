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
from src.xsubj.metrics import retrieval_metrics, weighted_csls_similarity
from src.xsubj.train import (
    embed,
    embed_heads,
    embed_tta,
    load_checkpoint,
    target_embeddings,
)
from src.xsubj.utils import get_device, load_config, resolve
from src.xsubj.video.embeddings import load_video_bank


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--config", default=None, help="data config for the test set (default: checkpoint's)")
    ap.add_argument("--subjects", nargs="*", default=None)
    ap.add_argument("--candidates", default=None)
    ap.add_argument("--top", type=int, default=5)
    ap.add_argument("--tta-offsets", nargs="+", type=float, default=None,
                    metavar="SECONDS",
                    help="enable EEG test-time crop averaging at these start offsets")
    ap.add_argument("--tta-duration", type=float, default=0.8,
                    help="duration in seconds for each TTA crop (default: 0.8)")
    args = ap.parse_args()

    device = get_device()
    ck_cfg = load_config(args.config) if args.config else None
    import torch
    cfg = ck_cfg or torch.load(args.checkpoint, map_location="cpu", weights_only=False)["cfg"]
    bank = load_video_bank(cfg)
    model, ck = load_checkpoint(args.checkpoint, bank)
    model.to(device)
    candidate_embeddings = target_embeddings(model, bank, device)
    head_weights = {
        name: float(cfg["model"].get("head_weights", {}).get(name, 1.0))
        for name in model.head_names
    } if model.multi_head else None
    cand = json.loads(Path(args.candidates).read_text()) if args.candidates else {}
    lut = {s: i for i, s in enumerate(ck["subject_list"])}

    out = resolve("models/xsubj/predictions") / f"{Path(args.checkpoint).parent.name}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["subject", "trial", "prediction", f"top{args.top}", "true"])
        for d in load_preprocessed(cfg, args.subjects):
            subj = np.full(len(d.X), lut.get(d.subject, -1))
            if model.multi_head:
                head_predictions = (
                    embed_tta(
                        model, d.X, subj, device, d.fs, args.tta_offsets,
                        args.tta_duration)
                    if args.tta_offsets else
                    embed_heads(model, d.X, subj, device))
                if not isinstance(head_predictions, dict):
                    raise ValueError(
                        "Multi-head prediction expected per-head embeddings")
                all_head_embeddings = {
                    name: bank.spaces[name] for name in model.head_names}
                full_scores = weighted_csls_similarity(
                    head_predictions, all_head_embeddings, head_weights, k=10)
                for i in range(len(d.X)):
                    ids = (
                        np.asarray(cand[d.subject][i])
                        if d.subject in cand else bank.ids)
                    if d.subject in cand:
                        positions = bank.lookup(ids)
                        scores = weighted_csls_similarity(
                            head_predictions,
                            {name: bank.spaces[name][positions]
                             for name in model.head_names},
                            head_weights, k=10)[i]
                    else:
                        scores = full_scores[i]
                    order = np.argsort(-scores)
                    w.writerow([
                        d.subject, i, ids[order[0]],
                        " ".join(ids[order[: args.top]]),
                        d.video_id[i] if d.video_id is not None else ""])
                if d.video_id is not None and d.subject not in cand:
                    true_positions = bank.lookup(d.video_id)
                    ranks = (
                        full_scores > full_scores[
                            np.arange(len(d.X)), true_positions, None]
                    ).sum(axis=1)
                    print(
                        d.subject, "(late fusion)",
                        f"top1={(ranks == 0).mean():.3f}",
                        f"top{args.top}={(ranks < args.top).mean():.3f}")
                continue

            pred = (
                embed_tta(
                    model, d.X, subj, device, d.fs, args.tta_offsets,
                    args.tta_duration)
                if args.tta_offsets else embed(model, d.X, subj, device))
            if isinstance(pred, dict):
                raise ValueError("Single-head prediction received multiple embeddings")
            for i, z in enumerate(pred):
                ids = np.array(cand[d.subject][i]) if d.subject in cand else bank.ids
                order = np.argsort(
                    -(candidate_embeddings[bank.lookup(ids)] @ z))
                w.writerow([d.subject, i, ids[order[0]], " ".join(ids[order[: args.top]]),
                            d.video_id[i] if d.video_id is not None else ""])
            if d.video_id is not None:
                m = retrieval_metrics(
                    pred, bank.lookup(d.video_id), candidate_embeddings,
                    cfg["eval"]["k_candidates"])
                seen = "seen" if d.subject in lut else "unseen"
                print(d.subject, f"({seen})", "  ".join(f"{k}={v:.3f}" for k, v in m.items() if k.startswith("top1")))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
