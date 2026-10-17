"""Audio vote as a megafusion member: ridge from EEG to the CLAP embedding of each video's sound track.

Same EEG preprocessing and ridge features as the submission (configs/xsubj.yaml); the target is the
CLAP audio embedding (src.xsubj.video.foundation_features --model clap) instead of CLIP. Two ridges,
summed: pooled over the trn splits of all six subjects + one on the scored subject's own trn split.
Candidates are ranked after per-candidate normalisation, as in src.xsubj.submit.

Dev (sub-01..04) alone: top-1 4.9 %, top-5 21.1 %, mean rank 25.7 (chance 1.0 / 4.9 / 51.5). Added to
the submission ensemble as a separate similarity term (S_image + w * S_audio, w = 0.5 chosen by
cross-validation over training videos): mean rank 11.53 -> 11.22 (better in 188 vs 139 recordings,
p = 0.008), top-1 17.6 -> 18.9 %, top-5 47.8 -> 46.8 % (both within noise).

Writes, in the format of src.ed_member:
  <out>/ranked_subXX.npy           (102, 102) int64 video ids, ORIGINAL tst.npy row order, sub-01..06
  <dev-sorted-dir>/clap_XX_ridge_s0.npy   sub-01..04, LABEL-SORTED rows (for loo_generalization)

    python -m src.xsubj.audio_member
    python -m src.xsubj.audio_member --out work/clap_member --dev-sorted-dir work/tst_ranked
"""
from __future__ import annotations

import argparse

import numpy as np

from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import similarity
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.utils import load_config, resolve
from src.xsubj.video.embeddings import VideoBank, _load_file

CANDIDATES = np.arange(1001, 1103)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--audio-emb", default="models/xsubj/features/clap.npz")
    ap.add_argument("--out", default="models/xsubj/members/clap_ridge")
    ap.add_argument("--dev-sorted-dir", default="models/xsubj/members/clap_ridge/dev_sorted")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data = load_preprocessed(cfg)
    fs, tmin = data[0].fs, data[0].tmin
    bank = VideoBank(*_load_file(args.audio_emb), standardize=True)
    emb = lambda ids: bank.emb[bank.lookup(np.asarray(ids).astype(str))]
    trn = {d.subject: d for d in data if d.split == "trn"}

    pooled = RidgeDecoder(cfg["ridge"], fs, tmin).fit(np.concatenate([d.X for d in trn.values()]),
                                                       emb(np.concatenate([d.video_id for d in trn.values()])))
    out, dev = resolve(args.out), resolve(args.dev_sorted_dir)
    out.mkdir(parents=True, exist_ok=True)
    dev.mkdir(parents=True, exist_ok=True)
    cand = emb(CANDIDATES)
    scores = []
    for d in sorted((d for d in data if d.split == "tst"), key=lambda d: d.subject):  # original order
        own = RidgeDecoder(cfg["ridge"], fs, tmin).fit(trn[d.subject].X, emb(trn[d.subject].video_id))
        sim = similarity(pooled.predict(d.X) + own.predict(d.X), cand, col_norm=True)
        ranked = CANDIDATES[np.argsort(-sim, axis=1, kind="stable")].astype(np.int64)
        assert ranked.shape == (102, 102) and (np.sort(ranked, axis=1) == CANDIDATES).all()
        sub = d.subject.removeprefix("sub-")
        np.save(out / f"ranked_sub{sub}.npy", ranked)
        if d.video_id is None:
            print(f"{d.subject}: no labels, ranking written")
            continue
        y = d.video_id.astype(int)
        np.save(dev / f"clap_{sub}_ridge_s0.npy", ranked[np.argsort(y)])
        rank = np.argmax(ranked == y[:, None], axis=1) + 1
        scores.append(((rank == 1).mean(), (rank <= 5).mean(), rank.mean()))
        print(f"{d.subject}: top-1 {scores[-1][0]:.4f}  top-5 {scores[-1][1]:.4f}  mean rank {scores[-1][2]:.1f}")
    t1, t5, mr = np.mean(scores, axis=0)
    print(f"mean sub-01..04: top-1 {t1:.4f}  top-5 {t5:.4f}  mean rank {mr:.1f}\nwritten to {out} and {dev}")


if __name__ == "__main__":
    main()
