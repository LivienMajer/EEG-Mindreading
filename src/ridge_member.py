"""Daniel-style ridge member toward any video embedding, from any EEG time window.

Same as src.xsubj.audio_member (Daniel): EEG preprocessing and ridge features of
configs/xsubj.yaml, two ridges summed (pooled over the trn splits of all six subjects + one
on the scored subject's own trn split), candidates ranked after per-candidate normalisation.
Two things can change: the target embedding (--emb) and the EEG window of the ridge
features (--tmin/--tmax, s after onset; default 0.05-0.6 as in the config).

Used to test whether the time hierarchy of the RSA (early AlexNet layers ~105 ms, deep
layers ~225 ms) helps decoding: e.g. conv2 from 50-150 ms only, as an extra fusion member
(docs/megafusion_rebuild.md, section 7).

Writes <out-dir>/A<tag>_XX_s0.npy like src.prep_variants: sub-01..04 rows LABEL-SORTED (for
src.loo_new_members), sub-05/06 in ORIGINAL tst order (for src.make_submission --extra-tags).

    python -m src.ridge_member --emb work/rsa_features/alexnet_conv2.npz --tmin 0.05 --tmax 0.15 --tag _rconv2e
    python -m src.ridge_member --emb models/xsubj/features/clap.npz --tag _rclap   # = Daniel's audio member

Several --emb with matching --tmin/--tmax make one member from several (target, EEG window)
pairs, similarities summed (e.g. the frame shown at t decoded from t+0.05..t+0.6 s; section 13).
--data-tmax 3.0 keeps the whole recording in preprocessing (default: the config's 1 s).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import similarity
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.utils import load_config
from src.score_dump import save_scores
from src.xsubj.video.embeddings import VideoBank, _load_file

CANDIDATES = np.arange(1001, 1103)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--emb", nargs="+", required=True,
                    help="video embedding file(s); several = one (target, window) pair each, similarities summed")
    ap.add_argument("--tmin", type=float, nargs="+", default=None, help="ridge window start(s), s; default: config")
    ap.add_argument("--tmax", type=float, nargs="+", default=None, help="ridge window end(s), s; default: config")
    ap.add_argument("--erp-bin", type=float, default=None, help="ridge feature bin (s); default: config (0.01)")
    ap.add_argument("--data-tmax", type=float, default=None,
                    help="end of the preprocessed EEG window (s); default: config (1.0); 3.0 = whole recording")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--out-dir", default="work/prep_variants")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.data_tmax is not None:
        cfg["preprocess"]["tmax"] = args.data_tmax
    n = len(args.emb)
    tmins = args.tmin or [cfg["ridge"]["erp_tmin"]] * n
    tmaxs = args.tmax or [cfg["ridge"]["erp_tmax"]] * n
    if not len(tmins) == len(tmaxs) == n:
        raise SystemExit("--emb, --tmin and --tmax need the same number of values")
    data = load_preprocessed(cfg)
    fs, tmin = data[0].fs, data[0].tmin
    trn = {d.subject: d for d in data if d.split == "trn"}
    tst = sorted((d for d in data if d.split == "tst"), key=lambda d: d.subject)   # original row order
    sims = {d.subject: 0.0 for d in tst}
    for path, t0, t1 in zip(args.emb, tmins, tmaxs):
        r = dict(cfg["ridge"], erp_tmin=t0, erp_tmax=t1)
        if args.erp_bin is not None:
            r["erp_bin"] = args.erp_bin
        bank = VideoBank(*_load_file(path), standardize=True)
        emb = lambda ids: bank.emb[bank.lookup(np.asarray(ids).astype(str))]
        pooled = RidgeDecoder(r, fs, tmin).fit(np.concatenate([d.X for d in trn.values()]),
                                               emb(np.concatenate([d.video_id for d in trn.values()])))
        cand = emb(CANDIDATES)
        print(f"{args.tag}: {path}, ridge window {t0}-{t1} s, bin {r['erp_bin']} s", flush=True)
        for d in tst:
            own = RidgeDecoder(r, fs, tmin).fit(trn[d.subject].X, emb(trn[d.subject].video_id))
            sims[d.subject] = sims[d.subject] + similarity(pooled.predict(d.X) + own.predict(d.X), cand, col_norm=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    scores = []
    for d in tst:
        sim = sims[d.subject]
        ranked = CANDIDATES[np.argsort(-sim, axis=1, kind="stable")].astype(np.int64)
        assert ranked.shape == (102, 102) and (np.sort(ranked, axis=1) == CANDIDATES).all()
        sub = d.subject.removeprefix("sub-")
        path = out / f"A{args.tag}_{sub}_s0.npy"
        if d.video_id is None:
            np.save(path, ranked)
            save_scores(path, sim)
            continue
        y = d.video_id.astype(int)
        np.save(path, ranked[np.argsort(y)])
        save_scores(path, sim[np.argsort(y)])
        rank = np.argmax(ranked == y[:, None], axis=1) + 1
        scores.append(((rank == 1).mean(), (rank <= 5).mean(), rank.mean()))
        print(f"  {d.subject}: top-1 {scores[-1][0]:.4f}  top-5 {scores[-1][1]:.4f}  mean rank {scores[-1][2]:.1f}")
    t1, t5, mr = np.mean(scores, axis=0)
    print(f"  mean sub-01..04: top-1 {t1:.4f}  top-5 {t5:.4f}  mean rank {mr:.1f}")

if __name__ == "__main__":
    main()
