"""Train-split cross-validation of the submission's ridge votes, as a weight-selection criterion
that also exists for sub-05/06 (whose test labels are withheld).

Folds: the 1,000 training video ids are shuffled (fixed seed) and K disjoint sets of `fold_size`
videos are taken. In fold k those videos are removed from the training data of *every* subject
(as the 102 test videos are new to every subject), all ridge votes are refitted on the rest, and
each subject's trials of the fold's videos (at most `n_eval`, random subset) are ranked against
exactly those videos, with the same similarity() + per-candidate normalisation as submit.py.

Votes (pooled ridge on the trn data of all subjects + per-subject ridge, like submit.py):
  clip           CLIP ridge part of S_image (normalise(pooled + own)); the 9 neural encoders of the
                 submission saw all training videos, so they cannot be used here
  clip_enc       with --encoder: one ShallowConvNet trained per fold on the fold's training data,
                 added to the CLIP ridges like the encoder mean in submit.py
  <extra vote>   every entry of submit.extra_votes (audio, alexnet_conv3, videomae_l8)
Dev (sub-01..04 test, fitted on all training data) gets the same votes plus
  clip_ens9      the submission's S_image (9 encoders + both ridges)
  clip_enc1      checkpoint `all` (shallow, seed 0) + both ridges: counterpart of clip_enc
Test recordings of sub-05/06 are never loaded with labels here.

Caveats: training trials are single presentations at full amplitude, the test recordings look
like averages of ~2 presentations at ~0.65x amplitude, so absolute accuracies differ from test.
Per-split preprocessing statistics (mean response, scale, Euclidean alignment) are label-free and
include the held-out trials.

    python -m src.xsubj.cv_votes run                       # ridge votes, K=3 + dev (~5 min/fold on an idle CPU)
    python -m src.xsubj.cv_votes run --folds 1 2           # resume: only these folds (+ dev; --no-dev to skip)
    python -m src.xsubj.cv_votes run --folds 0 --no-dev    # one fold only
    python -m src.xsubj.cv_votes run --encoder             # + one shallow encoder per fold
    python -m src.xsubj.cv_votes report                    # tables, grid, Train-CV vs dev Spearman
"""
from __future__ import annotations

import argparse
import itertools
import json
import time

import numpy as np

from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import similarity
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.utils import load_config, resolve, seed_everything
from src.xsubj.video.embeddings import VideoBank, _load_file, load_video_bank

SUBJECTS = [f"sub-0{i}" for i in range(1, 7)]
DEV = SUBJECTS[:4]
CANDIDATES = np.arange(1001, 1103).astype(str)


def fold_videos(trn_ids: list[str], k: int, size: int, seed: int) -> list[np.ndarray]:
    ids = np.array(sorted(set().union(*map(set, trn_ids)), key=int))
    perm = np.random.default_rng(seed).permutation(ids)
    assert k * size <= len(perm), "folds would overlap"
    return [perm[i * size:(i + 1) * size] for i in range(k)]


def unit(p):
    return p / np.linalg.norm(p, axis=1, keepdims=True)


class Spaces:
    """CLIP + extra-vote target spaces, each a VideoBank over the same video ids."""

    def __init__(self, cfg):
        self.banks = {"clip": load_video_bank(cfg)}
        for v in cfg["submit"].get("extra_votes", []):
            self.banks[v["name"]] = VideoBank(*_load_file(v["emb"]), standardize=True)

    def target(self, name, ids):
        b = self.banks[name]
        return b.emb[b.lookup(ids)]


def fit_votes(cfg, spaces, train, fs, tmin, log):
    """train: {subject: (X, video_ids)} -> {space: (pooled ridge, {subject: ridge})}."""
    X = np.concatenate([x for x, _ in train.values()])
    ids = np.concatenate([v for _, v in train.values()])
    out = {}
    for name in spaces.banks:
        t0 = time.time()
        pooled = RidgeDecoder(cfg["ridge"], fs, tmin).fit(X, spaces.target(name, ids))
        own = {s: RidgeDecoder(cfg["ridge"], fs, tmin).fit(x, spaces.target(name, v)) for s, (x, v) in train.items()}
        out[name] = (pooled, own)
        log(f"  fitted {name} ({spaces.banks[name].dim}-d) in {time.time() - t0:.0f}s")
    return out


def vote_sims(votes, spaces, subject, X, cand_ids, enc_preds: dict | None = None) -> dict:
    """Same score terms as submit.py: `<vote>` normalised per candidate over the given recordings
    (col_norm, as submit.py), `raw:<vote>` the plain cosine/dot similarity (for fusions without col_norm)."""
    sims = {}
    for name, (pooled, own) in votes.items():
        cand = spaces.target(name, cand_ids)
        p = pooled.predict(X) + own[subject].predict(X)
        terms = {name: unit(p) if name == "clip" else p}
        if name == "clip":
            terms.update({enc_name: unit(p + e) for enc_name, e in (enc_preds or {}).items()})
        for n, q in terms.items():
            sims[n] = similarity(q, cand, True)
            sims[f"raw:{n}"] = similarity(q, cand, False)
    return sims


def train_fold_encoder(cfg, data_trn, bank, keep, fs, log):
    """One ShallowConvNet (submission recipe, seed 0) on the fold's training trials."""
    from src.xsubj.eeg.dataset import Trials
    from src.xsubj.train import fit, split_by_video
    subject_list = sorted(d.subject for d in data_trn)
    trials = Trials(data_trn, bank, subject_list)
    trials = trials.subset(np.isin(trials.video_id, keep))
    tr, va = split_by_video(trials, cfg["train"]["val_frac"], cfg["seed"])
    seed_everything(cfg["seed"])
    model = fit(cfg, tr, va, bank, subject_list, fs, log=lambda *a: None)
    return model, {s: i for i, s in enumerate(subject_list)}


def encode(model, lut, subject, X):
    from src.xsubj.train import embed
    from src.xsubj.utils import get_device
    return embed(model.to(get_device()), X, np.full(len(X), lut.get(subject, -1)), get_device())


def run(args):
    cfg = load_config(args.config)
    seed_everything(cfg["seed"])
    out = resolve(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = lambda *a: print(*a, flush=True)  # noqa: E731
    data = load_preprocessed(cfg)
    trn = {d.subject: d for d in data if d.split == "trn"}
    fs, tmin = data[0].fs, data[0].tmin
    spaces = Spaces(cfg)
    folds = fold_videos([d.video_id for d in trn.values()], args.k, args.fold_size, args.seed)
    meta = {"k": args.k, "fold_size": args.fold_size, "n_eval": args.n_eval, "seed": args.seed,
            "encoder": args.encoder, "folds": [f.tolist() for f in folds]}
    (out / "meta.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(args.seed + 1)

    for k, held in enumerate(folds):
        if args.folds is not None and k not in args.folds:
            continue
        t0 = time.time()
        log(f"fold {k}: {len(held)} videos held out from every subject")
        train = {s: (d.X[~np.isin(d.video_id, held)], d.video_id[~np.isin(d.video_id, held)]) for s, d in trn.items()}
        votes = fit_votes(cfg, spaces, train, fs, tmin, log)
        enc = None
        if args.encoder:
            keep = np.setdiff1d(np.concatenate([d.video_id for d in trn.values()]), held)
            enc = train_fold_encoder(cfg, list(trn.values()), spaces.banks["clip"], keep, fs, log)
            log(f"  trained fold encoder ({time.time() - t0:.0f}s so far)")
        for s, d in trn.items():
            idx = np.flatnonzero(np.isin(d.video_id, held))
            if len(idx) > args.n_eval:
                idx = np.sort(rng.choice(idx, args.n_eval, replace=False))
            X, ids = d.X[idx], d.video_id[idx]  # candidates = exactly the videos of these trials
            enc_preds = {"clip_enc": unit(encode(*enc, s, X))} if enc else None
            sims = vote_sims(votes, spaces, s, X, ids, enc_preds)
            np.savez(out / f"cv_{s}_f{k}.npz", labels=np.arange(len(ids)), video_ids=ids, **sims)
        log(f"  fold {k} done in {time.time() - t0:.0f}s")

    if args.dev:  # fit on all training data, score the dev test splits sub-01..04
        log("dev: fitting on all training data")
        train = {s: (d.X, d.video_id) for s, d in trn.items()}
        votes = fit_votes(cfg, spaces, train, fs, tmin, log)
        enc_paths = {"clip_ens9": cfg["submit"]["encoders"], "clip_enc1": ["all"]}
        encoders = {}
        if not args.no_dev_encoders:
            from src.xsubj.train import load_checkpoint
            for n in cfg["submit"]["encoders"]:
                m, ck = load_checkpoint(resolve("models/xsubj/checkpoints") / n / "model.pt", spaces.banks["clip"].dim)
                encoders[n] = (m, {s: i for i, s in enumerate(ck["subject_list"])})
        for s in DEV:
            d = next(d for d in data if d.subject == s and d.split == "tst")
            assert d.video_id is not None
            enc_preds = {}
            if encoders:
                per = {n: unit(encode(m, lut, s, d.X)) for n, (m, lut) in encoders.items()}
                for key, names in enc_paths.items():
                    enc_preds[key] = unit(np.mean([per[n] for n in names], 0))
            # candidates in recording order of tst_stim -> label of recording i = its candidate column
            cand = CANDIDATES
            sims = vote_sims(votes, spaces, s, d.X, cand, enc_preds)
            labels = np.searchsorted(cand.astype(int), d.video_id.astype(int))
            np.savez(out / f"dev_{s}.npz", labels=labels, video_ids=cand, **sims)
        log("dev done")


# ---------------------------------------------------------------- report

def ranks(S, labels):
    return (S > S[np.arange(len(labels)), labels][:, None]).sum(1)  # 0 = best


def metrics(r):
    return {"top1": float((r == 0).mean()), "top5": float((r < 5).mean()), "rank": float(r.mean() + 1)}


def load_sets(out, k):
    """Folds that are not computed yet are skipped (the run can be resumed with --folds)."""
    cv = {s: [dict(np.load(out / f"cv_{s}_f{i}.npz")) for i in range(k) if (out / f"cv_{s}_f{i}.npz").exists()]
          for s in SUBJECTS}
    dev = {s: dict(np.load(out / f"dev_{s}.npz")) for s in DEV if (out / f"dev_{s}.npz").exists()}
    return cv, dev


def grid_scores(sets, clip_key, extras, grid):
    """sets: list of npz dicts (folds) -> (n_combos, 3) array of top1, top5, mean rank over all trials."""
    W = np.array(grid)                                                  # (C, 1 + n_extra)
    rs = []
    for z in sets:
        V = np.stack([z[clip_key]] + [z[e] for e in extras])           # (1+E, n, m)
        S = np.einsum("cv,vnm->cnm", W, V)
        lab = z["labels"]
        t = S[:, np.arange(len(lab)), lab]
        rs.append((S > t[:, :, None]).sum(2))                          # (C, n)
    r = np.concatenate(rs, 1)
    return np.stack([(r == 0).mean(1), (r < 5).mean(1), r.mean(1) + 1], 1)


def spearman(a, b):
    from scipy.stats import spearmanr
    return float(spearmanr(a, b).statistic)


def fmt_w(w, names):
    return " ".join(f"{n}={x:g}" for n, x in zip(names, w))


def report(args):
    from scipy.stats import rankdata  # noqa: F401  (scipy needed for spearman)
    out = resolve(args.out)
    meta = json.loads((out / "meta.json").read_text())
    cfg = load_config(args.config)
    extras = [v["name"] for v in cfg["submit"]["extra_votes"]]
    cv, dev = load_sets(out, meta["k"])
    if not args.col_norm:  # evaluate the raw similarities under the vote names
        for z in [z for f in cv.values() for z in f] + list(dev.values()):
            z.update({k[4:]: v for k, v in list(z.items()) if k.startswith("raw:")})
    lines = [f"col_norm: {args.col_norm}\n"]
    p = lambda *a: lines.append(" ".join(str(x) for x in a))  # noqa: E731

    # 1) single votes
    votes = [v for v in ["clip", "clip_enc"] + extras if v in cv["sub-01"][0]]
    p("## Single votes, Train-CV (top-1 / top-5 / mean rank; "
      f"{len(cv['sub-01'])}/{meta['k']} folds x <= {meta['n_eval']} trials per subject, candidates = fold videos)\n")
    p("| vote | " + " | ".join(SUBJECTS) + " | mean |")
    p("|---" * (len(SUBJECTS) + 2) + "|")
    for v in votes:
        cells, acc = [], []
        for s in SUBJECTS:
            r = np.concatenate([ranks(z[v], z["labels"]) for z in cv[s]])
            m = metrics(r)
            acc.append(m)
            cells.append(f"{100 * m['top1']:.1f} / {100 * m['top5']:.1f} / {m['rank']:.1f}")
        mm = {k: np.mean([a[k] for a in acc]) for k in acc[0]}
        p(f"| {v} | " + " | ".join(cells) + f" | {100 * mm['top1']:.1f} / {100 * mm['top5']:.1f} / {mm['rank']:.1f} |")
    if dev:
        dvotes = [v for v in ["clip", "clip_enc1", "clip_ens9"] + extras if v in dev["sub-01"]]
        p("\n## Single votes, dev test (fit on all trn, 102 test recordings, 102 candidates)\n")
        p("| vote | " + " | ".join(DEV) + " | mean |")
        p("|---" * (len(DEV) + 2) + "|")
        for v in dvotes:
            ms = [metrics(ranks(dev[s][v], dev[s]["labels"])) for s in DEV]
            mm = {k: np.mean([a[k] for a in ms]) for k in ms[0]}
            p(f"| {v} | " + " | ".join(f"{100 * m['top1']:.1f} / {100 * m['top5']:.1f} / {m['rank']:.1f}" for m in ms)
              + f" | {100 * mm['top1']:.1f} / {100 * mm['top5']:.1f} / {mm['rank']:.1f} |")

    # 2) weight grid
    wc = np.round(np.arange(0, 1.0001, 0.25), 2)
    we = np.round(np.arange(0, 1.5001, 0.25), 2)
    grid = [w for w in itertools.product(wc, *[we] * len(extras)) if any(w)]
    names = ["clip"] + extras
    default = tuple([cfg["submit"]["image_weight"]] + [v["weight"] for v in cfg["submit"]["extra_votes"]])
    if default not in grid:  # e.g. audio 0.3 is off the grid: add it so it can be compared
        grid.append(default)
    di = grid.index(default)
    sel = lambda g: (g[:, 0] + g[:, 1]) / 2  # noqa: E731  (submission selection metric: mean of top-1, top-5)
    p(f"\n## Weight grid: {len(grid)} combos (clip {wc.tolist()}, each extra {we.tolist()}); "
      "selection metric = (top-1 + top-5) / 2, ties -> lower mean rank\n")

    cv_clip_keys = [k for k in ["clip", "clip_enc"] if k in cv["sub-01"][0]]
    G_cv = {(s, ck): grid_scores(cv[s], ck, extras, grid) for s in SUBJECTS for ck in cv_clip_keys}
    dev_clip_keys = [k for k in ["clip", "clip_enc1", "clip_ens9"] if dev and k in dev["sub-01"]]
    G_dev = {(s, dk): grid_scores([dev[s]], dk, extras, grid) for s in dev for dk in dev_clip_keys}

    def best(g):
        return int(np.lexsort((g[:, 2], -sel(g)))[0])

    if dev:
        p("### Spearman over the grid, Train-CV vs dev test, per subject (selection metric | -mean rank)\n")
        p("| Train-CV clip vote -> dev clip vote | " + " | ".join(DEV) + " | mean of 01..04 (pooled scores) |")
        p("|---" * (len(DEV) + 2) + "|")
        pairs = [("clip", "clip")] + ([("clip_enc", "clip_enc1")] if "clip_enc" in cv_clip_keys else []) \
            + [(ck, "clip_ens9") for ck in cv_clip_keys]
        for ck, dk in pairs:
            if dk not in dev_clip_keys:
                continue
            cells = []
            for s in DEV:
                a, b = G_cv[(s, ck)], G_dev[(s, dk)]
                cells.append(f"{spearman(sel(a), sel(b)):+.2f} \\| {spearman(-a[:, 2], -b[:, 2]):+.2f}")
            A = np.mean([G_cv[(s, ck)] for s in DEV], 0)
            B = np.mean([G_dev[(s, dk)] for s in DEV], 0)
            cells.append(f"{spearman(sel(A), sel(B)):+.2f} \\| {spearman(-A[:, 2], -B[:, 2]):+.2f}")
            p(f"| {ck} -> {dk} | " + " | ".join(cells) + " |")
        # cross-subject baseline: how well does dev of one subject predict dev of another?
        dk = "clip_ens9" if "clip_ens9" in dev_clip_keys else dev_clip_keys[0]
        cross = [spearman(sel(G_dev[(a, dk)]), sel(G_dev[(b, dk)])) for a, b in itertools.combinations(DEV, 2)]
        p(f"\nReference: dev-vs-dev Spearman between subjects ({dk}, selection metric): "
          f"mean {np.mean(cross):+.2f} (range {min(cross):+.2f}..{max(cross):+.2f})")
        cvx = [spearman(sel(G_cv[(s, cv_clip_keys[-1])]), sel(G_dev[(t, dk)])) for s in DEV for t in DEV if s != t]
        p(f"Reference: Train-CV of subject A vs dev of another subject B ({cv_clip_keys[-1]} -> {dk}): "
          f"mean {np.mean(cvx):+.2f}")

        p("\n### Does choosing weights on Train-CV pick good dev weights? "
          f"(dev score with {dk}: top-1 / top-5 / mean rank, mean over sub-01..04)\n")
        p("| chosen by | weights | dev top-1 | dev top-5 | dev rank |")
        p("|---|---|---|---|---|")
        Dm = np.mean([G_dev[(s, dk)] for s in DEV], 0)
        def row(label, i_per_subject):
            vals = np.mean([G_dev[(s, dk)][i_per_subject[s]] for s in DEV], 0)
            ws = {s: grid[i_per_subject[s]] for s in DEV}
            wtxt = fmt_w(ws[DEV[0]], names) if len(set(ws.values())) == 1 else "per subject"
            p(f"| {label} | {wtxt} | {100 * vals[0]:.1f} | {100 * vals[1]:.1f} | {vals[2]:.1f} |")
        if di is not None:
            row("submission default (dev-tuned)", {s: di for s in DEV})
        row("dev oracle (best mean over 01..04)", {s: best(Dm) for s in DEV})
        for ck in cv_clip_keys:
            row(f"Train-CV per subject ({ck})", {s: best(G_cv[(s, ck)]) for s in DEV})
            A6 = np.mean([G_cv[(s, ck)] for s in SUBJECTS], 0)
            row(f"Train-CV pooled 6 subjects ({ck})", {s: best(A6) for s in DEV})
        row("clip only", {s: grid.index(tuple([1.0] + [0.0] * len(extras))) for s in DEV})

    p("\n### Best weights according to Train-CV (CV top-1 / top-5 / mean rank at that point)\n")
    p("| clip vote | subject | best weights | CV score | CV score at submission default |")
    p("|---|---|---|---|---|")
    for ck in cv_clip_keys:
        for s in SUBJECTS + ["pooled"]:
            g = np.mean([G_cv[(t, ck)] for t in SUBJECTS], 0) if s == "pooled" else G_cv[(s, ck)]
            i = best(g)
            d = f"{100 * g[di, 0]:.1f} / {100 * g[di, 1]:.1f} / {g[di, 2]:.1f}" if di is not None else "-"
            p(f"| {ck} | {s} | {fmt_w(grid[i], names)} | {100 * g[i, 0]:.1f} / {100 * g[i, 1]:.1f} / {g[i, 2]:.1f} | {d} |")
    # top-10 for 05/06: is the optimum sharp or flat?
    ck = cv_clip_keys[-1]
    for s in ["sub-05", "sub-06"]:
        g = G_cv[(s, ck)]
        order = np.lexsort((g[:, 2], -sel(g)))[:10]
        p(f"\nTop-10 for {s} ({ck}): " + "; ".join(f"[{fmt_w(grid[i], names)}] {100 * sel(g)[i]:.1f}" for i in order))
    text = "\n".join(lines)
    tag = "" if args.col_norm else "_nocolnorm"
    (out / f"report{tag}.md").write_text(text, encoding="utf-8")
    np.savez(out / f"grid{tag}.npz", grid=np.array(grid), names=np.array(names),
             **{f"cv/{s}/{ck}": v for (s, ck), v in G_cv.items()},
             **{f"dev/{s}/{dk}": v for (s, dk), v in G_dev.items()})
    print(text)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["run", "report"])
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--out", default="models/xsubj/cv")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--fold-size", type=int, default=120,
                    help="videos per fold; each subject saw 85-97%% of them, so ~102 trials remain")
    ap.add_argument("--n-eval", type=int, default=102)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, nargs="+", default=None,
                    help="run only these folds (default all); fold sets depend only on --k/--fold-size/--seed")
    ap.add_argument("--encoder", action="store_true", help="also train one shallow encoder per fold")
    ap.add_argument("--dev", action=argparse.BooleanOptionalAction, default=True,
                    help="also write dev-test score matrices for sub-01..04")
    ap.add_argument("--no-dev-encoders", action="store_true")
    ap.add_argument("--col-norm", action=argparse.BooleanOptionalAction, default=True,
                    help="report: use the per-candidate normalised similarities (submit.py) or the raw ones")
    args = ap.parse_args()
    run(args) if args.mode == "run" else report(args)


if __name__ == "__main__":
    main()
