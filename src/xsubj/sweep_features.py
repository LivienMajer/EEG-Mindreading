"""EEG feature sweep for the ridge decoder (official protocol: train on all trn splits,
score the labeled tst splits among 102 candidates). Mean rank is the main criterion; it is
far less noisy than top-1 with 4 x 102 test recordings.

    python -m src.xsubj.sweep_features                          # feature variants, 0-1 s data
    python -m src.xsubj.sweep_features --tmax 3.0               # + variants that need 0-3 s
    python -m src.xsubj.sweep_features --tmax 3.0 --curve       # temporal decoding curve
"""
from __future__ import annotations

import argparse
import copy
import json
import time

import numpy as np

from src.xsubj.eeg.dataset import Trials
from src.xsubj.eeg.preprocess import load_preprocessed
from src.xsubj.metrics import retrieval_metrics
from src.xsubj.nets.ridge import RidgeDecoder
from src.xsubj.utils import load_config, resolve
from src.xsubj.video.embeddings import load_video_bank

FB = [[1, 4], [4, 8], [8, 13], [13, 30]]
# name: (overrides of the `ridge` config section, minimum data tmax in s)
VARIANTS = {
    "erp50 0-1s + bandpower (current)": ({}, 1.0),
    "erp50 0-1s": ({"bandpower": False}, 1.0),
    "bandpower only": ({"erp": False}, 1.0),
    "erp25 0-1s": ({"erp_bin": 0.025, "bandpower": False}, 1.0),
    "erp20 0-1s": ({"erp_bin": 0.02, "bandpower": False}, 1.0),
    "erp10 0-1s (raw 100 Hz)": ({"erp_bin": 0.01, "bandpower": False}, 1.0),
    "erp10 50-600ms (organiser window)": ({"erp_bin": 0.01, "erp_tmin": 0.05, "erp_tmax": 0.6, "bandpower": False}, 1.0),
    "erp10 50-500ms": ({"erp_bin": 0.01, "erp_tmin": 0.05, "erp_tmax": 0.5, "bandpower": False}, 1.0),
    "erp10 0-700ms": ({"erp_bin": 0.01, "erp_tmin": 0.0, "erp_tmax": 0.7, "bandpower": False}, 1.0),
    "erp20 50-600ms": ({"erp_bin": 0.02, "erp_tmin": 0.05, "erp_tmax": 0.6, "bandpower": False}, 1.0),
    "erp50 50-600ms": ({"erp_tmin": 0.05, "erp_tmax": 0.6, "bandpower": False}, 1.0),
    "erp20 0-400ms": ({"erp_bin": 0.02, "erp_tmax": 0.4, "bandpower": False}, 1.0),
    "riemann broadband": ({"erp": False, "bandpower": False, "riemann": True}, 1.0),
    "riemann filter-bank": ({"erp": False, "bandpower": False, "riemann": True, "riemann_bands": FB}, 1.0),
    "erp20 0-1s + riemann filter-bank": ({"erp_bin": 0.02, "bandpower": False, "riemann": True, "riemann_bands": FB}, 1.0),
    "erp50 0-3s": ({"erp_tmax": 3.0, "bandpower": False}, 3.0),
    "erp50 0-3s + bandpower": ({"erp_tmax": 3.0}, 3.0),
    "erp100 0-3s": ({"erp_bin": 0.1, "erp_tmax": 3.0, "bandpower": False}, 3.0),
    "erp20 0-1s + erp100 1-3s": (None, 3.0),  # built in code: fine early, coarse late
}


def score(cfg, ridge_cfg, data, bank) -> dict:
    trn = [d for d in data if d.split == "trn"]
    tst = [d for d in data if d.split == "tst" and d.video_id is not None]
    tr = Trials(trn, bank, [])
    model = RidgeDecoder(ridge_cfg, data[0].fs, data[0].tmin).fit(tr.X, bank.emb[tr.y])
    per = []
    for d in tst:
        te = Trials([d], bank, [])
        per.append(retrieval_metrics(model.predict(te.X), te.y, bank.emb, ["all"]))
    return {"top1": float(np.mean([p["top1@all"] for p in per])),
            "top5": float(np.mean([p["top5@all"] for p in per])),
            "mean_rank": float(np.mean([p["mean_rank@all"] for p in per])),
            "alpha": float(model.model[-1].alpha_),
            "per_subject_top1": [p["top1@all"] for p in per]}


class TwoScale(RidgeDecoder):
    """erp20 over 0-1 s plus erp100 over 1-3 s."""

    def _feat(self, X):
        from src.xsubj.nets.ridge import erp_bins
        return np.concatenate([erp_bins(X, self.fs, self.tmin, 0.02, 0.0, 1.0),
                               erp_bins(X, self.fs, self.tmin, 0.1, 1.0, 3.0)], 1)

    def fit(self, X, target_emb):
        self.model.fit(self._feat(X), target_emb)
        return self

    def predict(self, X):
        p = self.model.predict(self._feat(X))
        return p / np.linalg.norm(p, axis=1, keepdims=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/xsubj.yaml")
    ap.add_argument("--tmax", type=float, default=None, help="override preprocess.tmax (data window)")
    ap.add_argument("--curve", action="store_true", help="temporal decoding curve instead of variants")
    ap.add_argument("--only", nargs="*", default=None, help="substrings of variant names to run")
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.tmax:
        cfg["preprocess"]["tmax"] = args.tmax
    data_tmax = cfg["preprocess"]["tmax"]
    cfg["ridge"]["alphas"] = [10.0 ** k for k in range(-1, 8)]
    data = load_preprocessed(cfg)
    bank = load_video_bank(cfg)
    rows = []

    if args.curve:  # 100 ms windows, 50 ms steps, two 50 ms bins each
        print(f"{'window':>12}  {'top5':>6}  {'mean rank':>9}")
        for t0 in np.arange(0.0, data_tmax - 0.1 + 1e-9, 0.05):
            r = {**cfg["ridge"], "erp_bin": 0.05, "erp_tmin": float(t0), "erp_tmax": float(t0) + 0.1, "bandpower": False}
            s = score(cfg, r, data, bank)
            rows.append({"t0": float(t0), **s})
            print(f"{t0 * 1000:5.0f}-{(t0 + 0.1) * 1000:4.0f}ms  {s['top5']:6.3f}  {s['mean_rank']:9.1f}", flush=True)
    else:
        print(f"{'variant':38s} {'top1':>6} {'top5':>6} {'mean rank':>9} {'alpha':>8}")
        for name, (over, need) in VARIANTS.items():
            if need > data_tmax + 1e-9 or (args.only and not any(o in name for o in args.only)):
                continue
            t = time.time()
            if over is None:
                trn = [d for d in data if d.split == "trn"]
                tr = Trials(trn, bank, [])
                m = TwoScale(cfg["ridge"], data[0].fs, data[0].tmin).fit(tr.X, bank.emb[tr.y])
                per = [retrieval_metrics(m.predict(Trials([d], bank, []).X), Trials([d], bank, []).y, bank.emb, ["all"])
                       for d in data if d.split == "tst" and d.video_id is not None]
                s = {"top1": float(np.mean([p["top1@all"] for p in per])),
                     "top5": float(np.mean([p["top5@all"] for p in per])),
                     "mean_rank": float(np.mean([p["mean_rank@all"] for p in per])),
                     "alpha": float(m.model[-1].alpha_), "per_subject_top1": [p["top1@all"] for p in per]}
            else:
                s = score(cfg, {**copy.deepcopy(cfg["ridge"]), **over}, data, bank)
            rows.append({"variant": name, "data_tmax": data_tmax, **s})
            print(f"{name:38s} {s['top1']:6.3f} {s['top5']:6.3f} {s['mean_rank']:9.1f} {s['alpha']:8.0e}"
                  f"   ({time.time() - t:.0f}s)", flush=True)
        print(f"{'chance':38s} {1 / 102:6.3f} {5 / 102:6.3f} {51.5:9.1f}")

    out = resolve("models/xsubj/results") / f"sweep_{'curve' if args.curve else 'features'}_tmax{data_tmax}_{time.strftime('%H%M%S')}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    main()
