"""Nested LOO comparison of fusion structures that include Daniel's latest ensemble.

Daniel's submission (branch feature/clip-lowlevel-forward-model, commit 9d62e5b: CLIP image
ensemble of 9 encoders + 2 ridges, plus ridge votes into AlexNet conv3, VideoMAE layer 8 and
CLAP audio) overlaps with our members 3 (his older CLIP encoder) and 12 (VideoMAE-l8 ridges).
Its rankings enter as member tag _dnew (src.xsubj_member --src <his submit output> --tag _dnew).

Variants, each with Sinkhorn (tau chosen together with the weights):
  A  current fusion: members 1-7, 9-12
  B  substitution: members 3, 11, 12 replaced by Daniel's ensemble
  C  member 3 replaced by Daniel's ensemble (11, 12 kept)
  D  blend: current fusion and Daniel's ensemble, Daniel's share of the total weight searched

For each held-out dev subject all free weights and tau are chosen on the other three (mean
top-1, ties by top-5) and scored on the held-out one; the submission's weights are the same
choice on all four (printed per variant). Paired tests against A and against
Daniel alone on the same 408 recordings. Daniel's own weights were fitted on these four
subjects, so every variant containing him is optimistic by that much.

    python -m src.loo_fusion_variants
"""

from __future__ import annotations

import itertools
from pathlib import Path

import numpy as np
from scipy import stats

from .loo_new_members import SUBS, old_members
from .sinkhorn import mean_rank_matrix, rank

TAUS = (2, 3, 5)
TST, PROBE, ED, NEW = Path("work/tst_ranked"), Path("work/xsubj_probe"), Path("work/ed_5e3154e"), Path("work/prep_variants")


def true_rank(ranked, y):
    return np.argmax(ranked == y[:, None], axis=1) + 1


def main():
    Y = {s: np.load(TST / f"truth_{s}_s0.npy") for s in SUBS}
    base = {s: old_members(s, TST, PROBE, "emb_train__clip-vitb32_s0", ED) for s in SUBS}   # clip alex ED ridge tr tr tr
    ext = {s: {t: np.load(NEW / f"A{t}_{s}_s0.npy") for t in ("_fc6", "_vmae8", "_dfc6", "_dvmae8", "_dnew")} for s in SUBS}
    G = (0, 3, 5, 9)

    def members(s, ed=True, d11_12=True, dnew=False):
        m = [base[s][0], base[s][1]] + ([base[s][2]] if ed else []) + base[s][3:]
        m += [ext[s]["_fc6"], ext[s]["_vmae8"]]
        m += [ext[s]["_dfc6"], ext[s]["_dvmae8"]] if d11_12 else []
        return m + ([ext[s]["_dnew"]] if dnew else [])

    variants = {
        "A current (1-7, 9-12)": (dict(),
            [(9, a, 5, 2, 1, 1, 1, f, v, d1, d2) for a in (9, 0) for f, v, d1, d2 in itertools.product(G, repeat=4)]),
        "B 3, 11, 12 -> Daniel": (dict(ed=False, d11_12=False, dnew=True),
            [(9, a, 2, 1, 1, 1, f, v, dn) for a in (9, 0) for f, v in itertools.product(G, repeat=2)
             for dn in (5, 9, 15, 20, 30, 45, 60)]),
        "C 3 -> Daniel": (dict(ed=False, dnew=True),
            [(9, a, 2, 1, 1, 1, f, v, d1, d2, dn) for a in (9, 0) for f, v, d1, d2 in itertools.product((3, 5, 9), repeat=4)
             for dn in (9, 20, 45)]),
        "D blend with Daniel": (dict(dnew=True),
            [(9, 9, 5, 2, 1, 1, 1, 5, 9, 9, 9, round(60 * sh / (1 - sh), 1)) for sh in (0.25, 0.4, 0.5, 0.6, 0.75)]),
    }

    def ranks(s, kw, w, tau):
        return true_rank(rank(-mean_rank_matrix(members(s, **kw), list(w)), tau), Y[s])

    results = {}
    for name, (kw, grid) in variants.items():
        cache = {}

        def r(s, w, tau):
            k = (s, w, tau)
            if k not in cache:
                cache[k] = ranks(s, kw, w, tau)
            return cache[k]

        held = []
        for h in SUBS:
            dev = [s for s in SUBS if s != h]
            w, tau = max(((w, t) for w in grid for t in TAUS),
                         key=lambda o: (np.mean([np.mean(r(s, *o) == 1) for s in dev]),
                                        np.mean([np.mean(r(s, *o) <= 5) for s in dev])))
            held.append(r(h, w, tau))
            print(f"[{name}] hold {h}: weights {':'.join(str(x) for x in w)}, tau {tau}", flush=True)
        results[name] = np.concatenate(held)
        # the submission's weights: the same choice on all four dev subjects
        w, tau = max(((w, t) for w in grid for t in TAUS),
                     key=lambda o: (np.mean([np.mean(r(s, *o) == 1) for s in SUBS]),
                                    np.mean([np.mean(r(s, *o) <= 5) for s in SUBS])))
        print(f"[{name}] chosen on all four (for the submission): weights {':'.join(str(x) for x in w)}, tau {tau}", flush=True)

    dan = np.concatenate([true_rank(ext[s]["_dnew"], Y[s]) for s in SUBS])
    results["Daniel alone (9d62e5b)"] = dan
    print("\nLOO held-out, sub-01..04 (408 recordings):")
    for k, x in results.items():
        print(f"  {k:26s} top-1 {np.mean(x == 1) * 100:5.2f}  top-5 {np.mean(x <= 5) * 100:5.2f}  mean rank {x.mean():.2f}")
    print("\npaired:")
    a = results["A current (1-7, 9-12)"]
    for k, b in results.items():
        for ref_name, ref in (("A", a), ("Daniel", dan)):
            if b is ref:
                continue
            line = []
            for kk in (1, 5):
                g, l = int(np.sum((b <= kk) & (ref > kk))), int(np.sum((ref <= kk) & (b > kk)))
                line.append(f"top-{kk} +{g}/-{l} p={stats.binomtest(g, g + l).pvalue if g + l else 1:.3f}")
            pw = stats.wilcoxon(b, ref).pvalue if np.any(b != ref) else 1.0
            print(f"  {k:26s} vs {ref_name:6s}: " + "; ".join(line) + f"; rank p={pw:.2g}")


if __name__ == "__main__":
    main()
