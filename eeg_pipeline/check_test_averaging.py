"""
Are the test trials averages of several repetitions? (QC follow-up)

QC found test trials at 0.60-0.70x the amplitude of training trials. Two explanations:

  averaging   test trial = mean of k repetitions: noise shrinks, the stimulus response does not
  gain        test data simply scaled down: signal and noise shrink together

Test: correlate two subjects' responses to the SAME video, minus their correlation for
different videos (= the shared stimulus-locked part). Correlation ignores scale, so

  gain        -> r_test / r_train ~ 1
  averaging   -> r_test / r_train ~ 1 / sqrt(R_a * R_b),   R = test/train variance ratio

The test labels of sub-01..04 are used only to pair trials of the same video across
subjects, never for preprocessing or modelling. Assumes the shared response to a video is
about as strong for the test videos as for the training videos.

    python -m eeg_pipeline.check_test_averaging --data_dir data
"""
import argparse
import itertools

import numpy as np
from scipy import signal

from . import preprocessing as pp

FS_OUT = 100
WINDOW = (0.05, 0.6)          # s after onset, where the stimulus information is
BANDS = {'delta 1-4': (1, 4), 'theta 4-8': (4, 8), 'alpha 8-13': (8, 13),
         'beta 13-30': (13, 30), 'gamma 30-45': (30, 45)}


def prep(X, window=WINDOW):
    """Per-trial demeaning, 500 -> 100 Hz, remove the split's mean response, crop."""
    X = X - X.mean(axis=-1, keepdims=True)
    X = signal.decimate(X, pp.SFREQ // FS_OUT, axis=-1, ftype='fir', zero_phase=True)
    X = X - X.mean(axis=0, keepdims=True)
    a, b = int(window[0] * FS_OUT), int(window[1] * FS_OUT)
    return X[..., a:b]


def corr_matrix(A, B):
    """Correlation of every trial in A with every trial in B (rows aligned by video)."""
    a = A.reshape(len(A), -1)
    b = B.reshape(len(B), -1)
    a = (a - a.mean(1, keepdims=True)) / a.std(1, keepdims=True)
    b = (b - b.mean(1, keepdims=True)) / b.std(1, keepdims=True)
    return a @ b.T / a.shape[1]


def shared_signal(C, idx=None):
    """Mean same-video correlation minus mean different-video correlation.
    idx: bootstrap sample of video positions; pairs of the same video never count as 'different'."""
    if idx is None:
        idx = np.arange(len(C))
    sub = C[np.ix_(idx, idx)]
    same_video = idx[:, None] == idx[None, :]
    return np.diag(sub).mean() - sub[~same_video].mean()


def band_ratios(Xtr, Xte):
    """Test/train power ratio per frequency band (whole 0-3 s, after per-trial demeaning)."""
    f, ptr = signal.welch(Xtr - Xtr.mean(-1, keepdims=True), fs=pp.SFREQ, nperseg=500, axis=-1)
    _, pte = signal.welch(Xte - Xte.mean(-1, keepdims=True), fs=pp.SFREQ, nperseg=500, axis=-1)
    ptr, pte = ptr.mean((0, 1)), pte.mean((0, 1))
    return {name: pte[(f >= lo) & (f < hi)].mean() / ptr[(f >= lo) & (f < hi)].mean()
            for name, (lo, hi) in BANDS.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args(argv)

    trn, tst, var_ratio = {}, {}, {}
    print('Variance ratio test/train (window 50-600 ms) and per-band power ratios (0-3 s)')
    for sub in range(1, 7):
        ids, X = pp.load_train(args.data_dir, sub)
        Xte = pp.load_test(args.data_dir, sub)
        labels = pp.load_test_labels(args.data_dir, sub)
        trn[sub] = dict(zip(ids, prep(X)))
        P = prep(Xte)
        if labels is not None:
            tst[sub] = dict(zip(labels, P))
        var_ratio[sub] = P.var() / np.stack(list(trn[sub].values())).var()
        bands = band_ratios(X, Xte)
        print(f'  sub-{sub:02d}  R = {var_ratio[sub]:.3f} (amplitude {np.sqrt(var_ratio[sub]):.2f}x, '
              f'k ~ {1 / var_ratio[sub]:.1f} if noise-dominated)  bands: '
              + '  '.join(f'{k.split()[0]} {v:.2f}' for k, v in bands.items()))

    print('\nShared same-video signal between subject pairs (correlation, same - different video)')
    print('  pair     r_train   r_test   observed ratio   predicted: averaging / gain')
    obs, pred, r_tr_pairs, C_te = [], [], [], {}
    pairs = list(itertools.combinations(sorted(tst), 2))
    common = sorted(set.intersection(*(set(tst[s]) for s in tst)))
    for a, b in pairs:
        v_tr = sorted(set(trn[a]) & set(trn[b]))
        r_tr = shared_signal(corr_matrix(np.stack([trn[a][v] for v in v_tr]),
                                         np.stack([trn[b][v] for v in v_tr])))
        C_te[a, b] = corr_matrix(np.stack([tst[a][v] for v in common]),
                                 np.stack([tst[b][v] for v in common]))
        r_te = shared_signal(C_te[a, b])
        p = 1 / np.sqrt(var_ratio[a] * var_ratio[b])
        obs.append(r_te / r_tr)
        pred.append(p)
        r_tr_pairs.append(r_tr)
        print(f'  {a:02d}-{b:02d}    {r_tr:.4f}    {r_te:.4f}     {r_te / r_tr:5.2f}           '
              f'{p:.2f} / 1.00   ({len(v_tr)} train, {len(common)} test videos)')

    # bootstrap over test videos for the mean ratio (training side has ~900 videos: held fixed)
    rng = np.random.default_rng(0)
    boots = [np.mean([shared_signal(C_te[pair], idx) / r for pair, r in zip(pairs, r_tr_pairs)])
             for idx in (rng.integers(0, len(common), len(common)) for _ in range(args.n_boot))]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f'\nMean observed ratio {np.mean(obs):.2f}  (95% bootstrap CI over test videos {lo:.2f}-{hi:.2f})')
    print(f'Predicted if averaging {np.mean(pred):.2f}, if gain 1.00')


if __name__ == '__main__':
    main()
