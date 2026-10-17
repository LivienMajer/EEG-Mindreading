"""
Do the channel names from the organizers follow the row order of the data files?

    python -m eeg_pipeline.check_channel_order --data_dir data

Neighbouring electrodes pick up similar signals, so if the order is right, the correlation
between two channels (training trials of all subjects) falls with their distance on the
scalp (standard 10-05 positions from MNE). Compared against 5000 random orderings.
"""
import argparse

import numpy as np
from scipy import stats

from . import preprocessing as pp


def positions(names):
    import mne
    pos = mne.channels.make_standard_montage('standard_1005').get_positions()['ch_pos']
    return np.array([pos[n] for n in names])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--n_perm', type=int, default=5000)
    args = ap.parse_args(argv)

    names = pp.CHANNELS
    C = []
    for sub in range(1, 7):
        _, X = pp.load_train(args.data_dir, sub)
        X = X - X.mean(-1, keepdims=True)
        C.append(np.corrcoef(X.transpose(1, 0, 2).reshape(len(names), -1)))
    C = np.mean(C, 0)
    P = positions(names)
    D = np.linalg.norm(P[:, None] - P[None], axis=-1)
    iu = np.triu_indices(len(names), 1)

    rho = stats.spearmanr(-D[iu], C[iu]).statistic
    rng = np.random.default_rng(0)
    null = np.array([stats.spearmanr(-D[iu], C[np.ix_(p, p)][iu]).statistic
                     for p in (rng.permutation(len(names)) for _ in range(args.n_perm))])
    p = (1 + np.sum(null >= rho)) / (1 + args.n_perm)
    print(f'channel correlation vs closeness on the scalp: rho = {rho:.3f}')
    print(f'random orderings: mean {null.mean():.3f}, max {null.max():.3f}, p = {p:.4f}')
    for n in ('O1', 'Oz', 'O2', 'P7', 'P8'):
        i = names.index(n)
        print(f'  {n}: most correlated with {[names[j] for j in np.argsort(-C[i])[1:4]]}, '
              f'nearest on the scalp {[names[j] for j in np.argsort(D[i])[1:4]]}')


if __name__ == '__main__':
    main()
