"""
EEG preprocessing 

Manual use:

1) Command line, fixed settings
       python -m eeg_pipeline.preprocessing --data_dir <folder with trn/ tst/ tst_stim/> --out_dir preprocessed \\
              --baseline trial --normalize split

2) From Python (e.g. in the encoder notebook)
       from eeg_pipeline import load_preprocessed
       d = load_preprocessed('preprocessed', 1)
       d['X_train']      # [n_train, channels, time]
       d['train_ids']    # video id of each training trial
       d['X_test']       # [102, channels, time], ORIGINAL order (needed for submission)
       d['test_labels']  # true video ids, only for sub-01..04 (validation only!)
       d['times'], d['channels'], d['config']

Steps, in order (every step can be switched off):
    0. reference  optional common average reference (subtract the mean over channels
                  at every sample). 
    1. baseline   subtract each trial's channel mean ('trial', default), or the mean
                  over 0-50 ms ('onset'). 'detrend' removes each trial's
                  linear slow drift instead (slow <1 Hz activity dominates here).
    2. filter     zero-phase Butterworth band-pass (default: low-pass 40 Hz only) and an
                  optional notch (50 Hz).
    3. resample   500 Hz -> 100 Hz (default) with an anti-aliasing FIR filter, as in
                  the organizers' baseline.
    4. crop       keep tmin..tmax seconds (default: the whole 0-3 s).
    5. channels   drop listed channels, or the ones the QC marked flat/clipping
                  (the same channels are dropped for every subject, so models and
                  predictions stay comparable across subjects).
    6. trials     optionally drop training trials flagged by the QC. Test trials are
                  NEVER dropped or reordered.
    7. normalize  per-channel z-score with statistics from the training trials
                  (default). 'split' fits the statistics on test trials separately
                  (unsupervised normalization of test data is allowed by the rules);
                  'trial' z-scores every trial on its own.
"""

import argparse
import glob
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

SFREQ = 500
# Rows of every trial, from the organizers; order verified with check_channel_order.
# All parietal/occipital (10-10 system). Organizer preprocessing before release:
# band-pass 0.1-40 Hz, blink components removed, 1000 -> 500 Hz.
CHANNELS = ['Pz', 'P3', 'P7', 'O1', 'Oz', 'O2', 'P4', 'P8', 'P1', 'P5',
            'PO7', 'PO3', 'POz', 'PO4', 'PO8', 'P6', 'P2']

DEFAULT_CONFIG = {
    'sfreq': SFREQ,
    'reference': None,            # 'average' = common average reference, or None
    'baseline': 'trial',          # 'trial' (mean), 'onset' (0-50 ms), 'detrend', or None
    'baseline_window': [0.0, 0.05],
    'l_freq': None,               # Hz high-pass, None = off
    'h_freq': 40.0,               # Hz low-pass, None = off
    'filter_order': 4,
    'notch': None,                # e.g. 50
    'sfreq_out': 100,             # must divide sfreq; None = keep 500 Hz
    'tmin': 0.0,
    'tmax': 3.0,
    'drop_channels': [],          # channel indices
    'auto_drop_from_qc': False,   # drop channels the QC flagged as flat / clipping
    'reject_train_trials': False, # drop training trials the QC flagged
    'normalize': 'train',         # 'train', 'split', 'trial', or None
}


# ─────────────────────────────────────────────
# Loading
# ─────────────────────────────────────────────

def _find(data_dir, folder, sub, suffix):
    data_dir = Path(data_dir)
    for pattern in (f'sub-{sub:02d}_{suffix}', f'sub{sub:02d}_{suffix}'):
        hits = sorted(glob.glob(str(data_dir / folder / pattern)))
        if hits:
            return hits[0]
    return None


def load_train(data_dir, sub):
    """Training trials sorted by video id. Returns (ids [n], X [n, 17, 1500])."""
    path = _find(data_dir, 'trn', sub, 'trn.npz')
    if path is None:
        raise FileNotFoundError(f'no training file for sub-{sub:02d} in {data_dir}/trn')
    z = np.load(path)
    keys = sorted(z.files, key=int)
    ids = np.array([int(k) for k in keys])
    X = np.stack([z[k] for k in keys]).astype(np.float32)
    return ids, X


def load_test(data_dir, sub):
    """Test trials in the ORIGINAL file order. Returns X [102, 17, 1500]."""
    path = _find(data_dir, 'tst', sub, 'tst.npy')
    if path is None:
        raise FileNotFoundError(f'no test file for sub-{sub:02d} in {data_dir}/tst')
    return np.load(path).astype(np.float32)


def load_test_labels(data_dir, sub):
    """True video ids of the test trials (sub-01..04 only), else None."""
    path = _find(data_dir, 'tst_stim', sub, 'tst_stim.npy')
    return np.load(path).astype(int) if path else None


# ─────────────────────────────────────────────
# Processing steps
# ─────────────────────────────────────────────

def apply_baseline(X, cfg):
    mode = cfg.get('baseline')
    if mode is None:
        return X
    if mode == 'trial':
        return X - X.mean(axis=-1, keepdims=True)
    if mode == 'detrend':                 # remove each trial's linear slow drift
        return signal.detrend(X, axis=-1, type='linear')
    if mode == 'onset':
        a = int(cfg['baseline_window'][0] * cfg['sfreq'])
        b = max(int(cfg['baseline_window'][1] * cfg['sfreq']), a + 1)
        return X - X[..., a:b].mean(axis=-1, keepdims=True)
    raise ValueError(f'unknown baseline mode {mode!r}')


def apply_filters(X, cfg):
    fs = cfg['sfreq']
    lo, hi, order = cfg.get('l_freq'), cfg.get('h_freq'), cfg.get('filter_order', 4)
    padlen = min(X.shape[-1] - 1, fs)    # reflect-pad ~1 s to limit edge effects
    if cfg.get('notch'):
        b, a = signal.iirnotch(cfg['notch'], Q=30, fs=fs)
        X = signal.filtfilt(b, a, X, axis=-1, padlen=padlen)
    if lo and hi:
        sos = signal.butter(order, [lo, hi], btype='bandpass', fs=fs, output='sos')
    elif lo:
        sos = signal.butter(order, lo, btype='highpass', fs=fs, output='sos')
    elif hi:
        sos = signal.butter(order, hi, btype='lowpass', fs=fs, output='sos')
    else:
        return X
    return signal.sosfiltfilt(sos, X, axis=-1, padlen=padlen)


def apply_resample(X, cfg):
    fs, out = cfg['sfreq'], cfg.get('sfreq_out')
    if not out or out == fs:
        return X, fs
    if fs % out:
        raise ValueError(f'sfreq_out ({out}) must divide sfreq ({fs})')
    return signal.decimate(X, fs // out, axis=-1, ftype='fir', zero_phase=True), out


def apply_crop(X, fs, cfg):
    a = int(round(cfg['tmin'] * fs))
    b = int(round(cfg['tmax'] * fs))
    b = min(b, X.shape[-1])
    return X[..., a:b], np.arange(a, b) / fs


def apply_reference(X, cfg):
    """Common average reference: subtract the mean over channels at every sample."""
    if cfg.get('reference') == 'average':
        return X - X.mean(axis=1, keepdims=True)
    return X


def preprocess_epochs(X, cfg):
    """Stateless steps 0-4 (same operation on every trial, train or test)."""
    X = apply_reference(X, cfg)
    X = apply_baseline(X, cfg)
    X = apply_filters(X, cfg)
    X, fs = apply_resample(X, cfg)
    X, times = apply_crop(X, fs, cfg)
    return X.astype(np.float32), times


class ChannelNormalizer:
    """Per-channel z-score; statistics are fitted on the data passed to fit()."""

    def fit(self, X):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True) + 1e-8
        return self

    def transform(self, X):
        return ((X - self.mean_) / self.std_).astype(np.float32)


def normalize(X_train, X_test, mode):
    if mode is None:
        return X_train, X_test
    if mode == 'train':
        n = ChannelNormalizer().fit(X_train)
        return n.transform(X_train), n.transform(X_test)
    if mode == 'split':
        return (ChannelNormalizer().fit(X_train).transform(X_train),
                ChannelNormalizer().fit(X_test).transform(X_test))
    if mode == 'trial':
        z = lambda X: ((X - X.mean(-1, keepdims=True))
                       / (X.std(-1, keepdims=True) + 1e-8)).astype(np.float32)
        return z(X_train), z(X_test)
    raise ValueError(f'unknown normalize mode {mode!r}')


# ─────────────────────────────────────────────
# QC integration
# ─────────────────────────────────────────────

def qc_bad_channels(qc_dir, sub, flags=('flat', 'clipping')):
    path = Path(qc_dir) / f'sub-{sub:02d}' / 'qc_channels.csv'
    if not path.exists():
        return []
    df = pd.read_csv(path).fillna('')
    return [i for i, f in enumerate(df['flags'])
            if any(flag in str(f).split(';') for flag in flags)]


def qc_bad_train_trials(qc_dir, sub):
    path = Path(qc_dir) / f'sub-{sub:02d}' / 'qc_trials.csv'
    if not path.exists():
        return []
    df = pd.read_csv(path)
    bad = df[(df['split'] == 'train') & (df['reject_suggested'])]
    return bad['index_in_file'].astype(int).tolist()


# ─────────────────────────────────────────────
# Whole subject
# ─────────────────────────────────────────────

def prepare_subject(data_dir, sub, cfg=None, qc_dir=None, drop_channels=None,
                    reject_train_idx=None):
    """
    Load and preprocess one subject. Returns a dict (see module docstring).
    `drop_channels` overrides cfg['drop_channels'] (used to apply one list to all).
    `reject_train_idx`: positions (in video-id order) of training trials to drop;
    overrides the QC-folder lookup. Test trials are never dropped.
    """
    cfg = {**DEFAULT_CONFIG, **(cfg or {})}
    train_ids, X_train = load_train(data_dir, sub)
    X_test = load_test(data_dir, sub)
    labels = load_test_labels(data_dir, sub)

    X_train, times = preprocess_epochs(X_train, cfg)
    X_test, _ = preprocess_epochs(X_test, cfg)

    n_ch = X_train.shape[1]
    drop = sorted(set(drop_channels if drop_channels is not None else cfg['drop_channels']))
    keep = [c for c in range(n_ch) if c not in drop]
    X_train, X_test = X_train[:, keep], X_test[:, keep]

    n_rejected = 0
    bad = None
    if reject_train_idx is not None:
        bad = set(int(i) for i in reject_train_idx)
    elif cfg['reject_train_trials'] and qc_dir:
        bad = set(qc_bad_train_trials(qc_dir, sub))
    if bad:
        mask = np.array([i not in bad for i in range(len(train_ids))])
        X_train, train_ids = X_train[mask], train_ids[mask]
        n_rejected = int((~mask).sum())

    X_train, X_test = normalize(X_train, X_test, cfg['normalize'])

    return {
        'X_train': X_train, 'train_ids': train_ids,
        'X_test': X_test,
        'test_labels': labels,
        'times': times, 'channels': np.array(keep),
        'n_train_rejected': n_rejected,
        'config': cfg,
    }


def save_subject(out_dir, sub, d):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: d[k] for k in ('X_train', 'train_ids', 'X_test', 'times', 'channels')}
    if d['test_labels'] is not None:
        arrays['test_labels'] = d['test_labels']
    np.savez_compressed(out_dir / f'sub-{sub:02d}.npz', **arrays)
    (out_dir / 'config.json').write_text(json.dumps(d['config'], indent=2))


def load_preprocessed(out_dir, sub):
    """Load what save_subject() wrote. test_labels is None for sub-05/06."""
    z = np.load(Path(out_dir) / f'sub-{sub:02d}.npz')
    d = {k: z[k] for k in z.files}
    d.setdefault('test_labels', None)
    cfg_path = Path(out_dir) / 'config.json'
    d['config'] = json.loads(cfg_path.read_text()) if cfg_path.exists() else None
    return d


# ─────────────────────────────────────────────
# Command line
# ─────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser(description='Preprocess the hackathon EEG for the team')
    p.add_argument('--data_dir', required=True, help='folder with trn/, tst/, tst_stim/')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--reference', default='none', choices=['none', 'average'],
                   help='average = common average reference')
    p.add_argument('--baseline', default='trial',
                   choices=['onset', 'trial', 'detrend', 'none'])
    p.add_argument('--l_freq', type=float, default=None)
    p.add_argument('--h_freq', type=float, default=40.0, help='0 = no low-pass')
    p.add_argument('--notch', type=float, default=None)
    p.add_argument('--sfreq_out', type=int, default=100)
    p.add_argument('--tmin', type=float, default=0.0)
    p.add_argument('--tmax', type=float, default=3.0)
    p.add_argument('--normalize', default='split', choices=['train', 'split', 'trial', 'none'])
    p.add_argument('--drop_channels', type=int, nargs='*', default=[])
    p.add_argument('--qc_dir', default=None, help='output folder of eeg_qc.py')
    p.add_argument('--auto_drop_from_qc', action='store_true',
                   help='drop channels the QC flagged flat/clipping in ANY subject')
    p.add_argument('--reject_train_trials', action='store_true',
                   help='drop training trials the QC flagged (test trials are kept)')
    args = p.parse_args()

    cfg = {
        'reference': None if args.reference == 'none' else args.reference,
        'baseline': None if args.baseline == 'none' else args.baseline,
        'l_freq': args.l_freq,
        'h_freq': args.h_freq or None,
        'notch': args.notch,
        'sfreq_out': args.sfreq_out,
        'tmin': args.tmin, 'tmax': args.tmax,
        'normalize': None if args.normalize == 'none' else args.normalize,
        'drop_channels': args.drop_channels,
        'auto_drop_from_qc': args.auto_drop_from_qc,
        'reject_train_trials': args.reject_train_trials,
    }
    if (args.auto_drop_from_qc or args.reject_train_trials) and not args.qc_dir:
        p.error('--auto_drop_from_qc / --reject_train_trials need --qc_dir')

    drop = set(args.drop_channels)
    if args.auto_drop_from_qc:
        for sub in args.subjects:
            drop |= set(qc_bad_channels(args.qc_dir, sub))
    drop = sorted(drop)
    cfg['drop_channels'] = drop
    print(f'Config: {json.dumps(cfg)}')
    if drop:
        print(f'Dropping channels {drop} for every subject')

    for sub in args.subjects:
        try:
            d = prepare_subject(args.data_dir, sub, cfg, qc_dir=args.qc_dir,
                                drop_channels=drop)
        except FileNotFoundError as e:
            print(f'sub-{sub:02d}: {e}')
            continue
        save_subject(args.out_dir, sub, d)
        print(f'sub-{sub:02d}: train {d["X_train"].shape} '
              f'(rejected {d["n_train_rejected"]}), test {d["X_test"].shape}, '
              f'labels {"yes" if d["test_labels"] is not None else "no"}')
    print(f'\nSaved to {Path(args.out_dir).resolve()}  (load with load_preprocessed)')


if __name__ == '__main__':
    main()
