"""
Tests on synthetic data (run: python -m pytest eeg_pipeline/tests -q, or python eeg_pipeline/tests/test_pipeline.py)

"realistic": built like the challenge data as QC found it -- low-passed at ~45 Hz,
             large trial offsets and slow drift, mean over trials removed, test trials
             scaled by 0.65, clean channels.
"faulty":    a dead channel, 50 Hz line noise, blink-like artefact trials, no
             low-pass, train and test at the same scale.
"""
import json
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from eeg_pipeline import load_preprocessed, run_pipeline  # noqa: E402
from eeg_pipeline.auto import decide, parse_overrides, subject_metrics  # noqa: E402

FS, NS, NCH, NTR, NTE = 500, 1500, 17, 160, 102


def _make(root, sub, kind, seed):
    rng = np.random.default_rng(seed)
    n = NTR + NTE
    t = np.arange(NS) / FS
    x = rng.normal(0, 0.3, (n, NCH, NS))
    slow = signal.sosfilt(signal.butter(2, 1.0, fs=FS, output='sos'),
                          rng.normal(0, 1, (n, 1, NS + 1000)), axis=-1)[..., 1000:]
    x += slow / slow.std() * 0.3
    x += rng.normal(0, 0.3, (n, NCH, 1))                    # trial offsets
    x += 0.05 * np.exp(-((t - 0.15) / 0.04) ** 2)           # small evoked response
    if kind == 'realistic':
        x = signal.sosfiltfilt(signal.butter(8, 45, fs=FS, output='sos'), x, axis=-1)
        x -= x.mean(0)                                      # mean over trials removed
        x[NTR:] *= 0.65                                     # test scaled down
    else:
        x[:, 3] = 0.0                                       # dead channel
        x += 0.3 * np.sin(2 * np.pi * 50 * t)               # line noise
        x[5:8, :3] += 30 * np.exp(-((t - 1.0) / 0.1) ** 2)  # blink trials
    # each test trial gets a unique marker so we can verify the order is kept
    x[NTR:, 0, -1] += np.arange(NTE) * 1e-3

    ids = rng.choice(np.arange(1, 1001), NTR, replace=False)
    (root / 'trn').mkdir(parents=True, exist_ok=True)
    (root / 'tst').mkdir(exist_ok=True)
    (root / 'tst_stim').mkdir(exist_ok=True)
    np.savez(root / 'trn' / f'sub-{sub:02d}_trn.npz',
             **{str(i): x[k].astype(np.float32) for k, i in enumerate(ids)})
    np.save(root / 'tst' / f'sub-{sub:02d}_tst.npy', x[NTR:].astype(np.float32))
    if sub <= 4:
        np.save(root / 'tst_stim' / f'sub-{sub:02d}_tst_stim.npy',
                rng.permutation(np.arange(1001, 1103)))
    return x


def _dataset(kind):
    tmp = Path(tempfile.mkdtemp())
    raw = {sub: _make(tmp / 'data', sub, kind, seed=sub) for sub in (1, 5)}
    return tmp, raw


def _decisions(kind):
    tmp, _ = _dataset(kind)
    metrics = [subject_metrics(tmp / 'data', s)[0] for s in (1, 5)]
    cfg, reject, log, errors = decide(metrics)
    return cfg, reject, log, errors


def test_realistic_decisions():
    cfg, reject, log, errors = _decisions('realistic')
    assert not errors
    assert cfg['drop_channels'] == []
    assert cfg['notch'] is None
    assert cfg['h_freq'] is None                 # already low-passed
    assert cfg['baseline'] == 'trial'            # large offsets
    assert cfg['normalize'] == 'split'           # test scaled 0.65
    assert all(len(v) == 0 for v in reject.values())


def test_faulty_decisions():
    cfg, reject, log, errors = _decisions('faulty')
    assert not errors
    assert cfg['drop_channels'] == [3]           # dead channel
    assert cfg['notch'] == 50.0                  # line noise
    assert cfg['h_freq'] == 40.0                 # not filtered yet
    assert cfg['normalize'] == 'train'           # same scale
    for sub in (1, 5):
        assert len(reject[sub]) == 3             # the 3 blink trials, nothing else


def test_end_to_end_outputs():
    tmp, raw = _dataset('realistic')
    out = tmp / 'prep'
    cfg = run_pipeline(tmp / 'data', out, subjects=(1, 5), verbose=False)
    for f in ('sub-01.npz', 'sub-05.npz', 'config.json', 'decisions.md',
              'decisions.json', 'qc_metrics.csv'):
        assert (out / f).exists(), f
    d1, d5 = load_preprocessed(out, 1), load_preprocessed(out, 5)
    assert d1['X_train'].shape == (NTR, NCH, 300)
    assert d1['X_test'].shape == (NTE, NCH, 300)
    assert d1['test_labels'] is not None and len(d1['test_labels']) == NTE
    assert d5['test_labels'] is None
    assert np.all(np.isfinite(d1['X_train'])) and np.all(np.isfinite(d1['X_test']))
    assert np.all(np.diff(d1['train_ids']) > 0)  # sorted by video id
    # split normalization -> both splits have per-channel mean ~0, std ~1
    for X in (d1['X_train'], d1['X_test']):
        assert np.allclose(X.mean(axis=(0, 2)), 0, atol=1e-3)
        assert np.allclose(X.std(axis=(0, 2)), 1, atol=1e-2)
    # labels file is copied unchanged and the test order is preserved
    gt = np.load(tmp / 'data' / 'tst_stim' / 'sub-01_tst_stim.npy')
    assert np.array_equal(d1['test_labels'], gt)
    saved_cfg = json.loads((out / 'config.json').read_text())
    assert saved_cfg == json.loads(json.dumps(cfg))


def test_test_order_preserved():
    """Preprocess without any cross-trial step and check each test trial maps to itself."""
    tmp, raw = _dataset('realistic')
    out = tmp / 'prep'
    run_pipeline(tmp / 'data', out, subjects=(1,), verbose=False,
                 overrides={'normalize': None, 'baseline': None, 'sfreq_out': None})
    X = load_preprocessed(out, 1)['X_test']
    assert np.allclose(X, raw[1][NTR:].astype(np.float32), atol=1e-5)


def test_overrides():
    o = parse_overrides(['baseline=detrend', 'l_freq=1', 'reference=average',
                         'h_freq=none', 'tmax=1.0'])
    assert o == {'baseline': 'detrend', 'l_freq': 1, 'reference': 'average',
                 'h_freq': None, 'tmax': 1.0}
    tmp, _ = _dataset('realistic')
    out = tmp / 'prep'
    cfg = run_pipeline(tmp / 'data', out, subjects=(1,), verbose=False,
                       overrides={'tmax': 1.0, 'reference': 'average'})
    assert cfg['tmax'] == 1.0 and cfg['reference'] == 'average'
    assert load_preprocessed(out, 1)['X_train'].shape[-1] == 100
    assert 'Manual overrides' in (out / 'decisions.md').read_text()


def test_integrity_failure_stops():
    tmp, raw = _dataset('realistic')
    x = np.load(tmp / 'data' / 'tst' / 'sub-01_tst.npy')
    x[0, 0, 0] = np.nan
    np.save(tmp / 'data' / 'tst' / 'sub-01_tst.npy', x)
    try:
        run_pipeline(tmp / 'data', tmp / 'prep', subjects=(1,), verbose=False)
    except RuntimeError as e:
        assert 'NaN' in str(e)
    else:
        raise AssertionError('pipeline should stop on NaN values')


if __name__ == '__main__':
    tests = [v for k, v in dict(globals()).items() if k.startswith('test_')]
    for fn in tests:
        fn()
        print(f'ok  {fn.__name__}')
    print(f'\nall {len(tests)} tests passed')
