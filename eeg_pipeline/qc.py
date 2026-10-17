"""
EEG Quality Control
* Input: .npz (training, dict video_id -> 17 x 1500) and .npy (test, 102 x 17 x 1500)
* 17 unnamed channels at 500 Hz (names can be given with --channel_names).
* Some checks:
    - integrity: NaN/Inf, sizes, duplicated trials (also train <-> test), clipping
    - is there a reliable visual response? split-half reliability per channel
    - response latency (first peak should be ~80-130 ms; otherwise check onsets)
    - train vs test comparability (amplitude ratio, average-response correlation)
    - trial offsets (DC) and signs that the data were already filtered
    - cross-subject consistency of the average response

Nothing is removed or modified in the data. The script only reports. Test trials must
never be dropped (every test recording has to be ranked); flagged *training* trials
can optionally be excluded or down-weighted later, and that choice should be validated
on the sub-01..04 test sets.

Usage (full report with plots; the automatic pipeline uses the same checks)
-----
    python -m eeg_pipeline.qc --data_dir /path/to/challenge_data --output_dir qc_out
    python -m eeg_pipeline.qc --data_dir . --output_dir qc_out --subjects 1 2 3 4 5 6

The data directory must contain the folders `trn/` and `tst/` as provided by the
organizers (sub-XX_trn.npz / subXX_trn.npz and sub-XX_tst.npy).
"""

import argparse
import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import signal

import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving figures
import matplotlib.pyplot as plt


# ─────────────────────────────────────────────
# Defaults
# ─────────────────────────────────────────────
SFREQ = 500
N_CHANNELS = 17

POWERLINE_FREQ = 50          # Hz (Europe)
POWERLINE_REL_THRESH = 5.0   # 50 Hz power vs neighbouring bins
LOW_FREQ_CUTOFF = 1.0        # Hz
LOW_FREQ_REL_THRESH = 5.0    # <1 Hz power vs 1-40 Hz power
EMG_BAND = (20, 45)
BRAIN_BAND = (1, 20)

FLAT_STD = 1e-6              # absolute: a channel this flat is dead in any unit
CHANNEL_Z = 3.0              # robust z across channels
TRIAL_Z = 4.0                # robust z across trials (per channel)
MIN_BAD_CHANNELS = 2         # trial is flagged if >= this many channels are extreme
CORR_HIGH = 0.98             # possible electrode bridging


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def robust_z(x, axis=None):
    """Robust z-score (median / MAD), less affected by the outliers we want to find."""
    x = np.asarray(x, dtype=float)
    med = np.median(x, axis=axis, keepdims=True)
    mad = np.median(np.abs(x - med), axis=axis, keepdims=True) * 1.4826
    mad = np.where(mad < 1e-12, 1e-12, mad)
    return (x - med) / mad


def find_subject_files(data_dir, sub):
    """Return (train_path, test_path) for one subject; tolerant to naming variants."""
    data_dir = Path(data_dir)
    trn = (sorted(glob.glob(str(data_dir / 'trn' / f'sub-{sub:02d}_trn.npz')))
           + sorted(glob.glob(str(data_dir / 'trn' / f'sub{sub:02d}_trn.npz'))))
    tst = (sorted(glob.glob(str(data_dir / 'tst' / f'sub-{sub:02d}_tst.npy')))
           + sorted(glob.glob(str(data_dir / 'tst' / f'sub{sub:02d}_tst.npy'))))
    return (trn[0] if trn else None), (tst[0] if tst else None)


def load_subject(data_dir, sub):
    """
    Load training and test trials of one subject.

    Returns
        eeg:    np.ndarray [n_trials, n_channels, n_samples] (train first, then test)
        meta:   pd.DataFrame with one row per trial: split, video_id (train only), index
    """
    trn_path, tst_path = find_subject_files(data_dir, sub)
    arrays, rows = [], []

    if trn_path:
        trn = np.load(trn_path)
        ids = sorted(trn.files, key=int)
        arrays.append(np.stack([trn[i] for i in ids]).astype(np.float32))
        rows += [{'split': 'train', 'video_id': int(i), 'index_in_file': k}
                 for k, i in enumerate(ids)]
    else:
        print(f'  WARNING: no training file found for sub-{sub:02d}')

    if tst_path:
        tst = np.load(tst_path).astype(np.float32)
        arrays.append(tst)
        # test labels are not loaded on purpose: QC must not depend on them
        rows += [{'split': 'test', 'video_id': np.nan, 'index_in_file': k}
                 for k in range(len(tst))]
    else:
        print(f'  WARNING: no test file found for sub-{sub:02d}')

    if not arrays:
        return None, None
    return np.concatenate(arrays), pd.DataFrame(rows)


# ─────────────────────────────────────────────
# Channel-level checks (from eeg_preprocessing.py)
# ─────────────────────────────────────────────

def mean_psd(eeg, sfreq):
    """Average PSD over trials, per channel. eeg: [trials, channels, samples]."""
    nperseg = min(eeg.shape[-1], sfreq * 2)
    freqs, psd = signal.welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1)
    # median, not mean: a few artefact trials should not mark a whole channel as bad
    return freqs, np.median(psd, axis=0)      # [channels, freqs]


def channel_checks(eeg, ch_names, sfreq):
    """
    One row per channel: amplitude, flatness, outlying variance, powerline,
    low-frequency drift, EMG-like high-frequency power, bridging.
    """
    n_ch = eeg.shape[1]
    trial_std = eeg.std(axis=-1)                  # [trials, channels]
    ch_std = np.median(trial_std, axis=0)         # typical amplitude per channel
    std_z = robust_z(np.log(ch_std + 1e-12))      # log: variance is skewed

    freqs, psd = mean_psd(eeg, sfreq)
    df_f = freqs[1] - freqs[0]

    # powerline: 50 Hz bin vs neighbours (skip if 50 Hz is above Nyquist)
    pl_ratio = np.full(n_ch, np.nan)
    if POWERLINE_FREQ < sfreq / 2:
        idx = int(np.argmin(np.abs(freqs - POWERLINE_FREQ)))
        k = max(1, int(round(4 / df_f)))          # ~4 Hz on each side
        neigh = np.concatenate([psd[:, max(idx - k, 0):idx - 1],
                                psd[:, idx + 2:idx + k + 1]], axis=1)
        pl_ratio = psd[:, idx] / (neigh.mean(axis=1) + 1e-30)

    low = psd[:, freqs < LOW_FREQ_CUTOFF].mean(axis=1)
    mid = psd[:, (freqs >= LOW_FREQ_CUTOFF) & (freqs < 40)].mean(axis=1)
    low_ratio = low / (mid + 1e-30)

    emg = psd[:, (freqs >= EMG_BAND[0]) & (freqs <= EMG_BAND[1])].mean(axis=1)
    brain = psd[:, (freqs >= BRAIN_BAND[0]) & (freqs <= BRAIN_BAND[1])].mean(axis=1)
    emg_ratio = emg / (brain + 1e-30)
    emg_z = robust_z(np.log(emg_ratio + 1e-12))

    # correlation on all trials concatenated (channels x time)
    flat = eeg.transpose(1, 0, 2).reshape(n_ch, -1)
    with np.errstate(invalid='ignore', divide='ignore'):   # dead channels give NaN
        corr = np.corrcoef(flat)
    np.fill_diagonal(corr, np.nan)
    bridged = [[ch_names[j] for j in range(n_ch) if corr[i, j] > CORR_HIGH]
               for i in range(n_ch)]
    with np.errstate(invalid='ignore'):
        mean_abs_corr = np.array([np.nanmean(np.abs(r)) if np.isfinite(r).any() else 0.0
                                  for r in corr])
    corr_z = robust_z(mean_abs_corr)

    # Outlier flags need BOTH a large robust z (unusual vs other channels) AND a
    # meaningful size: when all channels are very similar, the MAD is tiny and a 5 %
    # difference would otherwise look like an outlier.
    rel_std = ch_std / (np.median(ch_std) + 1e-30)
    rel_emg = emg_ratio / (np.median(emg_ratio) + 1e-30)
    low_z = robust_z(np.log(low_ratio + 1e-12))
    corr_drop = np.median(mean_abs_corr) - mean_abs_corr

    rows = []
    for i, ch in enumerate(ch_names):
        flags = []
        if ch_std[i] < FLAT_STD:
            flags.append('flat')
        if std_z[i] > CHANNEL_Z and rel_std[i] > 1.5:
            flags.append('high_variance')
        if std_z[i] < -CHANNEL_Z and rel_std[i] < 0.67:
            flags.append('low_variance')
        if pl_ratio[i] > POWERLINE_REL_THRESH:
            flags.append('powerline')
        if low_ratio[i] > LOW_FREQ_REL_THRESH and low_z[i] > CHANNEL_Z:
            flags.append('low_freq_drift')    # only if worse than the other channels
        if emg_z[i] > CHANNEL_Z and rel_emg[i] > 1.5:
            flags.append('high_freq_noise')
        if bridged[i]:
            flags.append('possible_bridge')
        if corr_z[i] < -CHANNEL_Z and corr_drop[i] > 0.2:
            flags.append('uncorrelated_with_others')
        rows.append({
            'channel': ch,
            'median_trial_std': ch_std[i],
            'std_robust_z': std_z[i],
            'powerline_ratio': pl_ratio[i],
            'low_freq_ratio': low_ratio[i],
            'emg_ratio': emg_ratio[i],
            'mean_abs_corr': mean_abs_corr[i],
            'bridged_with': ';'.join(bridged[i]),
            'flags': ';'.join(flags),
        })
    return pd.DataFrame(rows), (freqs, psd), corr


# ─────────────────────────────────────────────
# Trial-level checks (from eeg_qc.py, windows -> trials)
# ─────────────────────────────────────────────

def trial_checks(eeg, sfreq, trial_z=TRIAL_Z, min_bad_channels=MIN_BAD_CHANNELS,
                 skip_channels=()):
    """
    One row per trial. Each measure is compared with the other trials of the same
    channel (robust z), then a trial is flagged when several channels are extreme.

    Checks
        high_amplitude  peak-to-peak much larger than usual (blinks, movement)
        high_variance   overall variance much larger than usual
        slow_drift      large difference between start and end of the trial
                        (generic replacement for the thesis eye-movement check)
        muscle          20-45 Hz / 1-20 Hz power ratio much larger than usual
        flat            a channel is (almost) constant in this trial

    skip_channels: indices of channels already flagged as dead/flat for the whole
    recording; they are a channel problem, not a trial problem, so they are ignored.
    """
    keep = np.array([c for c in range(eeg.shape[1]) if c not in set(skip_channels)])
    full_n_ch = eeg.shape[1]
    eeg = eeg[:, keep]
    n_tr, n_ch, n_s = eeg.shape
    ptp = np.ptp(eeg, axis=-1)                             # [trials, channels]
    std = eeg.std(axis=-1)
    edge = int(0.2 * sfreq)
    drift = np.abs(eeg[:, :, -edge:].mean(-1) - eeg[:, :, :edge].mean(-1))

    nperseg = min(n_s, sfreq)
    freqs, psd = signal.welch(eeg, fs=sfreq, nperseg=nperseg, axis=-1)
    emg = psd[..., (freqs >= EMG_BAND[0]) & (freqs <= EMG_BAND[1])].mean(-1)
    brain = psd[..., (freqs >= BRAIN_BAND[0]) & (freqs <= BRAIN_BAND[1])].mean(-1)
    emg_ratio = emg / (brain + 1e-30)

    z = {
        'high_amplitude': robust_z(np.log(ptp + 1e-12), axis=0),
        'high_variance': robust_z(np.log(std + 1e-12), axis=0),
        'slow_drift': robust_z(np.log(drift + 1e-12), axis=0),
        'muscle': robust_z(np.log(emg_ratio + 1e-12), axis=0),
    }
    n_flat = (std < FLAT_STD).sum(axis=1)

    out = {}
    for name, zz in z.items():
        n_bad = (zz > trial_z).sum(axis=1)
        out[f'n_ch_{name}'] = n_bad
        out[name] = n_bad >= min_bad_channels
    out['n_ch_flat'] = n_flat
    out['flat'] = n_flat > 0
    out['max_ptp'] = ptp.max(axis=1)
    out['max_amp_z'] = z['high_amplitude'].max(axis=1)

    df = pd.DataFrame(out)
    df['reject_suggested'] = (df['high_amplitude'] | df['high_variance']
                              | df['flat'] | (df['slow_drift'] & df['muscle']))
    # variance map with all channels (skipped ones shown as 0)
    var_z = np.zeros((n_tr, full_n_ch))
    var_z[:, keep] = z['high_variance']
    return df, var_z


# ─────────────────────────────────────────────
# Task-specific checks (new for the hackathon)
# ─────────────────────────────────────────────

EXPECTED_TEST_TRIALS = 102
RESP_WINDOW = (0.0, 1.0)     # s; where the visual evoked response lives
EARLY_PEAK_WINDOW = (0.05, 0.20)
BASELINE_WINDOW = (0.0, 0.05)  # s; before visual responses arrive (no pre-stimulus data)


def integrity_checks(eeg, meta, sfreq):
    """NaN/Inf, unexpected sizes, duplicated trials, amplifier clipping."""
    issues = []
    n_bad = int((~np.isfinite(eeg)).sum())
    if n_bad:
        issues.append(f'{n_bad} NaN/Inf values')
    n_tst = int((meta['split'] == 'test').sum())
    if n_tst and n_tst != EXPECTED_TEST_TRIALS:
        issues.append(f'{n_tst} test trials (expected {EXPECTED_TEST_TRIALS})')
    if eeg.shape[-1] != 3 * sfreq:
        issues.append(f'{eeg.shape[-1]} samples per trial (expected {3 * sfreq})')

    # duplicated trials (identical arrays), including train <-> test
    flat = np.ascontiguousarray(eeg.reshape(len(eeg), -1))
    seen, dup_pairs = {}, []
    for i, row in enumerate(flat):
        h = row.tobytes()
        if h in seen:
            dup_pairs.append((seen[h], i))
        else:
            seen[h] = i
    if dup_pairs:
        splits = meta['split'].values
        n_cross = sum(splits[a] != splits[b] for a, b in dup_pairs)
        issues.append(f'{len(dup_pairs)} duplicated trials ({n_cross} between train and test)')

    # clipping: many samples sitting exactly at a channel's global min or max
    finite = np.where(np.isfinite(eeg), eeg, np.nan)
    gmax = np.nanmax(finite, axis=(0, 2))
    gmin = np.nanmin(finite, axis=(0, 2))
    n_at_ext = ((eeg == gmax[None, :, None]) | (eeg == gmin[None, :, None])).sum(axis=(0, 2))
    clipped = [i for i, n in enumerate(n_at_ext) if n > 20]
    return issues, clipped


def demean_baseline(x, sfreq, window=BASELINE_WINDOW):
    """
    Subtract each trial's channel mean in the first ms after onset, then low-pass
    at 40 Hz. Used only for the response checks below (not to change the data):
    line noise that is time-locked to the trials would otherwise survive averaging
    and dominate the average response.
    """
    a, b = int(window[0] * sfreq), max(int(window[1] * sfreq), 1)
    x = x - x[..., a:b].mean(axis=-1, keepdims=True)
    sos = signal.butter(4, 40, btype='lowpass', fs=sfreq, output='sos')
    return signal.sosfiltfilt(sos, x, axis=-1)


def split_half_reliability(train, sfreq, window=RESP_WINDOW, n_splits=50):
    """
    How reproducible is the average response? Split the training trials randomly
    into two halves, average each half, correlate the two averages per channel over
    the response window, and correct for halving the data (Spearman-Brown).

    Only the stimulus-general response can be tested this way: each video was
    shown once per subject, so video-specific reliability cannot be measured.

    With strong slow (<1 Hz) activity, one odd/even split is very unstable (in
    simulations the same true response gave values from -1.5 to 0.9), so the
    correlation is averaged over many random splits. Spearman-Brown is applied only
    to positive correlations (it is meaningless for negative ones).
    """
    x = demean_baseline(train, sfreq)
    a, b = int(window[0] * sfreq), int(window[1] * sfreq)
    x = x[:, :, a:b]
    rng = np.random.default_rng(0)
    rs = []
    for _ in range(n_splits):
        p = rng.permutation(len(x))
        h1, h2 = x[p[:len(x) // 2]].mean(0), x[p[len(x) // 2:]].mean(0)
        r = []
        for c in range(x.shape[1]):
            if h1[c].std() < 1e-12 or h2[c].std() < 1e-12:
                r.append(np.nan)
            else:
                r.append(np.corrcoef(h1[c], h2[c])[0, 1])
        rs.append(r)
    r = np.nanmean(np.array(rs), axis=0)
    return np.where(r > 0, 2 * r / (1 + r), r)


def response_latency(train, sfreq):
    """
    Global field power (spread of the average response across channels) and the
    latency of its first peak. Visual responses should peak around 80-130 ms.
    """
    erp = demean_baseline(train, sfreq).mean(0)
    gfp = erp.std(axis=0)
    a, b = int(EARLY_PEAK_WINDOW[0] * sfreq), int(EARLY_PEAK_WINDOW[1] * sfreq)
    early = (a + int(np.argmax(gfp[a:b]))) / sfreq
    c = int(RESP_WINDOW[1] * sfreq)
    overall = int(np.argmax(gfp[a:c])) / sfreq + EARLY_PEAK_WINDOW[0]
    return gfp, early, overall


def train_test_shift(eeg, meta, sfreq):
    """
    Are test trials comparable to training trials? Per channel: amplitude ratio and
    correlation between the average train response and the average test response
    (averaging over all test trials does not need test labels).
    """
    tr = eeg[(meta['split'] == 'train').values]
    te = eeg[(meta['split'] == 'test').values]
    if len(tr) == 0 or len(te) == 0:
        return None
    amp_ratio = np.median(te.std(-1), 0) / (np.median(tr.std(-1), 0) + 1e-30)
    a, b = int(RESP_WINDOW[0] * sfreq), int(RESP_WINDOW[1] * sfreq)
    tr_b, te_b = demean_baseline(tr, sfreq), demean_baseline(te, sfreq)
    e_tr = tr_b.mean(0)[:, a:b]
    e_te = te_b.mean(0)[:, a:b]

    def corr_per_channel(x, y):
        return np.array([np.corrcoef(x[c], y[c])[0, 1]
                         if x[c].std() > 1e-12 and y[c].std() > 1e-12 else np.nan
                         for c in range(x.shape[0])])

    erp_corr = corr_per_channel(e_tr, e_te)

    # Reference: how well does an average of the same number of TRAINING trials
    # match the full training average? (average of a few trials can be very noisy)
    rng = np.random.default_rng(0)
    null = []
    for _ in range(200):
        idx = rng.choice(len(tr), size=min(len(te), len(tr) // 2), replace=False)
        rest = np.setdiff1d(np.arange(len(tr)), idx)
        null.append(np.nanmedian(corr_per_channel(tr_b[rest].mean(0)[:, a:b],
                                                  tr_b[idx].mean(0)[:, a:b])))
    null = np.array(null)
    observed = float(np.nanmedian(erp_corr))
    percentile = float((null < observed).mean() * 100)

    # which frequencies differ? test/train power per band, median over channels
    freqs, p_tr = mean_psd(tr, sfreq)
    _, p_te = mean_psd(te, sfreq)
    bands = {'<1 Hz': (0, 1), 'delta 1-4': (1, 4), 'theta 4-8': (4, 8),
             'alpha 8-13': (8, 13), 'beta 13-30': (13, 30), 'gamma 30-45': (30, 45)}
    band_ratio = {}
    for name, (lo, hi) in bands.items():
        m = (freqs >= lo) & (freqs < hi) & (freqs > 0)
        band_ratio[name] = float(np.median(p_te[:, m].mean(1) / (p_tr[:, m].mean(1) + 1e-30)))

    per_channel = pd.DataFrame({'amp_ratio_test_vs_train': amp_ratio,
                                'erp_corr_train_vs_test': erp_corr})
    subject_level = {
        'amp_ratio': float(np.nanmedian(amp_ratio)),
        'erp_corr': observed,
        'erp_corr_null_median': float(np.median(null)),
        'erp_corr_null_5pct': float(np.percentile(null, 5)),
        'erp_corr_percentile': percentile,
        'band_power_ratio': band_ratio,
        'erp_train': tr_b.mean(0).mean(0), 'erp_test': te_b.mean(0).mean(0),
        'erp_subsets': [tr_b[rng.choice(len(tr), len(te), replace=False)].mean(0).mean(0)
                        for _ in range(30)],
    }
    return per_channel, subject_level


def offset_and_filter_info(eeg, sfreq):
    """
    Per channel: how much trial offsets (DC) vary compared with the signal itself,
    and whether the spectrum looks already high-/low-pass filtered.
    """
    means = eeg.mean(-1)                              # [trials, channels]
    stds = np.median(eeg.std(-1), 0)
    offset_spread = means.std(0) / (stds + 1e-30)     # >1: offsets dominate -> demean
    freqs, psd = mean_psd(eeg, sfreq)
    band = (freqs >= 1) & (freqs < 40)
    hf = psd[:, freqs > 45].mean(1) / (psd[:, band].mean(1) + 1e-30)
    vlow = psd[:, (freqs > 0) & (freqs < 0.6)].mean(1) / (psd[:, band].mean(1) + 1e-30)
    return pd.DataFrame({'offset_spread': offset_spread,
                         'power_above_45Hz_rel': hf,
                         'power_below_0.6Hz_rel': vlow})


# ─────────────────────────────────────────────
# Plots
# ─────────────────────────────────────────────

def plot_reliability(rel, ch_names, out_path, title):
    fig, ax = plt.subplots(figsize=(10, 4))
    order = np.argsort(-np.nan_to_num(rel, nan=-1))
    ax.bar(range(len(rel)), np.nan_to_num(rel[order]), color='#4C72B0')
    ax.set_xticks(range(len(rel)))
    ax.set_xticklabels([ch_names[i] for i in order], rotation=90, fontsize=7)
    ax.axhline(0.5, color='grey', linestyle=':', linewidth=0.8)
    ax.set_ylim(min(-0.1, np.nanmin(rel) if np.isfinite(rel).any() else 0), 1)
    ax.set_ylabel('split-half reliability')
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_train_test(level, sfreq, out_path, title):
    """
    Left: channel-averaged response of all train trials, all test trials, and of
    random sets of training trials as large as the test set (grey: how noisy an
    average of that many trials is). Right: test/train power per frequency band.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5),
                                   gridspec_kw={'width_ratios': [2, 1]})
    t = np.arange(len(level['erp_train'])) / sfreq
    for k, s in enumerate(level['erp_subsets']):
        ax1.plot(t, s, color='0.75', linewidth=0.6,
                 label='random train subsets (same size as test)' if k == 0 else None)
    ax1.plot(t, level['erp_train'], color='#4C72B0', linewidth=1.6, label='all train trials')
    ax1.plot(t, level['erp_test'], color='#DD8452', linewidth=1.6, label='all test trials')
    ax1.axhline(0, color='black', linewidth=0.5)
    ax1.set_xlabel('Time from video onset (s)')
    ax1.set_ylabel('Mean over channels (file units)')
    ax1.set_title('Average response: train vs test')
    ax1.legend(fontsize=8)

    names = list(level['band_power_ratio'])
    vals = [level['band_power_ratio'][n] for n in names]
    ax2.bar(range(len(vals)), vals, color='#DD8452')
    ax2.axhline(1, color='black', linewidth=0.8)
    ax2.set_xticks(range(len(vals)))
    ax2.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('power test / train')
    ax2.set_title('Which frequencies differ?')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_gfp_all(gfps, sfreq, out_path):
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for tag, g in gfps.items():
        ax.plot(np.arange(len(g)) / sfreq, g / (g.max() + 1e-30), linewidth=1, label=tag)
    ax.set_xlim(0, 1.0)
    ax.set_xlabel('Time from video onset (s)')
    ax.set_ylabel('Global field power (normalized)')
    ax.set_title('Average response strength per subject')
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_erp(eeg, meta, ch_names, sfreq, out_path, title):
    """Average over training trials; the most direct check that visual responses exist."""
    train = eeg[(meta['split'] == 'train').values]
    erp = train.mean(axis=0)
    times = np.arange(erp.shape[-1]) / sfreq
    fig, ax = plt.subplots(figsize=(11, 5))
    for i, ch in enumerate(ch_names):
        ax.plot(times, erp[i], linewidth=1, label=ch)
    for t in (0.1, 0.2, 0.3):
        ax.axvline(t, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlabel('Time from video onset (s)')
    ax.set_ylabel('Amplitude (file units)')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=6, loc='upper right')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_variance_map(var_z, meta, ch_names, out_path, title):
    """Channel x trial map of variance robust z; bad channels show as rows, bad trials as columns."""
    fig, ax = plt.subplots(figsize=(13, 4.5))
    im = ax.imshow(var_z.T, aspect='auto', cmap='RdBu_r', vmin=-6, vmax=6,
                   interpolation='nearest')
    n_trn = int((meta['split'] == 'train').sum())
    ax.axvline(n_trn - 0.5, color='black', linewidth=1)
    ax.text(n_trn + 2, -0.8, 'test →', fontsize=8)
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_xlabel('Trial (train, then test)')
    ax.set_title(title)
    fig.colorbar(im, ax=ax, label='variance, robust z')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_psd(freqs, psd, ch_names, out_path, title):
    fig, ax = plt.subplots(figsize=(9, 5))
    for i, ch in enumerate(ch_names):
        ax.semilogy(freqs, psd[i], linewidth=0.8, label=ch)
    ax.axvline(POWERLINE_FREQ, color='grey', linestyle=':', linewidth=0.8)
    ax.set_xlim(0, min(100, freqs[-1]))
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (file units² / Hz)')
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_corr(corr, ch_names, out_path, title):
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(corr, cmap='RdBu_r', vmin=-1, vmax=1)
    ax.set_xticks(range(len(ch_names)))
    ax.set_xticklabels(ch_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ch_names)))
    ax.set_yticklabels(ch_names, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def plot_spotcheck(eeg, trial_df, meta, ch_names, sfreq, out_path, title, n=6):
    """Raw traces of the most extreme trials, for visual confirmation (as in the thesis)."""
    worst = trial_df['max_amp_z'].sort_values(ascending=False).index[:n]
    if len(worst) == 0:
        return
    fig, axes = plt.subplots(len(worst), 1, figsize=(13, 2.6 * len(worst)))
    axes = np.atleast_1d(axes)
    times = np.arange(eeg.shape[-1]) / sfreq
    for ax, idx in zip(axes, worst):
        tr = eeg[idx]
        spacing = np.percentile(np.abs(tr), 99) * 2 + 1e-12
        for c in range(tr.shape[0]):
            ax.plot(times, tr[c] + c * spacing, linewidth=0.5)
        ax.set_yticks(np.arange(tr.shape[0]) * spacing)
        ax.set_yticklabels(ch_names, fontsize=6)
        m = meta.iloc[idx]
        ax.set_title(f"{m['split']} trial #{m['index_in_file']}  "
                     f"(max amplitude z = {trial_df.loc[idx, 'max_amp_z']:.1f})", fontsize=9)
    axes[-1].set_xlabel('Time (s)')
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def run_subject(data_dir, sub, out_dir, ch_names, sfreq, trial_z):
    tag = f'sub-{sub:02d}'
    print(f'\n[{tag}]')
    eeg, meta = load_subject(data_dir, sub)
    if eeg is None:
        return None

    if eeg.shape[1] != len(ch_names):
        print(f'  NOTE: {eeg.shape[1]} channels in file, {len(ch_names)} names given; '
              f'using generic names')
        ch_names = [f'ch{i:02d}' for i in range(eeg.shape[1])]

    n_trn = int((meta['split'] == 'train').sum())
    n_tst = int((meta['split'] == 'test').sum())
    print(f'  trials: {n_trn} train + {n_tst} test, shape per trial {eeg.shape[1:]}')

    sub_dir = out_dir / tag
    sub_dir.mkdir(parents=True, exist_ok=True)

    issues, clipped = integrity_checks(eeg, meta, sfreq)
    for msg in issues:
        print(f'  INTEGRITY: {msg}')
    if not np.isfinite(eeg).all():
        print('  NaN/Inf values replaced by 0 for the remaining checks')
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

    ch_df, (freqs, psd), corr = channel_checks(eeg, ch_names, sfreq)
    dead = [i for i, f in enumerate(ch_df['flags']) if 'flat' in f]
    for i in clipped:
        ch_df.loc[i, 'flags'] = ';'.join(filter(None, [ch_df.loc[i, 'flags'], 'clipping']))

    train = eeg[(meta['split'] == 'train').values]
    rel = split_half_reliability(train, sfreq) if len(train) > 10 else np.full(len(ch_names), np.nan)
    ch_df['split_half_reliability'] = rel
    gfp, lat_early, lat_peak = response_latency(train, sfreq) if len(train) else (None, np.nan, np.nan)
    shift_out = train_test_shift(eeg, meta, sfreq)
    shift, level = shift_out if shift_out is not None else (None, None)
    if shift is not None:
        ch_df = pd.concat([ch_df, shift], axis=1)
    ch_df = pd.concat([ch_df, offset_and_filter_info(eeg, sfreq)], axis=1)

    # trials are compared within their own split, so a train/test level difference
    # (reported above) is not mistaken for bad trials
    parts, var_parts = [], []
    for split in ('train', 'test'):
        mask = (meta['split'] == split).values
        if mask.sum() == 0:
            continue
        d, v = trial_checks(eeg[mask], sfreq, trial_z=trial_z, skip_channels=dead)
        d.index = np.where(mask)[0]
        parts.append(d)
        var_parts.append(v)
    tr_df = pd.concat(parts).sort_index()
    var_z = np.concatenate(var_parts)
    tr_df = pd.concat([meta, tr_df], axis=1)

    ch_df.to_csv(sub_dir / 'qc_channels.csv', index=False)
    tr_df.to_csv(sub_dir / 'qc_trials.csv', index=False)

    typical = float(np.median(ch_df['median_trial_std']))
    print(f'  typical channel std: {typical:.3g} (file units; ~1-50 suggests µV, '
          f'~1e-6..5e-5 suggests V)')
    print(f'  first response peak: {lat_early * 1000:.0f} ms '
          f'(largest peak in 0-1 s: {lat_peak * 1000:.0f} ms)')
    good = ch_df.loc[ch_df['split_half_reliability'] >= 0.5, 'channel'].tolist()
    print(f'  reliability: median {np.nanmedian(rel):.2f}; '
          f'{len(good)} channels >= 0.5: {", ".join(good) if good else "none"}')
    if level is not None:
        shifted = not 0.8 <= level['amp_ratio'] <= 1.25
        print(f'  train vs test: amplitude ratio {level["amp_ratio"]:.2f}'
              f'{"  <- SHIFT" if shifted else ""}')
        unusual = level['erp_corr_percentile'] < 5
        print(f'  train vs test average response: correlation {level["erp_corr"]:.2f}; '
              f'random train subsets of the same size give median '
              f'{level["erp_corr_null_median"]:.2f} (5th percentile '
              f'{level["erp_corr_null_5pct"]:.2f})'
              f'{"  <- UNUSUAL" if unusual else "  -> within what noise produces"}')
        print('  power test/train per band: ' + ', '.join(
            f'{k} {v:.2f}' for k, v in level['band_power_ratio'].items()))
        plot_train_test(level, sfreq, sub_dir / 'train_vs_test.png',
                        f'{tag}: are test trials comparable to training trials?')
    off = np.nanmedian(ch_df['offset_spread'])
    print(f'  trial offsets: spread / signal = {off:.2f}'
          f'{"  -> per-trial demeaning recommended" if off > 0.3 else ""}')
    print(f'  spectrum: power >45 Hz = {np.nanmedian(ch_df["power_above_45Hz_rel"]):.3g}, '
          f'<0.6 Hz = {np.nanmedian(ch_df["power_below_0.6Hz_rel"]):.2f} '
          f'(relative to 1-40 Hz; very small values suggest the data were already filtered)')
    lowr = np.nanmedian(ch_df['low_freq_ratio'])
    if lowr > LOW_FREQ_REL_THRESH:
        print(f'  slow activity (<1 Hz) is {lowr:.0f}x the 1-40 Hz power in ALL channels '
              f'-> try per-trial detrending or a high-pass in preprocessing')
    mac = np.nanmedian(ch_df['mean_abs_corr'])
    if mac > 0.6:
        print(f'  channels are strongly correlated (median |r| = {mac:.2f}): a large '
              f'common signal -> try common average reference in preprocessing')
    flagged_ch = ch_df[ch_df['flags'] != '']
    if len(flagged_ch):
        for _, r in flagged_ch.iterrows():
            print(f'  channel {r["channel"]}: {r["flags"]}')
    else:
        print('  no channel flagged')

    for split in ('train', 'test'):
        part = tr_df[tr_df['split'] == split]
        if len(part):
            print(f'  {split}: {int(part["reject_suggested"].sum())}/{len(part)} trials '
                  f'flagged ({part["reject_suggested"].mean():.1%})')

    plot_erp(eeg, meta, ch_names, sfreq, sub_dir / 'erp.png',
             f'{tag}: average response over training trials')
    plot_variance_map(var_z, meta, ch_names, sub_dir / 'variance_map.png',
                      f'{tag}: trial variance per channel')
    plot_psd(freqs, psd, ch_names, sub_dir / 'psd.png', f'{tag}: mean PSD per channel')
    plot_corr(corr, ch_names, sub_dir / 'channel_correlation.png',
              f'{tag}: channel correlation')
    plot_spotcheck(eeg, tr_df, meta, ch_names, sfreq, sub_dir / 'spotcheck_worst.png',
                   f'{tag}: most extreme trials')
    plot_reliability(rel, ch_names, sub_dir / 'reliability.png',
                     f'{tag}: split-half reliability of the average response')

    erp = demean_baseline(train, sfreq).mean(0) if len(train) else None
    return {
        'subject': tag,
        'n_train': n_trn,
        'n_test': n_tst,
        'integrity_issues': ' | '.join(issues),
        'first_peak_ms': lat_early * 1000,
        'median_reliability': float(np.nanmedian(rel)),
        'typical_channel_std': typical,
        '_gfp': gfp,
        '_erp': erp,
        'flagged_channels': ' | '.join(f'{r.channel}:{r.flags}'
                                       for r in flagged_ch.itertuples()),
        'train_trials_flagged': int(tr_df.loc[tr_df.split == 'train',
                                              'reject_suggested'].sum()),
        'test_trials_flagged': int(tr_df.loc[tr_df.split == 'test',
                                             'reject_suggested'].sum()),
    }


def main():
    p = argparse.ArgumentParser(description='EEG QC for the mind-reading hackathon data')
    p.add_argument('--data_dir', required=True,
                   help='folder that contains trn/ and tst/')
    p.add_argument('--output_dir', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--sfreq', type=int, default=SFREQ)
    p.add_argument('--trial_z', type=float, default=TRIAL_Z,
                   help='robust z threshold for trial-level flags')
    p.add_argument('--channel_names', nargs='+', default=None,
                   help='17 channel names in file order, if the organizers provide them')
    args = p.parse_args()

    ch_names = args.channel_names or [f'ch{i:02d}' for i in range(N_CHANNELS)]
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for sub in args.subjects:
        s = run_subject(args.data_dir, sub, out_dir, ch_names, args.sfreq, args.trial_z)
        if s:
            summaries.append(s)

    if not summaries:
        print('No data found. Check --data_dir (it must contain trn/ and tst/).')
        sys.exit(1)

    # cross-subject comparison of the average response
    gfps = {s['subject']: s.pop('_gfp') for s in summaries}
    erps = {s['subject']: s.pop('_erp') for s in summaries}
    gfps = {k: v for k, v in gfps.items() if v is not None}
    if gfps:
        plot_gfp_all(gfps, args.sfreq, out_dir / 'gfp_all_subjects.png')
    tags = [k for k, v in erps.items() if v is not None]
    if len(tags) > 1:
        b = int(RESP_WINDOW[1] * args.sfreq)
        vecs = np.stack([erps[t][:, :b].ravel() for t in tags])
        isc = pd.DataFrame(np.corrcoef(vecs), index=tags, columns=tags)
        isc.to_csv(out_dir / 'erp_correlation_between_subjects.csv')
        print('\nAverage-response correlation between subjects (0-1 s, all channels):')
        print(isc.round(2).to_string())

    summary = pd.DataFrame(summaries)
    summary.to_csv(out_dir / 'qc_summary.csv', index=False)
    print('\n' + '=' * 60)
    print(summary.drop(columns=['flagged_channels']).to_string(index=False))
    print('(flagged channels per subject: see qc_summary.csv and qc_channels.csv)')
    print(f'\nAll outputs in: {out_dir.resolve()}')


if __name__ == '__main__':
    main()
