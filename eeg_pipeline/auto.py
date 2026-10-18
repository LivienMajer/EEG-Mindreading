"""
Automatic QC -> decisions -> preprocessing
==========================================
    python -m eeg_pipeline --data_dir data --out_dir prep
    python -m eeg_pipeline --data_dir data --out_dir prep_detrend --set baseline=detrend

Optional settings that QC cannot decide (whether they help depends on the model
score, not on data quality) are left at their defaults and can be set with --set:
reference=average, baseline=detrend, l_freq=1, tmax=1.0, normalize=trial, ...
Overrides are applied after the rules and are listed in the report.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocessing as pp
from . import qc

# ─────────────────────────────────────────────
# Decision thresholds (change here, everything is logged)
# ─────────────────────────────────────────────
EXPECTED_CHANNELS = 17
EXPECTED_SAMPLES = 1500
EXPECTED_TEST_TRIALS = 102

DROP_CHANNEL_FLAGS = ('flat', 'clipping')  # channel is unusable -> drop for all subjects
EXTREME_TRIAL_Z = 6.0          # drop a TRAINING trial only for clear artefacts
MAX_REJECT_FRACTION = 0.05     # if more would be dropped, thresholds are off -> keep all
POWERLINE_RATIO = 5.0          # 50 Hz bin vs neighbours -> notch
ALREADY_LOWPASSED = 1e-3       # power >45 Hz relative to 1-40 Hz below this -> skip low-pass
DEFAULT_LOWPASS = 40.0
OFFSET_SPREAD = 0.3            # trial offsets vs within-trial signal -> per-trial demeaning
AMP_RATIO_RANGE = (0.8, 1.25)  # test/train amplitude outside this -> normalize per split


# ─────────────────────────────────────────────
# 1. Metrics
# ─────────────────────────────────────────────

def subject_metrics(data_dir, sub, sfreq=pp.SFREQ):
    """Decision-relevant QC numbers for one subject (no plots, no test labels)."""
    eeg, meta = qc.load_subject(data_dir, sub)
    if eeg is None:
        return None, None
    ch_names = [f'ch{i:02d}' for i in range(eeg.shape[1])]
    issues, clipped = qc.integrity_checks(eeg, meta, sfreq)
    n_nonfinite = int((~np.isfinite(eeg)).sum())
    if n_nonfinite:
        eeg = np.nan_to_num(eeg, nan=0.0, posinf=0.0, neginf=0.0)

    ch_df, (freqs, psd), corr = qc.channel_checks(eeg, ch_names, sfreq)
    for i in clipped:
        ch_df.loc[i, 'flags'] = ';'.join(filter(None, [ch_df.loc[i, 'flags'], 'clipping']))
    dead = [i for i, f in enumerate(ch_df['flags']) if 'flat' in str(f).split(';')]

    tr_mask = (meta['split'] == 'train').values
    te_mask = ~tr_mask
    tr_df, _ = qc.trial_checks(eeg[tr_mask], sfreq, skip_channels=dead)
    tr_df['index_in_file'] = meta.loc[tr_mask, 'index_in_file'].values

    info = qc.offset_and_filter_info(eeg, sfreq)
    tr, te = eeg[tr_mask], eeg[te_mask]
    amp_ratio = (float(np.median(np.median(te.std(-1), 0) / (np.median(tr.std(-1), 0) + 1e-30)))
                 if te_mask.any() and tr_mask.any() else np.nan)

    # was the mean over trials removed? (informational; see check_centering)
    both = eeg - eeg.mean(-1, keepdims=True)
    mean_vs_noise = float(np.median(np.abs(both.mean(0))
                                    / (both.std(0) / np.sqrt(len(both)) + 1e-30)))
    r_tr_te = (float(np.corrcoef(tr.mean(0).ravel(), te.mean(0).ravel())[0, 1])
               if te_mask.any() and tr_mask.any() else np.nan)

    m = {
        'subject': sub,
        'n_train': int(tr_mask.sum()),
        'n_test': int(te_mask.sum()),
        'n_channels': int(eeg.shape[1]),
        'n_samples': int(eeg.shape[2]),
        'n_nonfinite': n_nonfinite,
        'integrity_issues': ' | '.join(issues),
        'channels_to_drop': [i for i, f in enumerate(ch_df['flags'])
                             if set(str(f).split(';')) & set(DROP_CHANNEL_FLAGS)],
        'channel_flags': {ch: f for ch, f in zip(ch_df['channel'], ch_df['flags']) if f},
        'extreme_train_trials': tr_df.loc[(tr_df['max_amp_z'] > EXTREME_TRIAL_Z)
                                          | tr_df['flat'], 'index_in_file']
                                     .astype(int).tolist(),
        'qc_flagged_train_trials': int(tr_df['reject_suggested'].sum()),
        'max_powerline_ratio': float(np.nanmax(ch_df['powerline_ratio'])),
        'power_above_45Hz_rel': float(np.nanmedian(info['power_above_45Hz_rel'])),
        'low_freq_ratio': float(np.nanmedian(ch_df['low_freq_ratio'])),
        'offset_spread': float(np.nanmedian(info['offset_spread'])),
        'amp_ratio_test_train': amp_ratio,
        'mean_channel_corr': float(np.nanmedian(ch_df['mean_abs_corr'])),
        'grand_mean_vs_noise': mean_vs_noise,
        'r_mean_train_test': r_tr_te,
    }
    return m, ch_df


# ─────────────────────────────────────────────
# 2. Rules
# ─────────────────────────────────────────────

def decide(metrics):
    """
    Turn per-subject metrics into one shared configuration.
    Returns (config, reject_train_idx per subject, decision log, errors).
    """
    cfg = dict(pp.DEFAULT_CONFIG)
    log, errors = [], []

    def rule(name, value, threshold, decision, reason):
        log.append({'rule': name, 'value': value, 'threshold': threshold,
                    'decision': decision, 'reason': reason})

    # R1 integrity -----------------------------------------------------------
    bad = [m['subject'] for m in metrics if m['n_nonfinite']]
    if bad:
        errors.append(f'NaN/Inf values in subjects {bad}')
    shape_bad = [m['subject'] for m in metrics
                 if m['n_channels'] != EXPECTED_CHANNELS or m['n_samples'] != EXPECTED_SAMPLES]
    if shape_bad:
        errors.append(f'unexpected trial shape in subjects {shape_bad}')
    test_bad = [m['subject'] for m in metrics if m['n_test'] != EXPECTED_TEST_TRIALS]
    if test_bad:
        errors.append(f'not {EXPECTED_TEST_TRIALS} test trials in subjects {test_bad}')
    issues = {m['subject']: m['integrity_issues'] for m in metrics if m['integrity_issues']}
    rule('integrity', issues or 'none', 'NaN/Inf, shapes, duplicates, clipping',
         'stop' if errors else 'ok',
         '; '.join(errors) if errors else 'all files complete and well-formed')

    # R2 channels: same channels dropped for every subject ----------------------
    drop = sorted({c for m in metrics for c in m['channels_to_drop']})
    cfg['drop_channels'] = drop
    rule('drop channels', {m['subject']: m['channels_to_drop'] for m in metrics},
         f'flagged {"/".join(DROP_CHANNEL_FLAGS)} in any subject',
         f'drop {drop}' if drop else 'keep all',
         'unusable electrodes carry no brain signal' if drop
         else 'no flat or clipped channel; other channel flags are not reasons to drop')

    # R3 training trials -------------------------------------------------------
    reject = {}
    for m in metrics:
        idx = m['extreme_train_trials']
        frac = len(idx) / max(m['n_train'], 1)
        reject[m['subject']] = idx if frac <= MAX_REJECT_FRACTION else []
    n_rej = {s: len(v) for s, v in reject.items()}
    too_many = [m['subject'] for m in metrics
                if len(m['extreme_train_trials']) / max(m['n_train'], 1) > MAX_REJECT_FRACTION]
    rule('drop training trials', n_rej,
         f'max amplitude z > {EXTREME_TRIAL_Z} or flat channel; at most '
         f'{MAX_REJECT_FRACTION:.0%} per subject',
         'drop listed' if any(n_rej.values()) else 'keep all',
         ('only clear artefacts are dropped; test trials are never dropped'
          + (f'; subjects {too_many}: too many flagged, thresholds likely '
             f'mis-calibrated -> kept all' if too_many else '')))

    # R4 line noise ------------------------------------------------------------
    pl = max(m['max_powerline_ratio'] for m in metrics)
    cfg['notch'] = 50.0 if pl > POWERLINE_RATIO else None
    rule('notch 50 Hz', round(pl, 3), f'> {POWERLINE_RATIO}',
         'apply' if cfg['notch'] else 'skip',
         'line noise present' if cfg['notch'] else 'no 50 Hz line noise')

    # R5 low-pass --------------------------------------------------------------
    hf = max(m['power_above_45Hz_rel'] for m in metrics)
    cfg['h_freq'] = None if hf < ALREADY_LOWPASSED else DEFAULT_LOWPASS
    rule('low-pass', float(f'{hf:.3g}'), f'< {ALREADY_LOWPASSED} = already filtered',
         'skip' if cfg['h_freq'] is None else f'{DEFAULT_LOWPASS} Hz',
         'data already low-pass filtered; resampling has its own anti-alias filter'
         if cfg['h_freq'] is None else 'high-frequency noise present')

    # R6 per-trial demeaning ---------------------------------------------------
    off = max(m['offset_spread'] for m in metrics)
    cfg['baseline'] = 'trial' if off > OFFSET_SPREAD else None
    rule('per-trial demeaning', round(off, 3), f'> {OFFSET_SPREAD}',
         'apply' if cfg['baseline'] else 'skip',
         'trial offsets are large compared with the signal and carry no video '
         'information' if cfg['baseline'] else 'offsets are small')

    # R7 normalization ---------------------------------------------------------
    ratios = {m['subject']: round(m['amp_ratio_test_train'], 3) for m in metrics}
    lo, hi = AMP_RATIO_RANGE
    shifted = [s for s, r in ratios.items() if np.isfinite(r) and not lo <= r <= hi]
    cfg['normalize'] = 'split' if shifted else 'train'
    rule('normalization', ratios, f'test/train amplitude outside {AMP_RATIO_RANGE}',
         cfg['normalize'],
         f'test amplitude differs from training in subjects {shifted}: train and test '
         f'are z-scored with their own statistics (unsupervised, no labels)'
         if shifted else 'train and test comparable: statistics from training data')

    # informational ------------------------------------------------------------
    rule('mean over trials removed? (info)',
         {m['subject']: [round(m['grand_mean_vs_noise'], 2),
                         round(m['r_mean_train_test'], 2)] for m in metrics},
         '[grand mean / noise < 0.67, r(train mean, test mean) << 0]', 'no action',
         'if both hold, the average response was removed upstream: ERP-based '
         'analyses cannot be interpreted')
    rule('slow activity (info)', round(max(m['low_freq_ratio'] for m in metrics), 1),
         '<1 Hz vs 1-40 Hz power', 'no action',
         'optional variants: --set baseline=detrend or --set l_freq=1')
    rule('shared signal across channels (info)',
         round(max(m['mean_channel_corr'] for m in metrics), 2), 'median |r|',
         'no action', 'optional variant: --set reference=average')
    return cfg, reject, log, errors


# ─────────────────────────────────────────────
# 3. Overrides
# ─────────────────────────────────────────────

def parse_overrides(items):
    out = {}
    for item in items or []:
        if '=' not in item:
            raise ValueError(f'--set expects key=value, got {item!r}')
        k, v = item.split('=', 1)
        if k not in pp.DEFAULT_CONFIG:
            raise ValueError(f'unknown setting {k!r}; known: {sorted(pp.DEFAULT_CONFIG)}')
        if v.lower() in ('none', 'null', 'off'):
            out[k] = None
        else:
            try:
                out[k] = json.loads(v)
            except json.JSONDecodeError:
                out[k] = v
    return out


# ─────────────────────────────────────────────
# 4. Report
# ─────────────────────────────────────────────

def write_report(out_dir, metrics, log, cfg, overrides, reject, runtime):
    out_dir = Path(out_dir)
    pd.DataFrame([{k: v for k, v in m.items()
                   if k not in ('channel_flags', 'extreme_train_trials', 'channels_to_drop')}
                  for m in metrics]).to_csv(out_dir / 'qc_metrics.csv', index=False)
    (out_dir / 'decisions.json').write_text(json.dumps(
        {'config': cfg, 'overrides': overrides, 'rejected_train_trials': reject,
         'rules': log, 'metrics': metrics}, indent=2, default=str))

    lines = ['# Automatic preprocessing report', '',
             'Subjects: ' + ', '.join(f"sub-{m['subject']:02d}" for m in metrics) + '  ',
             f'Runtime: {runtime:.0f} s', '',
             '## Decisions', '',
             '| Rule | Measured | Threshold | Decision | Why |',
             '|---|---|---|---|---|']
    def cell(x):
        return str(x).replace('|', '\\|')

    for r in log:
        lines.append(f'| {cell(r["rule"])} | {cell(r["value"])} | {cell(r["threshold"])} | '
                     f'**{cell(r["decision"])}** | {cell(r["reason"])} |')
    if overrides:
        lines += ['', '## Manual overrides (applied after the rules)', '']
        lines += [f'- `{k}` = `{v}`' for k, v in overrides.items()]
    lines += ['', '## Final configuration', '', '```json',
              json.dumps(cfg, indent=2), '```', '',
              '## Per-subject QC numbers', '',
              pd.read_csv(out_dir / 'qc_metrics.csv').round(3).to_markdown(index=False)
              if _has_tabulate() else
              pd.read_csv(out_dir / 'qc_metrics.csv').round(3).to_string(index=False)]
    (out_dir / 'decisions.md').write_text('\n'.join(lines) + '\n')


def _has_tabulate():
    try:
        import tabulate  # noqa: F401
        return True
    except ImportError:
        return False


# ─────────────────────────────────────────────
# Entry points
# ─────────────────────────────────────────────

def run_pipeline(data_dir, out_dir, subjects=(1, 2, 3, 4, 5, 6), overrides=None,
                 strict=True, verbose=True):
    """Run QC, decide, preprocess and save. Returns the final config."""
    t0 = time.time()
    say = print if verbose else (lambda *a, **k: None)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics = []
    for sub in subjects:
        say(f'[QC] sub-{sub:02d} ...', end=' ', flush=True)
        m, _ = subject_metrics(data_dir, sub)
        if m is None:
            say('no data, skipped')
            continue
        metrics.append(m)
        say('done')
    if not metrics:
        raise FileNotFoundError(f'no subject data found in {data_dir} (needs trn/ and tst/)')

    cfg, reject, log, errors = decide(metrics)
    overrides = overrides or {}
    cfg.update(overrides)

    say('\nDecisions:')
    for r in log:
        say(f'  - {r["rule"]}: {r["decision"]}  ({r["reason"]})')
    if overrides:
        say(f'  - manual overrides: {overrides}')
    if errors and strict:
        write_report(out_dir, metrics, log, cfg, overrides, reject, time.time() - t0)
        raise RuntimeError('integrity check failed: ' + '; '.join(errors)
                           + f' (see {out_dir / "decisions.md"}; use --no-strict to continue)')

    say('')
    for m in metrics:
        sub = m['subject']
        d = pp.prepare_subject(data_dir, sub, cfg, drop_channels=cfg['drop_channels'],
                               reject_train_idx=reject.get(sub, []))
        pp.save_subject(out_dir, sub, d)
        say(f'[prep] sub-{sub:02d}: train {d["X_train"].shape} '
            f'(dropped {d["n_train_rejected"]}), test {d["X_test"].shape}, '
            f'labels {"yes" if d["test_labels"] is not None else "no"}')

    write_report(out_dir, metrics, log, cfg, overrides, reject, time.time() - t0)
    say(f'\nSaved to {out_dir.resolve()}  (report: decisions.md)')
    return cfg


def main(argv=None):
    p = argparse.ArgumentParser(
        prog='python -m eeg_pipeline',
        description='QC the hackathon EEG, decide preprocessing from fixed rules, '
                    'apply it and save the result with a decision report.')
    p.add_argument('--data_dir', required=True, help='folder with trn/, tst/ (and tst_stim/)')
    p.add_argument('--out_dir', required=True)
    p.add_argument('--subjects', type=int, nargs='+', default=[1, 2, 3, 4, 5, 6])
    p.add_argument('--set', dest='overrides', nargs='*', default=[], metavar='KEY=VALUE',
                   help='override a setting after the rules, e.g. baseline=detrend '
                        'reference=average l_freq=1 tmax=1.0 normalize=trial')
    p.add_argument('--no-strict', dest='strict', action='store_false',
                   help='continue even if the integrity check fails')
    args = p.parse_args(argv)
    try:
        run_pipeline(args.data_dir, args.out_dir, args.subjects,
                     parse_overrides(args.overrides), strict=args.strict)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        print(f'\nERROR: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
