"""
When does the brain organize videos like which AI model? Time-resolved RSA.

    python -m eeg_pipeline.rsa --data_dir data --features work/rsa_features work/clip_vitb32_daniel.npz
    python -m eeg_pipeline.rsa ... --groups           # also per electrode group (where)
    python -m eeg_pipeline.rsa ... --exclude_silent   # robustness: drop videos without sound

Brain: training trials of the videos seen by ALL six subjects (695). Per subject: baseline
= each channel's mean over 0-50 ms (epochs start at onset; demeaning over the whole trial
would leak the later response into early windows), 500 -> 100 Hz by averaging 10 ms bins
(no filter, so no activity is smeared to earlier latencies), removal of the subject's
mean response. For each 50 ms window
(20 ms steps), the pattern of a video is its 17 channels x 5 samples; the subject's RDM is
1 - Pearson correlation between the patterns of every pair of videos. Different videos are
different trials, so trial noise does not bias the RDM.

Models: one RDM per feature set (.npz with `ids` + `emb`). Features are z-scored over videos
(unsupervised); deep-network embeddings use correlation distance, the low-level time
courses (files named low_*) Euclidean distance, so brightness level counts, not only shape.

Score: Spearman correlation between model RDM and each subject's RDM, averaged over
subjects. Noise ceiling (Nili et al. 2014): lower bound = subject vs mean of the others,
upper bound = subject vs mean of all. Significance: one-sided t-test over the six subjects,
Benjamini-Hochberg FDR over time windows per model.

Also saved: Spearman correlation between every pair of model RDMs (how much two models
agree; e.g. loudness vs visual models), and with --groups the same analysis restricted to
four electrode groups (pp.CHANNELS; names from the organizers, order verified by
check_channel_order).

Only training data is used; test recordings and labels are not touched.
"""
import argparse
from pathlib import Path

import numpy as np
from scipy import stats

from . import preprocessing as pp

FS = 100
WIN = 5              # samples (50 ms)
STEP = 2             # samples (20 ms)
T_MAX = 0.8          # s
BASELINE = 25        # samples at 500 Hz: 0-50 ms, before any visual response reaches cortex
GROUPS = {
    'occipital': ['O1', 'Oz', 'O2', 'POz'],
    'left_lateral': ['P7', 'P5', 'PO7', 'P3'],
    'right_lateral': ['P8', 'P6', 'PO8', 'P4'],
    'midline_parietal': ['Pz', 'P1', 'P2', 'PO3', 'PO4'],
}
SILENT_LOG_RMS = -4  # low_loudness is log10 RMS; digital silence is -5
ORDER = ['low_luminance', 'low_contrast', 'low_saturation', 'low_edges', 'low_motion', 'low_loudness',
         'alexnet_conv1', 'alexnet_conv2', 'alexnet_conv3', 'alexnet_conv4', 'alexnet_conv5',
         'alexnet_fc6', 'alexnet_fc7', 'clip_vitb32_daniel',
         'videomae_l4', 'videomae_l8', 'videomae_l12', 'clap_audio']


# ─────────────────────────────────────────────
# RDMs
# ─────────────────────────────────────────────

def rdm(P, metric='correlation'):
    """Condensed upper triangle of the (n x n) dissimilarity matrix of the rows of P."""
    iu = np.triu_indices(len(P), 1)
    if metric == 'correlation':
        Z = P - P.mean(1, keepdims=True)
        Z /= np.linalg.norm(Z, axis=1, keepdims=True) + 1e-12
        return (1 - Z @ Z.T)[iu]
    sq = (P ** 2).sum(1)
    return np.sqrt(np.maximum(sq[:, None] + sq[None] - 2 * P @ P.T, 0))[iu]


def ranked(v):
    """Ranks centred and scaled so that a dot product of two ranked vectors = Spearman r."""
    r = stats.rankdata(v)
    r -= r.mean()
    return r / np.linalg.norm(r)


# ─────────────────────────────────────────────
# Brain and models
# ─────────────────────────────────────────────

def shared_videos(data_dir, subjects):
    sets = [set(pp.load_train(data_dir, s)[0].tolist()) for s in subjects]
    return np.array(sorted(set.intersection(*sets)))


def brain_data(data_dir, subjects, videos):
    """Per subject: (n_videos, 17, 100 Hz samples), preprocessed as in the module docstring."""
    out = []
    for s in subjects:
        ids, X = pp.load_train(data_dir, s)
        X = X[np.searchsorted(ids, videos)]
        X = X - X[..., :BASELINE].mean(-1, keepdims=True)     # onset baseline, see docstring
        X = X.reshape(*X.shape[:2], -1, pp.SFREQ // FS).mean(-1)   # 10 ms bins: no smearing in time
        out.append(X - X.mean(0, keepdims=True))
    return out


def brain_rdms(data, channels=None):
    """Raw RDM vectors (n_subjects, n_windows, n_pairs) and window centres in s."""
    starts = np.arange(0, int(T_MAX * FS) - WIN + 1, STEP)
    idx = slice(None) if channels is None else [pp.CHANNELS.index(c) for c in channels]
    B = np.array([[rdm(X[:, idx, a:a + WIN].reshape(len(X), -1)) for a in starts] for X in data])
    return B, (starts + WIN / 2) / FS


def silent_videos(features, videos):
    """Videos whose sound track is digital silence (or missing), from low_loudness.npz."""
    path = next(p / 'low_loudness.npz' for p in map(Path, features)
                if (p / 'low_loudness.npz').exists())
    z = np.load(path)
    pos = {int(i): k for k, i in enumerate(z['ids'])}
    return np.array([z['emb'][pos[int(v)]].max() < SILENT_LOG_RMS for v in videos])


def model_rdms(paths, videos):
    files = []
    for p in map(Path, paths):
        files += sorted(p.glob('*.npz')) if p.is_dir() else [p]
    models = {}
    for f in files:
        z = np.load(f)
        pos = {int(i): k for k, i in enumerate(z['ids'])}
        if not all(int(v) in pos for v in videos):
            print(f'  skipped {f.name}: not all videos present')
            continue
        E = z['emb'][[pos[int(v)] for v in videos]].astype(np.float64)
        E = (E - E.mean(0)) / (E.std(0) + 1e-12)
        models[f.stem] = ranked(rdm(E, 'euclidean' if f.stem.startswith('low_') else 'correlation'))
    order = [m for m in ORDER if m in models] + sorted(set(models) - set(ORDER))
    return {m: models[m] for m in order}


# ─────────────────────────────────────────────
# Analysis
# ─────────────────────────────────────────────

def noise_ceiling(B):
    """B: raw RDM vectors (n_subj, n_win, n_pairs) -> lower, upper bound per window."""
    n = len(B)
    lo, up = np.zeros(B.shape[1]), np.zeros(B.shape[1])
    for w in range(B.shape[1]):
        R = np.array([ranked(B[s, w]) for s in range(n)])
        mean_all = ranked(B[:, w].mean(0))
        for s in range(n):
            others = ranked(np.delete(B[:, w], s, axis=0).mean(0))
            lo[w] += R[s] @ others / n
            up[w] += R[s] @ mean_all / n
    return lo, up


def score(Br, m, t):
    """Model RDM vs subject RDMs: mean r per window, SEM, FDR-significant windows, LOSO peaks."""
    r = Br @ m                                                          # (n_subj, n_win)
    p = stats.ttest_1samp(r, 0, axis=0, alternative='greater').pvalue
    sig = stats.false_discovery_control(p) < 0.05
    loo_peaks = [t[np.argmax(np.delete(r, s, 0).mean(0))] for s in range(len(r))]
    return r.mean(0), r.std(0, ddof=1) / np.sqrt(len(r)), sig, loo_peaks


def write_csv(path, header, rows):
    path.write_text('\n'.join([header] + rows) + '\n')


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--data_dir', required=True)
    ap.add_argument('--features', nargs='+', required=True, help='.npz files or folders of them')
    ap.add_argument('--out_dir', default='results/rsa')
    ap.add_argument('--groups', action='store_true', help='also run per electrode group')
    ap.add_argument('--exclude_silent', action='store_true',
                    help='drop videos without sound (robustness of the audio results)')
    args = ap.parse_args(argv)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    tag = '_with_sound' if args.exclude_silent else ''

    subjects = list(range(1, 7))
    videos = shared_videos(args.data_dir, subjects)
    if args.exclude_silent:
        silent = silent_videos(args.features, videos)
        print(f'excluding {silent.sum()} silent videos')
        videos = videos[~silent]
    print(f'{len(videos)} videos seen by all {len(subjects)} subjects, '
          f'{len(videos) * (len(videos) - 1) // 2} video pairs')
    data = brain_data(args.data_dir, subjects, videos)
    B, t = brain_rdms(data)
    lo, up = noise_ceiling(B)
    Br = np.array([[ranked(b) for b in subj] for subj in B])          # (n_subj, n_win, n_pairs)
    models = model_rdms(args.features, videos)

    ms = [f'{x * 1000:.0f}ms' for x in t]
    rows = ['noise_ceiling_lower,' + ','.join(f'{v:.5f}' for v in lo),
            'noise_ceiling_upper,' + ','.join(f'{v:.5f}' for v in up)]
    summary, curves = [], {}
    for name, m in models.items():
        mean, se, sig, loo_peaks = score(Br, m, t)
        k = int(np.argmax(mean))
        curves[name] = (mean, se, sig)
        rows.append(name + ',' + ','.join(f'{v:.5f}' for v in mean))
        span = f'{t[sig][0] * 1000:.0f}-{t[sig][-1] * 1000:.0f}' if sig.any() else ''
        summary.append(f'{name},{t[k] * 1000:.0f},{min(loo_peaks) * 1000:.0f},'
                       f'{max(loo_peaks) * 1000:.0f},{mean[k]:.5f},{mean[k] / lo[k]:.3f},{span}')
    write_csv(out / f'rsa_timecourses{tag}.csv', 'model,' + ','.join(ms), rows)
    write_csv(out / f'rsa_summary{tag}.csv',
              'model,peak_ms,peak_loso_min_ms,peak_loso_max_ms,peak_r,fraction_of_lower_ceiling,'
              'significant_fdr_ms', summary)
    names = list(models)
    write_csv(out / f'rsa_model_overlap{tag}.csv', 'model,' + ','.join(names),
              [a + ',' + ','.join(f'{models[a] @ models[b]:.4f}' for b in names) for a in names])

    k = int(np.argmax(lo))
    print(f'\nNoise ceiling: peak at {t[k] * 1000:.0f} ms, lower {lo[k]:.4f}, upper {up[k]:.4f}')
    print(f'{"model":20s} {"peak":>6s} {"peak range (LOSO)":>18s} {"r":>8s} {"% lower ceiling":>16s}  significant (FDR)')
    for line in summary:
        name, pk, a, b, r, frac, span = line.split(',')
        print(f'{name:20s} {pk:>4s}ms {a:>8s}-{b:>4s} ms {float(r):8.4f} {float(frac) * 100:15.0f}%  '
              f'{span + " ms" if span else "none"}')

    if args.groups:
        print('\nPer electrode group: peak r (peak time), * = significant at the peak (FDR)')
        grows = []
        for g, chans in GROUPS.items():
            Bg, _ = brain_rdms(data, chans)
            Brg = np.array([[ranked(b) for b in subj] for subj in Bg])
            cells = []
            for n in names:
                mean, _, sig, _ = score(Brg, models[n], t)
                k = int(np.argmax(mean))
                cells.append(f'{n} {mean[k]:.4f} ({t[k] * 1000:.0f} ms){"*" if sig[k] else ""}')
                grows.append(f'{g},{" ".join(chans)},{n},{t[k] * 1000:.0f},{mean[k]:.5f},{int(sig[k])}')
            print(f'  {g} ({" ".join(chans)}):\n    ' + '\n    '.join(cells))
        write_csv(out / f'rsa_groups{tag}.csv', 'group,channels,model,peak_ms,peak_r,significant_fdr_at_peak',
                  grows)

    plot(t, curves, lo, up, out / f'rsa_timecourses{tag}.png')
    print(f'\nsaved results to {out}')


def plot(t, curves, lo, up, path):
    """Three panels: simple features, the AlexNet hierarchy, image/video/audio models.
    Only the lower noise ceiling is drawn: with single-trial RDMs the upper bound is
    inflated (each subject is partly compared with itself) and carries no information."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    panels = [
        ('Simple features (visual + loudness)', [n for n in curves if n.startswith('low_')], 'Greens'),
        ('AlexNet: early to deep layers', [n for n in curves if n.startswith('alexnet_')], 'Blues'),
        ('Image, video and audio models',
         [n for n in ('alexnet_conv2', 'alexnet_fc6', 'clip_vitb32_daniel', 'videomae_l4',
                      'videomae_l8', 'videomae_l12', 'clap_audio') if n in curves], None),
    ]
    fixed = {'alexnet_conv2': 'tab:cyan', 'alexnet_fc6': 'tab:blue', 'clip_vitb32_daniel': 'tab:red',
             'videomae_l4': '#c9a3e6', 'videomae_l8': '#8e44ad', 'videomae_l12': '#4a235a',
             'clap_audio': 'tab:orange'}
    top = max(max(c[0].max() for c in curves.values()), lo.max()) * 1.15
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.2), sharey=True)
    for ax, (title, names, cmap) in zip(axes, panels):
        ax.plot(t * 1000, lo, '--', color='0.4', lw=1.2, label='noise ceiling (lower bound)')
        for i, n in enumerate(names):
            color = fixed[n] if cmap is None else plt.get_cmap(cmap)(0.35 + 0.65 * (i + 1) / len(names))
            mean, se, sig = curves[n]
            ax.plot(t * 1000, mean, color=color, lw=1.8, label=n.replace('_daniel', ''))
            ax.fill_between(t * 1000, mean - se, mean + se, color=color, alpha=0.15)
            ax.plot(t[sig] * 1000, np.full(sig.sum(), -0.04 * top - 0.03 * top * i), 's',
                    color=color, ms=2.5)
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylim(-0.04 * top - 0.03 * top * 8, top)
        ax.set_title(title)
        ax.set_xlabel('time after video onset (ms, centre of 50 ms window)')
        ax.legend(fontsize=7.5, loc='upper right')
    axes[0].set_ylabel('Spearman r, model RDM vs brain RDM\n(mean over 6 subjects, ±SEM)')
    fig.text(0.5, 0.005, 'Squares below zero: significant time windows (one-sided t-test over '
             'subjects, FDR < 0.05). 695 training videos seen by all subjects.',
             ha='center', fontsize=8, color='0.3')
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    try:
        fig.savefig(path, dpi=150)
    except OSError:                       # Windows: the old figure is open in a viewer
        path = path.with_name(path.stem + '_new.png')
        fig.savefig(path, dpi=150)
    print(f'figure: {path}')


if __name__ == '__main__':
    main()
