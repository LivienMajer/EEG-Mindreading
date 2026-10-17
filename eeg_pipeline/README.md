# EEG QC + preprocessing

Automatic quality control and preprocessing for the mind-reading hackathon EEG.
One command checks the data, decides the preprocessing from fixed rules, applies it
to all subjects and writes a report explaining every decision.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.11
python -m eeg_pipeline --data_dir data --out_dir prep
```

`data/` must contain the organizers' folders `trn/`, `tst/` and (for sub-01..04)
`tst_stim/`. Runtime is about a minute for all six subjects.

Then, in any notebook or script:

```python
from eeg_pipeline import load_preprocessed

d = load_preprocessed('prep', 1)
d['X_train']      # (n_train, channels, time)  float32
d['train_ids']    # video id of each training trial (sorted)
d['X_test']       # (102, channels, time) in the ORIGINAL file order -> submission rows
d['test_labels']  # true video ids for sub-01..04 (validation only!), None for sub-05/06
d['times']        # seconds from video onset
d['channels']     # indices of the channels kept
d['config']       # the exact settings used
```

## What happens

1. **QC metrics** per subject: integrity (NaN/Inf, shapes, duplicates, clipping),
   channel quality, artefact trials, line noise, existing filtering, trial offsets,
   test-vs-train amplitude. Test labels are never used.
2. **Rules** turn the metrics into **one configuration shared by all subjects**
   (thresholds are constants at the top of `eeg_pipeline/auto.py`):

   | Rule | Applied when | Action |
   |---|---|---|
   | Integrity | NaN/Inf, wrong shape, not 102 test trials | stop with an error |
   | Drop channels | flat or clipped in any subject | drop for all subjects |
   | Drop training trials | max amplitude z > 6 or flat (≤ 5 % per subject) | drop those training trials; test trials never |
   | Notch 50 Hz | 50 Hz peak > 5x neighbours | notch filter |
   | Low-pass 40 Hz | data not already low-passed | low-pass |
   | Per-trial demeaning | trial offsets > 0.3x within-trial signal | subtract each trial's mean |
   | Normalization | test/train amplitude outside 0.8-1.25 | z-score train and test separately, else with training statistics |

   Then: downsample 500 → 100 Hz, keep 0-3 s.
3. **Preprocess and save** `prep/sub-XX.npz`, plus `config.json`,
   `decisions.md` (human-readable), `decisions.json` and `qc_metrics.csv`.

## Optional variants

Some choices cannot be decided by QC; only the model score on the sub-01..04 test
sets can tell whether they help. Create them as separate folders:

```bash
python -m eeg_pipeline --data_dir data --out_dir prep_detrend --set baseline=detrend
python -m eeg_pipeline --data_dir data --out_dir prep_car     --set reference=average
python -m eeg_pipeline --data_dir data --out_dir prep_hp1     --set l_freq=1
python -m eeg_pipeline --data_dir data --out_dir prep_trialz  --set normalize=trial
python -m eeg_pipeline --data_dir data --out_dir prep_1s      --set tmax=1.0
```

Overrides are applied after the rules and listed in `decisions.md`.
Compare variants by the **average over sub-01..04** (top-1, top-5, median rank).

## Full QC report with plots

```bash
python -m eeg_pipeline.qc --data_dir data --output_dir qc_out
```

Per subject: power spectra, channel correlations, trial-variance map, most extreme
trials, train-vs-test comparison. Note: the organizers most likely subtracted the
mean over trials, so the average-response plots (ERP, reliability, latency, GFP) are
not interpretable for this dataset.

## Are the test trials averaged?

```bash
python -m eeg_pipeline.check_test_averaging --data_dir data
```

Compares test/train variance with the correlation between two subjects' responses to
the same video. Correlation ignores scale, so a pure gain change leaves it unchanged,
while averaging k repetitions removes noise and raises it by ~1/sqrt(R_a * R_b)
(R = test/train variance ratio). Uses the sub-01..04 test labels only to pair trials
of the same video.

## Brain vs AI: when and where does the EEG organize videos like which model?

Representational similarity analysis (RSA). For every pair of videos, how different are
the EEG responses, and how different are the videos for a model? If the two
dissimilarity matrices (RDMs) correlate, brain and model organize the videos the same
way. Done in 50 ms windows (20 ms steps) over 0-800 ms, so it also shows *when*.

**Data:** training trials only, the 695 videos seen by all six subjects. Test recordings
and labels are not used. Preprocessing: baseline 0-50 ms per channel (no visual response
yet; demeaning over the whole trial would leak the later response into early windows),
10 ms bin averages instead of a filter (no smearing to earlier times), subject's mean
response removed. One RDM per subject (correlation distance over 17 channels x 5 samples).

**Models** (`rsa_features.py`; all public, released before the hackathon):

| Feature set | What it captures | Source |
|---|---|---|
| `low_luminance`, `low_contrast`, `low_saturation`, `low_edges`, `low_motion` | simple visual properties over time (16 frames) | computed from the frames |
| `low_loudness` | loudness over time (log RMS, 100 ms bins) | computed from the sound track |
| `alexnet_conv1..5`, `alexnet_fc6/7` | image network, early to deep layers | torchvision AlexNet `IMAGENET1K_V1` |
| `clip_vitb32_daniel` | image-text model (semantic) | OpenCLIP `ViT-B-32` / `openai` (open_clip 3.3.0), mean of 8 frames; Daniel's `models/xsubj/features/clip_vitb32.npz` |
| `videomae_l4/l8/l12` | video model (sees motion), early/middle/top layer | `MCG-NJU/videomae-base-finetuned-kinetics` |
| `clap_audio` | audio model (sound meaning) | `laion/clap-htsat-unfused` (same weights as `laion_clap` default) |

Statistics: Spearman r between model RDM and each subject's RDM, averaged over subjects;
noise ceiling (lower bound: each subject vs the mean of the others); one-sided t-test
over subjects, FDR over time windows. The upper ceiling bound is reported in the CSV but
not interpreted: with single-trial RDMs it is inflated (each subject is partly compared
with itself).

**Reproduce** (CPU is enough; times on an 8-core laptop):

```bash
pip install -r requirements.txt
# 1. CLIP features from Daniel's branch (Git Bash; in PowerShell `>` corrupts binary files)
git show origin/feature/clip-lowlevel-forward-model:models/xsubj/features/clip_vitb32.npz > work/clip_vitb32_daniel.npz
# 2. all other features for the 1102 videos (~45 min, VideoMAE is most of it)
python -m eeg_pipeline.rsa_features --stim_dir data --out_dir work/rsa_features \
    --models lowlevel alexnet videomae audio
# 3. channel order check (~30 s)
python -m eeg_pipeline.check_channel_order --data_dir data
# 4. RSA over time + per electrode group (~6 min), robustness without silent videos (~2 min)
python -m eeg_pipeline.rsa --data_dir data --features work/rsa_features work/clip_vitb32_daniel.npz --groups
python -m eeg_pipeline.rsa --data_dir data --features work/rsa_features work/clip_vitb32_daniel.npz --exclude_silent
```

`data` holds `trn/` and `stimuli_train/`, `stimuli_test/`. Outputs in `results/rsa/`:
`rsa_timecourses.png` (figure), `rsa_timecourses.csv` (r per model and window, plus
noise ceiling), `rsa_summary.csv` (peak time, peak r, % of ceiling, significant span),
`rsa_model_overlap.csv` (how much the models agree with each other), `rsa_groups.csv`
(per electrode group), and the same with suffix `_with_sound` for the robustness run.
Verified with Python 3.11.17, numpy 2.4.6, scipy 1.17.1, torch 2.14.1 (CPU),
torchvision 0.29.1, transformers 5.20.1, av 18.1.0, decord 0.6.0, mne 1.13.2.

**Findings** (`results/rsa/`; r = Spearman, % = of the lower noise ceiling at the peak):

- **The brain response has two waves**, ~100 ms and ~200-260 ms (subjects agree most then).
- **Hierarchy in time:** AlexNet conv1-3 match best at 105 ms (r 0.034-0.036, ~100%
  of the ceiling) and fade; conv4-fc7 peak at 225 ms (fc6 r 0.036, 122%) and stay
  ahead to 800 ms. Peaks are stable when any one subject is left out (conv1-3: 105 ms
  in every fold; fc6/fc7: 225-265 ms). Same order as in the network: simple structure
  first, abstract content later.
- **Hierarchy in space:** early layers are strongest over the occipital pole (conv2 r
  0.038 at 85 ms vs 0.018 at the lateral groups); deep layers move to the lateral
  parieto-occipital electrodes at ~205 ms (fc6 0.024-0.027), over lateral
  occipito-temporal cortex, where objects and scenes are processed.
- **Simple visual features explain only the first wave:** edges (0.021) and contrast
  (0.019) peak at 105 ms and fade by 300 ms; luminance and motion energy are weak.
  The contrast effect is not significant once the 26 silent videos are removed, so
  only edges count as robust.
- **Video vs image models:** VideoMAE's middle layer (l8, r 0.035 at 225 ms) is as good
  as AlexNet's deep layers; its top layer (l12, tuned to action classes) is the weakest
  model (0.012). CLIP is in between (0.026 at 265 ms). A video model that sees motion is
  not more brain-like than an image network here, and motion energy alone is weak.
- **Sound:** loudness over time matches the brain as well as the best visual features
  (r 0.027 at 125 ms, ~100% of the ceiling; peak at the latency of the auditory N1),
  and is nearly independent of all visual models (RDM overlap r <= 0.07). Robust
  without the silent videos (0.025 at 105 ms, 76%). CLAP (sound meaning) is weaker (0.020
  at 105 ms): the early response follows how loud, more than what kind of sound.
  Loudness is relatively strongest at the lateral electrodes, closest to the auditory
  cortex (1.5-1.6x the early visual layer there vs 0.7x occipital); significant at
  the right-lateral peak, not at the left-lateral one (subjects vary). Electrical
  pick-up from headphones would also be lateral but has no delay; the signal is weak
  before 45 ms and peaks at 105-125 ms, which argues for a cortical response.
- **Channel order:** the organizers' names follow the file rows (channel correlation
  falls with scalp distance, rho = 0.93; random orders <= 0.45, p = 0.0002).

Limits: 17 posterior electrodes only (no central/temporal sites); latencies of the
earliest effects may be 10-20 ms too early if the organizers' 0.1-40 Hz filter was
zero-phase (peaks are not affected); layers of one network are correlated with each
other (overlap 0.44-0.62 between AlexNet and VideoMAE), so "which layer" comparisons
are about the profile over time, not independent contributions.

## Findings on the challenge data (QC, 17 Oct)

- Channels (organizers; order verified with `check_channel_order`): Pz P3 P7 O1 Oz O2
  P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2, all parietal/occipital. Organizer
  preprocessing: band-pass 0.1-40 Hz, blink components removed, 1000 -> 500 Hz.

- Data are clean and already preprocessed (low-pass ~45 Hz, no line noise, almost no
  artefact trials, no bad channels).
- The mean over trials was most likely subtracted upstream (train and test averages
  anti-correlate, r = -0.71 to -0.92 in all subjects).
- Trial offsets are as large as the within-trial signal → per-trial demeaning.
- Test trials are 0.60-0.70x the amplitude of training trials in every subject
  → train and test normalized separately.
- **Test trials behave like averages of ~2 repetitions** (`check_test_averaging`):
  same-video correlation between subjects is 2.17x higher in test than in training
  (95% CI 1.85-2.48; averaging predicts 2.00, a pure gain change 1.00). Power drops
  less at low frequencies (delta ~0.5x) than at high ones (beta/gamma ~0.33x), as
  expected when noise is averaged out and the stimulus response is kept. Same in all
  6 subjects; implied k = 1.8-2.4. Strictly, this shows test trials carry about half
  the noise of training trials; another noise-reduction step upstream would look the
  same and cannot be excluded.

## Rules of the challenge this code respects

- Test trials are never dropped or reordered.
- Test labels (sub-01..04) are only loaded for validation, never used in QC or
  preprocessing decisions.
- Normalizing test data with its own statistics is unsupervised (no labels).

## Tests

```bash
python -m pytest eeg_pipeline/tests -q   # or: python eeg_pipeline/tests/test_pipeline.py
```

Synthetic data built like the challenge data, and data with planted faults (dead
channel, line noise, blink trials). Also checks that the test order is preserved
and that the pipeline stops on NaN values.
