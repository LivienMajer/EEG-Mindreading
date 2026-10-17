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

## Findings on the challenge data (QC, 17 Oct)

- Data are clean and already preprocessed (low-pass ~45 Hz, no line noise, almost no
  artefact trials, no bad channels).
- The mean over trials was most likely subtracted upstream (train and test averages
  anti-correlate, r = -0.71 to -0.92 in all subjects).
- Trial offsets are as large as the within-trial signal → per-trial demeaning.
- Test trials are 0.60-0.70x the amplitude of training trials in every subject,
  evenly across frequencies → train and test normalized separately.

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
