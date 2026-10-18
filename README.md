# EEG Mind Reading

Decoding which video a participant watched from their EEG activity.

**Challenge:** Given an EEG recording (3 s, video-watching) and a set of
candidate videos, identify the correct video.

## Approach

We align EEG embeddings with video embeddings (CLIP / video foundation model)
using contrastive learning, then match by cosine similarity.

```
videos ──► CLIP/foundation encoder ──► video embedding ─┐
                                                         ├─► cosine sim ──► pick best
EEG ────► EEG encoder ──► EEG embedding ─────────────────┘
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

## Pipeline

| Step | Script |
|---|---|
| Load / explore data | `src/challenge_data.py` (validated loader for the challenge format) |
| EEG preprocessing | `src/preprocess.py`, diagnostics via `src/snr_probe.py` |
| EEG QC + automatic preprocessing | `eeg_pipeline/` — `python -m eeg_pipeline --data_dir data --out_dir prep`, see [`eeg_pipeline/README.md`](eeg_pipeline/README.md) |
| Organizer baseline (repro) | `src/baseline_organizer.py` (`--sub 01..04` eval, `--sub 05/06` submission) |
| Video embeddings (GPU) | `src/extract_embeddings.py` — model-agnostic, per-frame npz |
| Embedding benchmark | `src/bench_embeddings.py` — scores ANY embedding npz on dev subjects |
| EEG encoder + projection | `src/eeg_encoder.py` |
| Contrastive training | `src/train_contrastive.py` |
| Candidate-set evaluation | `src/eval_matching.py` |

## EEG QC + preprocessing (`eeg_pipeline/`)

One command runs quality control on all six subjects, chooses the
preprocessing from fixed rules (one config shared by all subjects), applies
it and writes a report explaining every decision (`prep/decisions.md`):

```bash
python -m eeg_pipeline --data_dir data --out_dir prep   # data/ holds trn/, tst/, tst_stim/
```

```python
from eeg_pipeline import load_preprocessed

d = load_preprocessed('prep', 1)
d['X_train'], d['train_ids']   # (n_train, ch, time), video id per trial
d['X_test']                    # (102, ch, time), ORIGINAL order -> submission rows
d['test_labels']               # sub-01..04 only, for validation; None for 05/06
```

Test trials are never dropped or reordered, and test labels are never used
for QC or preprocessing decisions. **Train and test are z-scored
separately** because test trials are 0.60-0.70x the amplitude of training
trials in every subject; normalizing test with training statistics would
leave test data on a smaller scale than the model was trained on. The test
trials behave like averages of ~2 repetitions (about half the noise of a
training trial), so validation on held-out *training* trials underestimates
test scores.
Variants that QC cannot decide (detrend, average reference, 1 Hz high-pass,
per-trial z-score, shorter window) are one `--set` flag away; compare them
on the sub-01..04 average. Rules, variants, the QC plot report, tests, the
brain-vs-AI analysis and its references (RSA: Kriegeskorte et al. 2008):
[`eeg_pipeline/README.md`](eeg_pipeline/README.md).

`eeg_pipeline/` is generally universal and can be applied to any new data, given the data format is kept.

## Video-side strategy: which foundation model? (measured, not guessed)

Frame extraction is trivial (0.08 s/video, all 1102 videos in ~90 s even on
CPU). Encoding is the GPU part: CLIP ViT-B/32 ≈ 2.9 s/video on CPU ≈ 52 min
for all 1102 — a few minutes on GPU. Therefore: **sweep multiple models**
(pre-hackathon public releases only), each becomes one candidate "brain
space", then let the benchmark decide:

| Priority | Model | Why |
|---|---|---|
| 1 | `clip-vitb32`, `clip-vitl14` | strong image-space baselines, fast |
| 2 | `dinov2-b14` | self-supervised features, often best for early visual areas |
| 3 | `siglip2-b16` | newer, stronger contrastive image model |
| 4 | `alexnet-f3/6/8/10` | organizer baseline layers + depth comparison for analysis |
| 5 | video/audio models (InternVideo2, V-JEPA 2, CLAP) | temporal + audio branch, orthogonal signal |

**GPU teammate workflow:**

```bash
# 1) extract embeddings for train + test stimuli (one command, GPU)
python -m src.extract_embeddings --models clip-vitb32 clip-vitl14 dinov2-b14 \
    --stimuli-dir <root>/stimuli_train --out work/emb_train.npz --device cuda
python -m src.extract_embeddings --models clip-vitb32 clip-vitl14 dinov2-b14 \
    --stimuli-dir <root>/stimuli_test --out work/emb_test.npz --device cuda

# 2) benchmark each model on dev subjects (any machine)
python -m src.bench_embeddings --data-dir <root> \
    --emb-train work/emb_train__clip-vitb32.npz \
    --emb-test  work/emb_test__clip-vitb32.npz \
    --window snr   # 50-350 ms; use --window org for 50-600 ms comparison
```

Log every result in the findings table below — the model ranking is the
core scientific result for the presentation (which representation best
matches human visual EEG at which layer depth).

## Contrastive alignment: recommended phasing

1. **Ridge-first (today, safe):** the organizer's forward model IS an
   alignment method and works with ~900 trials/subject. Benchmark the
   embedding spaces with it (above), keep the best.
2. **Cross-subject + ensembling (next):** per-subject best-embedding
   ensemble + z-scored correlation averaging across models/subjects.
3. **Contrastive EEG encoder (GPU, upside):** InfoNCE aligning raw windowed
   EEG segments (50–350 ms) to frozen best video embeddings. With ~900
   trials/subject data is limited — train across all 6 subjects with
   subject embeddings as condition, evaluate vs. ridge with the same
   candidate-matching protocol. Only adopt if it beats ridge on dev.

## Team roles

| Person | Owns |
|---|---|
| 1 | preprocessing + baselines (`src/preprocess.py`) |
| 2 | video embeddings + alignment framework (**GPU: runs the embedding sweep**) |
| 3 | EEG encoder + training loop (**GPU: contrastive training, phase 3**) |
| 4 | eval harness, CV, plots, repo hygiene |

Machines with GPUs take the heavy jobs (embedding extraction, contrastive
training); CPU machines run benchmarks, analyses and plots — the npz format
makes every result shareable via git-lfs/pull requests to `work/`.

## Data layout (expected)

```
data/
├── train/          # 4 subjects
└── test/           # 2 held-out subjects
```

Adapt `src/load_data.py` once organizers' template scripts arrive.

## Findings log

| Date | Finding |
|---|---|
| 2026-10-17 | Organizer baseline (AlexNet features.3 + Ridge forward model): top-1 ~4% on dev test (4x chance, their numbers). Train-fit: 78%. |
| 2026-10-17 | EEG is pre-normalized (std≈0.5/ch, kurtosis≈3, no outlier trials, p99 ptp<3.1σ). No artifact rejection needed. |
| 2026-10-17 | PSD: strong 30–50 Hz power = stimulus-locked gamma, NOT line noise. Bandpass/notch/avg-ref/Laplacian all REDUCE cross-subject video identifiability (d'). Raw wins. |
| 2026-10-17 | Window tuning: 50–350 ms optimal (d'=0.129 vs 0.104 for organizer's 50–600 ms), consistent across all 9 subject pairs (3 repeats, incl. 04/05/06). |
| 2026-10-17 | QC (`eeg_pipeline`): test trials are 0.60-0.70x the amplitude of training trials in every subject -> z-score train and test separately. |
| 2026-10-17 | The mean over trials was most likely subtracted upstream: train and test average responses anti-correlate (r = -0.71 to -0.92, all subjects). Average-response plots (ERP, reliability) are therefore not interpretable. |
| 2026-10-17 | Trial offsets are as large as the within-trial signal -> per-trial demeaning. No bad channels, no line noise, almost no artefact trials (agrees with the pre-normalization finding above). |
| 2026-10-17 | **Brain vs AI (RSA, `python -m eeg_pipeline.rsa`, training trials only)**: hierarchy in time: AlexNet conv1-3 match the EEG best at 105 ms, conv4-fc7 at 225 ms; and in space: early layers occipital, deep layers lateral parieto-occipital. VideoMAE middle layer as brain-like as AlexNet deep layers; CLIP weaker; VideoMAE top layer weakest. Models reach the noise ceiling. Details and commands: `eeg_pipeline/README.md`. |
| 2026-10-17 | **Brain-likeness predicts decoding**: the per-subject InfoNCE EEGNet trained toward each RSA feature set decodes better the more brain-like the target (rho = 0.69-0.81 over 9 targets). AlexNet fc6 17.2/41.7 and VideoMAE-l8 13.0/44.9 vs CLIP 11.0/33.1; with Daniel's retrained E-D in the megafusion: LOO 24.02/55.15 vs 22.30/48.28. Details: `docs/megafusion_rebuild.md` on `experiment/eegnet-preprocessing`. |
| 2026-10-17 | **The EEG follows the sound**: loudness over time matches the brain as well as the best visual features (125 ms, ~100% of noise ceiling, N1 latency), independent of all visual models (RDM overlap r <= 0.07), robust without silent videos, relatively strongest at lateral electrodes. All fusion members are visual-only, so audio is an untapped, independent signal. |
| 2026-10-18 | **A video world model is not more brain-like** (RSA): V-JEPA 2 ViT-L peaks at layer 18 (r 0.030 at 225 ms) vs AlexNet fc6 0.036 and VideoMAE-l8 0.035, so it was not used as a decoding target. Details: `eeg_pipeline/README.md`. |
| 2026-10-17 | Channels (organizers, order verified): Pz P3 P7 O1 Oz O2 P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2 (posterior only). Organizer preprocessing: 0.1-40 Hz band-pass, blink ICA, 1000 -> 500 Hz. Explains why average reference hurts (it removes the shared posterior signal) and why an extra 40 Hz low-pass changes nothing. |
| 2026-10-17 | **Test trials behave like averages of ~2 repetitions** (`python -m eeg_pipeline.check_test_averaging`): same-video correlation between subjects is 2.17x higher in test than in training (95% CI 1.85-2.48, 6 subject pairs; averaging predicts 2.00, a pure gain change 1.00). Power drops less at low than at high frequencies (delta ~0.5x vs beta/gamma ~0.33x), all 6 subjects, implied k = 1.8-2.4. Explains the 0.6-0.7x test amplitude and why holdout validation on single training trials underestimates tst scores (InfoNCE sub-01: ~7% holdout vs 16.7% tst). |
|---|---|
