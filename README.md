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
| Organizer baseline (repro) | `src/baseline_organizer.py` (`--sub 01..04` eval, `--sub 05/06` submission) |
| Video embeddings (GPU) | `src/extract_embeddings.py` — model-agnostic, per-frame npz |
| Embedding benchmark | `src/bench_embeddings.py` — scores ANY embedding npz on dev subjects |
| EEG encoder + projection | `src/eeg_encoder.py` |
| Contrastive training | `src/train_contrastive.py` |
| Candidate-set evaluation | `src/eval_matching.py` |

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
|---|---|
