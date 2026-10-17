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
| 2026-10-17 | **InfoNCE holdout (Exp #1/#2)**: alexnet-f3, sub-01, 102 held-out train videos: **top-1 6.9% / top-5 23.5%** (7×/4.8× chance). Batch top-1 44% in-sample ≠ overfit: generalization confirmed. EEGNet k1=16/k2=8, 256-d target PCA space, τ=0.1, 200 ep. |
| 2026-10-18 | **Ridge reference on tst (alexnet-f3, snr window)**: sub-01 2.9%/9.8%, sub-02 4.9%/10.8%, sub-03 1.0%/9.8%, sub-04 1.0%/6.9% → **mean 2.45%/9.31%**. InfoNCE beats Ridge on sub-01 by 2.3× top-1 / 2.4× top-5 (6.9%/23.5% vs 2.9%/9.8%). Contrastive path validated against reference architecture. |
| 2026-10-18 | **Seed variance study (3 net-seeds, fixed split)**: alexnet 6.5%/27.1% mean, clip 7.5%/28.1% mean → target space effect within seed noise (±5pp top-5). The apparent identical clip==alexnet result at netseed 0 was COINCIDENCE. EEG side is the bottleneck, not the embedding space. |
| 2026-10-18 | **Ridge on CLIP (tst, all dev subjects)**: mean 3.43%/13.73% — CLIP is the better single target space for Ridge (2.45/9.31 on alexnet), consistent with EEG-bottleneck finding: even Ridge gains more from space swap than InfoNCE does. |
| 2026-10-18 | **RANK-ENSEMBLE BREAKTHROUGH (sub-01 holdout)**: fusing the two complementary spaces (mean rank) → **top-1 11.76% (12× chance) / top-5 38.24%** vs best single 7.84.%/33.33%. Key insight: top-1 hits of the two spaces OVERLAP ZERO (6 vs 8 hits, 0 shared), top-5 trial sets only 13/30 shared → spaces solve different trials, ~50% of trials in top-5 of at least one space. Fusion > single-space improvement. |
| 2026-10-18 | **FAIR tst COMPARISON (same protocol as ridge, sub-01–04, clip)**: InfoNCE **mean top-1 11.03% / top-5 33.09%** vs Ridge 3.43%/13.73% → **3.2×/2.4×**. Per subject: 16.7/37.3, 13.7/32.4, 5.9/34.3, 7.8/28.4. Holdout proxy UNDERESTIMATES tst (sub-01: 7%→16.7%) — conservative iteration on holdout is safe. |
| 2026-10-18 | **tst-Rank-Fusion Lektion**: Equal-Weight-Fusion mit deutlich schwächerem Member (Ridge 3.4%) zieht den Starken runter (11.0→7.4). Herabgewichtete Fusion 9:1 zeigt ein robustes Plateau (9:1→12.75%, 19:1→12.01%, beide > solo 11.03): +1.7pp top-1, hilft v.a. schwachen Subjects (sub-04 7.8→12.8). Nächstes Ziel: 2. Zielraum (alexnet) auf tst — Holdout zeigte ZERO top-1-Overlap zwischen den Räumen. |
|---|---|
