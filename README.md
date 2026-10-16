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
| Load / explore data | `src/load_data.py`, `notebooks/01_eda.ipynb` |
| EEG preprocessing | `src/preprocess.py` |
| Video embeddings | `src/video_embeddings.py` |
| EEG encoder + projection | `src/eeg_encoder.py` |
| Contrastive training | `src/train_contrastive.py` |
| Candidate-set evaluation | `src/eval_matching.py` |

Usage: `python -m src.eval_matching --config configs/default.yaml`

## Team roles

| Person | Owns |
|---|---|
| 1 | preprocessing + baselines (`src/preprocess.py`) |
| 2 | video embeddings + alignment framework |
| 3 | EEG encoder + training loop |
| 4 | eval harness, CV, plots, repo hygiene |

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
