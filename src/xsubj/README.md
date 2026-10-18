# `src.xsubj`: EEG → CLIP decoding and submission

Ranks the 102 candidate videos for each EEG test recording. EEG trials are mapped into the embedding space of a frozen CLIP model, and candidates are ranked by cosine similarity.

```
video ──► frozen CLIP ViT-B-32 (open_clip 3.3.0, ViT-B-32 / openai) ─────────► 512-d vector ─┐
                                                                                              ├─► cosine ─► ranking
EEG ──► preprocessing ──► 3 votes: 9 encoders | pooled ridge | per-subject ridge ─► 512-d ─┘
```


## Setup (Python 3.11)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install numpy scipy scikit-learn pyyaml joblib pillow opencv-python-headless open_clip_torch torch
```

`requirements.txt` in the repository root covers these as well. The code uses CUDA when available and runs on CPU otherwise.

Verified with **Python 3.11.17** (numpy 2.4.6, scipy 1.17.1, scikit-learn 1.9.1, torch 2.14.1 CPU, open_clip 3.3.0, opencv 5.0.0, joblib 1.6.0): a full `python -m src.xsubj.submit` from scratch, including preprocessing, reproduces the committed sub-05 and sub-06 files exactly.

Unzip the organisers' download so that this layout exists (`data/` is gitignored):

```
data/eeg-mind-reading-challenge/
├── trn/sub-XX_trn.npz          training EEG, {video id: 17 x 1500}, 500 Hz, from onset
├── tst/sub-XX_tst.npy          test EEG, 102 x 17 x 1500, shuffled order
├── tst_stim/sub-XX_tst_stim.npy  test labels (sub-01..04 only)
├── stimuli_train/  stimuli_test/ videos (only needed to re-extract video features)
```

## Usage

All commands run from the repository root.

| Task | Command | Time (CPU) |
|---|---|---|
| **Write the submission** (sub-01..06; scores sub-01..04) | `python -m src.xsubj.submit` | ~1 min |
| Evaluate pooled ridge (official protocol) | `python -m src.xsubj.eval --models ridge` | <1 min |
| Evaluate ridge + one encoder + their ensemble | `python -m src.xsubj.eval` | ~5 min |
| EEG feature sweep / temporal decoding curve | `python -m src.xsubj.sweep_features` (`--tmax 3.0 --curve`) | 1–3 min |
| Train an encoder | `python -m src.xsubj.train --set seed=1 model.arch=mlp --out mlp_s1` | 2–4 min |
| Re-extract CLIP features | `python -m src.xsubj.video.clip_features --stim data/eeg-mind-reading-challenge/stimuli_train data/eeg-mind-reading-challenge/stimuli_test` | ~13 min |
| Low-level video features | `python -m src.xsubj.video.lowlevel --stim <same folders>` | ~4 min |

`submit` writes `submissions/sub-XX_ranked_candidate_list.npy` (int, 102 × 102, row i = i-th recording of `tst/sub-XX_tst.npy` in its original order, video IDs 1001–1102 from most to least likely). Useful options: `--model ridge|subject-ridge|encoder`, `--encoders <names>`, `--no-col-norm`, `--subjects`.

`eval` and `train` accept `--set KEY=VALUE` to override any config value, e.g. `--set model.arch=eegnet train.xsubj_avg.prob=0.5`.

The first run of any command preprocesses the EEG once and caches it under `models/xsubj/cache/` (about 1 min).

## What is versioned

Everything needed to reproduce the submission without retraining or re-extracting:

| File | Content |
|---|---|
| `models/xsubj/features/clip_vitb32.npz` | CLIP embeddings of all 1,102 videos (mean of 8 frames) |
| `models/xsubj/checkpoints/<name>/model.pt` | the 9 encoders listed under `submit.encoders` in the config |
| `models/xsubj/checkpoints/ridge/ridge.joblib` | pooled ridge (refitted by `submit`, deterministic) |
| `models/xsubj/checkpoints/ridge_subject/ridge_subject.joblib` | per-subject ridges (refitted by `submit`) |
| `submissions/sub-05_…npy`, `sub-06_…npy` | the submitted rankings |

The training command of each encoder is listed in the `submit:` section of `configs/xsubj.yaml`.

## Method in short

- **Preprocessing** (`eeg/preprocess.py`), per subject and split, label-free: 40 Hz low-pass (no high-pass, no average reference: both cost accuracy, see `sweep_preprocess.py`), 0–1 s window, 100 Hz, removal of the split's mean response (the test recordings' mean response is inverted relative to training), scaling, Euclidean alignment.
- **EEG features for ridge** (`nets/ridge.py`): amplitude of every sample from 50 to 600 ms, 17 channels → 935 features. Band power and covariance features were tested and carry no information.
- **Neural encoders** (`nets/encoders.py`, `train.py`): ShallowConvNet, EEGNet and MLP with a per-subject channel-mixing layer, trained with InfoNCE against the CLIP embeddings of all training videos.
- **Submission** (`submit.py`, `submit:` in the config): CLIP image ensemble (mean of 9 encoders + pooled ridge + per-subject ridge, `image_weight: 0.75`) plus ridge votes (pooled + per-subject) into AlexNet conv3 (1.0), VideoMAE layer 8 (1.0) and CLAP audio (0.3) space (`submit.extra_votes`). Each candidate's similarities are z-scored over the subject's test recordings before ranking.
- **Validation** (`eval.py`, `metrics.py`): train on the training splits of all six subjects, score the test splits of sub-01..04 among 102 candidates; top-1, top-5, mean rank, permutation test. Test splits are never trained on.

## Module map

| Module | Role |
|---|---|
| `eeg/adapter.py` | loads the challenge format (and synthetic data for smoke tests) |
| `eeg/preprocess.py` | preprocessing and caching |
| `eeg/dataset.py` | trial containers, augmentation |
| `video/clip_features.py`, `video/lowlevel.py` | video feature extraction |
| `video/embeddings.py` | loads video features (one or several files) |
| `nets/ridge.py`, `nets/encoders.py` | ridge decoder, neural encoders |
| `train.py`, `eval.py`, `sweep_features.py`, `submit.py` | entry points |
| `metrics.py` | retrieval metrics, per-candidate normalisation, permutation test |
| `predict.py` | older single-model inference; use `submit.py` for the challenge format |
