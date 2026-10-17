# EEG Mind Reading

This repository contains separate video-image and video-audio embedding
extractors, plus a cross-subject EEG-to-video retrieval pipeline. The
extractors write standardized `.npz` archives consumed by `src.xsubj`.

## Requirements and data

Install the project dependencies from the repository root:

```powershell
python -m pip install -r requirements.txt
```

The challenge data is expected at the repository root:

- `stimuli_train\` and `stimuli_test\`: numeric-ID MP4 files.
- `trn\`: labeled EEG training archives for `sub-01` through `sub-06`.
- `tst\`: EEG test trials for all six subjects.
- `tst_stim\`: test labels for `sub-01` through `sub-04` (used for scoring;
  the labels for `sub-05` and `sub-06` are withheld).

If your data is stored elsewhere, update `data.challenge_dir` in
[`configs/xsubj.yaml`](configs/xsubj.yaml). Paths in that config are relative
to the repository root.

## 1. Extract video-image embeddings

The image extractor uses OpenCLIP by default. Run it once for each stimulus
split. Quote the script path because its directory name contains a space:

```powershell
python "src\extract embeddings\extract_video_embeddings.py" --videos stimuli_train --out work\train_vit_b32.npz --backend open_clip --model ViT-B-32 --pretrained openai --device cuda
python "src\extract embeddings\extract_video_embeddings.py" --videos stimuli_test --out work\test_vit_b32.npz --backend open_clip --model ViT-B-32 --pretrained openai --device cuda
```

Use `--device cpu` if CUDA is unavailable. For a small smoke test, add
`--limit 2` and write to a separate output filename; limited archives do not
contain all IDs needed for training or evaluation. The first run may download
pretrained weights.

## 2. Extract video-audio embeddings

The audio extractor uses LAION CLAP by default:

```powershell
python "src\extract embeddings\extract_audio_embeddings.py" --videos stimuli_train --out work\train_clap.npz --backend laion_clap --model HTSAT-tiny --device cuda
python "src\extract embeddings\extract_audio_embeddings.py" --videos stimuli_test --out work\test_clap.npz --backend laion_clap --model HTSAT-tiny --device cuda
```

Videos without an audio stream are retained with a zero audio vector and their
IDs are reported. Decode or model errors stop extraction with the video path
in the error. Use `--device cpu` when CUDA is unavailable.

Both extractors save `video_ids` and `embeddings` arrays, along with model
metadata. `configs/xsubj.yaml` combines the train and test archives for each
model into one video bank. CLIP and CLAP are normalized independently and
concatenated with equal norm, and the final vector is normalized as well.
They are not combined as raw unnormalized features. If a video has no audio,
its CLAP block remains zero and the available visual vector is normalized
without reducing its magnitude. Existing output archives are not overwritten;
add `--overwrite` when regenerating an existing path, including an older
archive with an incompatible schema.

## 3. Validate retrieval accuracy

Run the fast Ridge baseline first:

```powershell
python -m src.xsubj.eval --config configs/xsubj.yaml --models ridge
```

Run the EEG encoder alone, or the default comparison of Ridge and the
encoder:

```powershell
python -m src.xsubj.eval --config configs/xsubj.yaml --models encoder
python -m src.xsubj.eval --config configs/xsubj.yaml
```

Evaluation trains on labeled `trn` trials from all six subjects and scores
the labeled `tst` trials for `sub-01` through `sub-04` against their candidate
videos. It reports top-1/top-5 accuracy and other retrieval metrics and saves
the full results under `models\xsubj\results\`. This is the challenge's
labeled test split, not a validation split. During encoder training, a
video-level holdout drawn only from the training data is used for early
stopping. To evaluate a stricter leave-one-subject-out protocol, pass
`--mode loso`; it takes longer because it trains a separate model per scored
subject.

For a quick pipeline check, add `--epochs 1` to an encoder evaluation. This
is only a smoke test, not a meaningful accuracy run.

## 4. Train the final EEG encoder

After choosing the model settings, train on the labeled training trials from
all six subjects:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml
```

The checkpoint is saved to
`models\xsubj\checkpoints\all\model.pt`. Configuration overrides for an
experiment can be passed to evaluation with `--set`, for example:

```powershell
python -m src.xsubj.eval --models encoder --set model.arch=mlp train.xsubj_avg.prob=0
```

## 5. Generate ranked candidate outputs

The default submission command combines the trained encoder with a Ridge
decoder fitted on all labeled training trials. Generate the unlabeled
challenge predictions for `sub-05` and `sub-06` with:

```powershell
python -m src.xsubj.submit --config configs/xsubj.yaml --subjects sub-05 sub-06
```

The ranked candidate arrays are written to
`submissions\sub-05_ranked_candidate_list.npy` and
`submissions\sub-06_ranked_candidate_list.npy`. Each has shape `(102, 102)`;
each row ranks candidate video IDs 1001–1102. To use only the trained encoder,
add `--model encoder`. The `--model ridge` option runs the Ridge baseline
without loading an encoder checkpoint.

## Which EEG file does what?

- `src\xsubj\eeg\adapter.py` loads the challenge EEG trials and labels.
- `src\xsubj\eeg\preprocess.py` preprocesses and caches the EEG.
- `src\xsubj\nets\encoders.py` defines the trainable EEG encoder.
- `src\xsubj\nets\ridge.py` defines the fast Ridge baseline.
- `src\xsubj\train.py` trains and saves the EEG encoder.
- `src\xsubj\eval.py` evaluates retrieval accuracy on labeled test trials.
- `src\xsubj\submit.py` ranks candidates and writes submission arrays.
- `src\extract embeddings\eeg_encoder.py` contains standalone encoder
  building blocks; the end-to-end cross-subject workflow uses the
  `src.xsubj` modules above.

The existing `work\` archives from older extraction scripts may not use the
required `video_ids`/`embeddings` format. If `src.xsubj` reports an invalid
archive schema, regenerate the four archives with the commands above.
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
|---|---|
