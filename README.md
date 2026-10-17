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
