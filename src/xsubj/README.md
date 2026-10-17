# Cross-subject EEG retrieval

`src.xsubj` maps EEG trials into the embedding space of the video candidates.
The repository-root [README](../../README.md) documents the full sequence,
including image/audio feature extraction, validation, final training, and
submission generation.

## Commands

Run these from the repository root after producing the four archives named in
`configs/xsubj.yaml`:

```powershell
python -m src.xsubj.eval --config configs/xsubj.yaml --models ridge
python -m src.xsubj.eval --config configs/xsubj.yaml
python -m src.xsubj.train --config configs/xsubj.yaml
python -m src.xsubj.submit --config configs/xsubj.yaml --subjects sub-05 sub-06
```

`eval` reports retrieval accuracy for labeled test trials (subjects 01–04).
Its encoder training uses a video-level holdout from the training trials for
early stopping; the test split is used only for scoring. `train` fits the
final encoder on the labeled training trials from all subjects and saves
`models/xsubj/checkpoints/all/model.pt`. `submit` combines that checkpoint
with the Ridge decoder by default, then writes ranked candidate arrays for
subjects 05 and 06.

## Components

- `eeg/adapter.py`: challenge and synthetic EEG data loading.
- `eeg/preprocess.py`: per-subject/split signal preprocessing and caching.
- `eeg/dataset.py`: trial/label alignment with the video embedding bank.
- `video/embeddings.py`: standardized archive loading, model-space fusion,
  and video-ID lookup.
- `nets/encoders.py`: EEGNet, shallow, and MLP encoders plus subject-specific
  channel mixing.
- `nets/ridge.py`: Ridge decoder and ERP/bandpower/covariance features.
- `train.py`: contrastive EEG encoder training and checkpoint handling.
- `eval.py`: labeled retrieval evaluation.
- `submit.py`: ranked candidate generation.

Image and audio extractors must provide archives with `video_ids` and
`embeddings`. Train and test archives from the same model are merged; distinct
model spaces are normalized independently and concatenated at equal norm.
The final concatenated vectors are normalized too; when a modality is missing,
its block remains zero and the available modality retains unit norm. The
default config combines ViT-B/32 and CLAP. Re-extract older archives if they
do not have the standardized schema.
