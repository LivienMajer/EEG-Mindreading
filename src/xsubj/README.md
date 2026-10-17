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

The default config points at the existing combined CLIP+CLAP archives in
`work/`. The loader accepts both these legacy per-video/frame-key archives
(pooling frames as needed) and the newer standardized `video_ids` plus
`embeddings` format. If using separate standardized image/audio archives,
list the train and test files for each model under `data.video_emb`; distinct
model spaces are normalized independently and concatenated at equal norm.
The final vector is normalized too, so a missing modality does not reduce the
available embedding's norm.
