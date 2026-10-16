# `src.xsubj`: cross-subject EEG encoder

The EEG side of the pipeline, built for the held-out test subjects. It reads the team's data format (`src/load_data.py`) and the video embeddings written by `src/video_embeddings.py`.

## Usage

```bash
python -m src.xsubj.eval  --config configs/xsubj.yaml                 # leave-one-subject-out: ridge + encoder
python -m src.xsubj.eval  --config configs/xsubj.yaml --models ridge  # baseline only (seconds)
python -m src.xsubj.train --config configs/xsubj.yaml                 # final model on all labeled subjects
python -m src.xsubj.train --config configs/xsubj.yaml --finetune models/xsubj/checkpoints/all/model.pt --subject subject_05
python -m src.xsubj.predict --checkpoint models/xsubj/checkpoints/all/model.pt --subjects subject_05 subject_06
```

- **Data:** with `data.format: synthetic` (the default) it runs on generated data. Switch to `repo` once `data/train` and `data/test` exist.
- **Outputs:** everything goes to `models/xsubj/` (gitignored): checkpoints, LOSO results JSON, prediction CSVs and the preprocessing cache.

## What's different from the scaffold

**Preprocessing** (`eeg/preprocess.py`):
- Band-pass, notch, average reference, baseline, crop, resample to 128 Hz, per-subject scaling.
- **Euclidean alignment**: each subject's trials are whitened by their own mean spatial covariance. It needs no labels, so it also runs on the test subjects.

**Encoder** (`nets/encoders.py`): a ShallowConvNet or EEGNet backbone, plus a per-subject channel-mixing layer. Unseen subjects use the mean mixing matrix.

**Loss** (`train.py`):
- InfoNCE over *all* training videos, so repeated videos in a batch are never treated as negatives.
- A cosine regression term and a learnable temperature.
- Early stopping on held-out videos.

**Evaluation** (`eval.py`, `metrics.py`):
- Leave-one-subject-out for both the ridge baseline and the encoder.
- Top-1/top-5 among k candidates (2, 10, 50, all), with chance levels.
- A permutation test and accuracy after averaging repeats.

**Fine-tuning** (`--finetune`): for partially held-out subjects, only a new subject layer is trained on their labeled trials.

## Assumption to check when the data arrives

Video ids: the encoder matches the EEG label `str(y)` against the keys of `video_embeddings.npz` (the video file stems). If labels are ints and files are named `video_001.mp4`, add the mapping in `eeg/adapter.py::_load_repo`.
