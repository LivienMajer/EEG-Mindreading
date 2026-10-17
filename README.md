# EEG Mind Reading

This repository is being rebuilt in small steps. The current implemented
steps extract visual and audio embeddings from the challenge MP4 files through
separate model backends. EEG embedding, model training, and candidate matching
are not part of these extractors yet.

## Requirements

The video extractor uses PyTorch, OpenCLIP, Decord, Pillow, NumPy, and tqdm.
OpenCLIP is its first supported backend; its model name and pretrained weights
are selectable. Other model libraries can be added as backends without
changing video decoding, temporal pooling, or the output archive format.
From the repository root, install dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Extract video embeddings

The challenge videos are in `stimuli_train/` (IDs 1–1000) and `stimuli_test/`
(IDs 1001–1102). From the repository root, run:

```powershell
python -m src.extract_video_embeddings --videos stimuli_train --out work\train_vit_b32.npz --backend open_clip --model ViT-B-32 --pretrained openai --device cuda
python -m src.extract_video_embeddings --videos stimuli_test --out work\test_vit_b32.npz --backend open_clip --model ViT-B-32 --pretrained openai --device cuda
```

Use `--device cpu` if the installed PyTorch cannot use CUDA. The first run may
download model weights. Eight frames are sampled evenly from each video by
default. Each frame is encoded by the selected image model, frame vectors are
normalized and averaged, and the resulting per-video vector is normalized.

To swap to another model supported by OpenCLIP, change `--model`,
`--pretrained`, and the output filename. For example:

```powershell
python -m src.extract_video_embeddings --videos stimuli_train --out work\train_vit_l14.npz --backend open_clip --model ViT-L-14 --pretrained openai --device cuda
```

To try the extractor on two videos before processing a full split:

```powershell
python -m src.extract_video_embeddings --videos stimuli_train --out work\smoke_test.npz --device cuda --limit 2
```

The output is a compressed `.npz` archive containing:

- `video_ids`: numeric IDs in the same row order as the embeddings.
- `embeddings`: one normalized vector per video; ViT-B/32 produces 512 values.
- `backend`, `model`, `pretrained`, and `n_frames`: extraction settings.

Existing output files are not overwritten by default. Choose a new `--out`
path or explicitly pass `--overwrite` to replace one.

## Extract audio embeddings

Audio embeddings are extracted separately using LAION CLAP by default. The
extractor decodes each MP4's audio stream, converts it to mono 48 kHz audio,
and encodes the complete clip as one embedding. CLAP handles short clips by
repeat-padding its input internally. Its backend adapter is separate from
audio decoding and output writing; another audio model can be added by
implementing the same adapter interface. You can select another architecture
or supply a compatible checkpoint with `--model` or `--checkpoint`.

```powershell
python -m src.extract_audio_embeddings --videos stimuli_train --out work\train_clap.npz --backend laion_clap --model HTSAT-tiny --device cuda
python -m src.extract_audio_embeddings --videos stimuli_test --out work\test_clap.npz --backend laion_clap --model HTSAT-tiny --device cuda
```

Try the first eight training videos (including video 0008, which has no audio
stream in the current data) before processing the full split:

```powershell
python -m src.extract_audio_embeddings --videos stimuli_train --out work\clap_smoke_test.npz --device cuda --limit 8
```

Videos with no audio stream remain in the archive with zero vectors; their IDs
are listed in `missing_audio_video_ids` and reported to the console. Audio
decode or model errors are reported with the video filename and stop the run.
The archive contains `video_ids`, `embeddings`, `backend`, `model`,
`checkpoint`, `sample_rate`, and `missing_audio_video_ids`. Existing output
files are not overwritten unless `--overwrite` is passed.
