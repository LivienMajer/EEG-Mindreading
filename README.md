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
| 5 | `clap`, `clip-clap` | audio-only baseline and equal-weight multimodal fusion |

CLIP and CLAP are extracted as separate embedding files. Their latent spaces
are not directly comparable, so benchmark them independently before choosing
an EEG-side fusion strategy. CLAP uses each video's complete 3-second audio
track to create one normalized embedding per video. CLAP repeat-pads the short
audio internally to its expected 10-second input length. The first run
downloads the pretrained CLAP checkpoint. PyAV replaces invalid UTF-8 in MP4
metadata while reading so malformed tags do not interrupt audio extraction;
audio decode errors are still surfaced.

To create a fused representation, select `clip-clap`. It first averages the
sampled CLIP frame vectors into one visual vector per video, then L2-normalizes
that vector and the video's CLAP vector independently. The vectors are scaled
by `1/sqrt(2)` and concatenated, giving each modality equal contribution to
cosine similarity. The default ViT-B/32 + CLAP models produce one
1024-dimensional fused vector per video. If a video has no audio stream, the
extractor reports its ID and saves a normalized CLIP vector with a zero-filled
audio block, so visual matching can still include that candidate. In a
CLAP-only archive, the missing-audio entry is a zero vector and its ID is
reported; interpret CLAP-only scores with that limitation in mind.

**GPU teammate workflow:**

```bash
# 1) extract embeddings for train + test stimuli (one command, GPU)
python -m src.extract_embeddings --models clip-vitb32 clip-vitl14 dinov2-b14 clap clip-clap \
    --stimuli-dir <root>/stimuli_train --out work/emb_train.npz --device cuda
python -m src.extract_embeddings --models clip-vitb32 clip-vitl14 dinov2-b14 clap clip-clap \
    --stimuli-dir <root>/stimuli_test --out work/emb_test.npz --device cuda

# 2) benchmark each model on dev subjects (any machine)
python -m src.bench_embeddings --data-dir <root> \
    --emb-train work/emb_train__clip-vitb32.npz \
    --emb-test  work/emb_test__clip-vitb32.npz \
    --window snr   # 50-350 ms; use --window org for 50-600 ms comparison

# Run the same benchmark with the audio-only CLAP embeddings:
python -m src.bench_embeddings --data-dir <root> \
    --emb-train work/emb_train__clap.npz \
    --emb-test  work/emb_test__clap.npz \
    --window snr

# Benchmark the equal-weight audio+visual fused embeddings:
python -m src.bench_embeddings --data-dir <root> \
    --emb-train work/emb_train__clip-clap.npz \
    --emb-test  work/emb_test__clip-clap.npz \
    --window snr
```

For contrastive EEG-to-video alignment, pass the fused training archive to
`src.train_contrastive`; it reads the per-video fused vectors and infers the
EEG encoder's output size:

```bash
python -m src.train_contrastive --data <train-data> \
    --video-embeddings work/emb_train__clip-clap.npz
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
