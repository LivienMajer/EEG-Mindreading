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

To test intermediate ViT representations (patch-token means after blocks 6
and 12, plus the final CLIP embedding), extract a separate feature bank:

```powershell
python "src\extract embeddings\extract_video_embeddings.py" --videos stimuli_train --out work\emb_train__clip-vitb32-layers6-12.npz --model ViT-B-32 --pretrained openai --device cuda --intermediate-layers 6 12
python "src\extract embeddings\extract_video_embeddings.py" --videos stimuli_test --out work\emb_test__clip-vitb32-layers6-12.npz --model ViT-B-32 --pretrained openai --device cuda --intermediate-layers 6 12
```

The extractor normalizes each layer representation independently before
concatenation. To train/evaluate against these features and separate CLAP
features, pass all split archives to the evaluator:

```powershell
python -m src.xsubj.eval --config configs/xsubj.yaml --models encoder --mode official --video-emb work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__clap.npz work\emb_test__clap.npz
```

The test CLAP archive must be present as `work\emb_test__clap.npz`; extract it
with the audio commands below if it is not already available.
To save a final checkpoint trained against these targets, pass the same
`--video-emb` arguments to `src.xsubj.train` and choose a new `--out` run name.

For temporal visual features, the same extractor can run VideoMAE on a
chronologically sampled 16-frame sequence. It pools the model's contextualized
spatiotemporal tokens per video, rather than averaging independent image
embeddings:

```powershell
python "src\extract embeddings\extract_video_embeddings.py" --backend videomae --model MCG-NJU/videomae-base --videos stimuli_train --out work\emb_train__videomae.npz --n-frames 16 --device cuda
python "src\extract embeddings\extract_video_embeddings.py" --backend videomae --model MCG-NJU/videomae-base --videos stimuli_test --out work\emb_test__videomae.npz --n-frames 16 --device cuda
python -m src.xsubj.eval --config configs/xsubj.yaml --models encoder --mode official --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clap.npz work\emb_test__clap.npz
```

To retain intermediate VideoMAE features alongside the final-layer features,
extract each 1-based transformer block to its own archive. For example, layers
4 and 8 provide earlier/mid-level representations, while the existing
`emb_*__videomae.npz` archives remain the final-layer head:

```powershell
python "src\extract embeddings\extract_video_embeddings.py" --backend videomae --model MCG-NJU/videomae-base --videos stimuli_train --out work\emb_train__videomae-layer4.npz --n-frames 16 --intermediate-layers 4 --device cuda
python "src\extract embeddings\extract_video_embeddings.py" --backend videomae --model MCG-NJU/videomae-base --videos stimuli_test --out work\emb_test__videomae-layer4.npz --n-frames 16 --intermediate-layers 4 --device cuda
python "src\extract embeddings\extract_video_embeddings.py" --backend videomae --model MCG-NJU/videomae-base --videos stimuli_train --out work\emb_train__videomae-layer8.npz --n-frames 16 --intermediate-layers 8 --device cuda
python "src\extract embeddings\extract_video_embeddings.py" --backend videomae --model MCG-NJU/videomae-base --videos stimuli_test --out work\emb_test__videomae-layer8.npz --n-frames 16 --intermediate-layers 8 --device cuda
```

These archives hold mean-pooled contextualized tokens from the selected
transformer block, normalized as a separate feature space. Add their train/test
pairs to `--video-emb` and name the corresponding heads explicitly when
training. This example trains and fuses final VideoMAE, layers 4 and 8, CLIP,
and AlexNet fc6:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__videomae-layer4.npz work\emb_test__videomae-layer4.npz work\emb_train__videomae-layer8.npz work\emb_test__videomae-layer8.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__alexnet-fc6.npz work\emb_test__alexnet-fc6.npz --set model.multi_head=true model.head_names=[videomae,videomae_layer4,videomae_layer8,clip,alexnet] --out late_fusion_videomae_layers4_8_clip_alexnet
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\late_fusion_videomae_layers4_8_clip_alexnet\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_late_fusion_videomae_layers4_8_clip_alexnet --head-weights videomae=1 videomae_layer4=1 videomae_layer8=1 clip=1 alexnet=1
```

To augment rather than replace the static CLIP representation, add both
`work\emb_train__clip-vitb32-layers6-12.npz` and
`work\emb_test__clip-vitb32-layers6-12.npz` before the CLAP archives in
`--video-emb`. The first run downloads the VideoMAE checkpoint from Hugging
Face; model inference requires a 16-frame clip.

#### AlexNet fc6 features

The video extractor can also save the ImageNet-pretrained AlexNet fc6
activations (the first fully connected layer after its ReLU) averaged over
sampled frames. The first run may download torchvision's pretrained weights:

```powershell
python "src\extract embeddings\extract_video_embeddings.py" --backend alexnet --model alexnet --videos stimuli_train --out work\emb_train__alexnet-fc6.npz --device cuda
python "src\extract embeddings\extract_video_embeddings.py" --backend alexnet --model alexnet --videos stimuli_test --out work\emb_test__alexnet-fc6.npz --device cuda
```

Include both AlexNet archives with the VideoMAE and CLIP archives, enable
multi-head mode, and explicitly name the three heads to train the expanded
late-fusion ensemble:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__alexnet-fc6.npz work\emb_test__alexnet-fc6.npz --set model.multi_head=true model.head_names=[videomae,clip,alexnet] train.xsubj_avg.prob=0 --out late_fusion_videomae_clip_alexnet
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\late_fusion_videomae_clip_alexnet\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_late_fusion_videomae_clip_alexnet --head-weights videomae=1 clip=1 alexnet=1
```

This is an independent AlexNet prediction head fused after its own CSLS
scoring, not an AlexNet vector concatenated into either existing feature space.
AlexNet is now included in the default multi-head configuration, so training
that default requires the extracted AlexNet archives. To reproduce an older
two-head run, explicitly set `model.head_names=[videomae,clip]`.

#### Joint pretraining followed by per-subject adapter fine-tuning

Stage one trains the shared encoder jointly across all six subjects and
reserves the same global set of training video IDs for validation. Stage two
starts from that checkpoint and updates only each target subject's channel
adapter at a smaller learning rate. Reusing the global held-out video IDs for
stage-two early stopping avoids validating on videos already seen by the
joint stage. Stage-one checkpoints created before this workflow recorded
those validation IDs are not eligible; train a fresh checkpoint first.

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__alexnet-fc6.npz work\emb_test__alexnet-fc6.npz --set model.multi_head=true model.head_names=[videomae,clip,alexnet] train.xsubj_avg.prob=0 train.xsubj_contrastive_weight=0.1 --out eb_joint_stage1
python -m src.xsubj.train --finetune models\xsubj\checkpoints\eb_joint_stage1\model.pt --subjects all --finetune-lr 0.00005 --finetune-epochs 50 --finetune-patience 8 --out eb_joint_stage2_all
python -m src.xsubj.submit --model encoder --checkpoint models\xsubj\checkpoints\eb_joint_stage2_all\model.pt --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 --out work\submissions_eb_joint_stage2_all
```

The auxiliary loss in stage one uses same-video trials from different subjects
as positives, and different-video trials as negatives. Its batches retain all
trials for each selected video, ensuring cross-subject pairs are present.
Stage two has only one subject per adaptation pass, so it uses the original
video-retrieval loss and fine-tunes that subject's adapter only. The final
checkpoint retains a separate adapted adapter for each subject; test trials
are only used for evaluation. Set `train.xsubj_contrastive_weight=0` to
disable the auxiliary objective.

#### Patch-transformer EEG encoder (experimental, not in current ensemble)

The optional `patch_transformer` encoder uses the same preprocessed EEG window
as EEGNet so their rankings can be fused. With the current 0–1 s, 128 Hz
preprocessing, the 50/25-sample patch and stride settings are scaled from
250 Hz to 26/13 samples, producing 8 overlapping tokens. Each token is
projected to 256 dimensions, passed through a 3-layer, 8-head Transformer, and
combined by learned attention pooling.

Train a separate transformer checkpoint against the same three target spaces:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__alexnet-fc6.npz work\emb_test__alexnet-fc6.npz --set model.arch=patch_transformer model.multi_head=true model.head_names=[videomae,clip,alexnet] model.transformer_dim=256 model.transformer_heads=8 model.transformer_layers=3 model.patch_length=50 model.patch_stride=25 model.patch_reference_fs=250 train.xsubj_avg.prob=0 --out patch_transformer_videomae_clip_alexnet
```

This architecture was tested separately and underperformed the existing
EEGNet on labeled subjects 01–04 (3.7%/14.5% mean top-1/top-5). It is not
included in the current EEGNet predictions.

#### Stronger training-only EEG augmentation

The training dataset can optionally apply same-video cross-subject mixup,
continuous time masking, channel masking, and stronger circular time shifts.
The current input is 1 s at 128 Hz, so 50–100 ms masks cover about 6–13
samples and a 50 ms maximum shift is about 6 samples. Validation and test
trials are not augmented. For an augmentation experiment with EEGNet, use the same feature archives as above
and add these overrides:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__alexnet-fc6.npz work\emb_test__alexnet-fc6.npz --set model.multi_head=true model.head_names=[videomae,clip,alexnet] train.xsubj_avg.prob=0 train.aug.time_shift=0.05 train.aug.time_mask_prob=0.3 train.aug.time_mask_min_ms=50 train.aug.time_mask_max_ms=100 train.aug.channel_mask_prob=0.2 train.aug.channel_mask_min=2 train.aug.channel_mask_max=4 train.aug.same_video_mixup_prob=0.5 train.aug.same_video_mixup_alpha=0.4 --out eegnet_augmented_videomae_clip_alexnet
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\eegnet_augmented_videomae_clip_alexnet\model.pt --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 --out work\submissions_eegnet_augmented_videomae_clip_alexnet --head-weights videomae=1 clip=1 alexnet=1
```

This submits rankings directly from the augmented EEGNet checkpoint: no
checkpoint-level rank fusion is applied. Its three feature heads still use the
existing weighted CSLS late fusion. Do not use sub-05/06 labels to choose
augmentation strengths or fusion settings. This run reached validation rank
score 0.736. On labeled sub-01–04, it scored 8.8%/31.6% top-1/top-5 versus
12.7%/40.0% for the existing EEGNet checkpoint, so these augmentation
strengths did not improve this single-seed result.

To retrain the current best combined feature setup and score its checkpoint on
the labeled test trials, run these commands. Use a new `--out` name for every
run so you do not overwrite a previous checkpoint:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clap.npz work\emb_test__clap.npz --out clip_layers6_12_videomae_rerun
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\clip_layers6_12_videomae_rerun\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_clip_layers6_12_videomae_rerun
```

The first command trains and saves the model. The second uses that exact
checkpoint to rank the 102 candidates for each subject and reports top-1 and
top-5 against the available test labels. Submission uses candidate
column-normalization by default; add `--no-col-norm` to disable it.

To compare Cross-Domain Similarity Local Scaling (CSLS, k=10) against raw
cosine and column-normalized ranking during evaluation, use:

```powershell
python -m src.xsubj.eval --config configs/xsubj.yaml --models encoder --mode official --video-emb work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clap.npz work\emb_test__clap.npz
```

Evaluation reports all three ranking methods and their permutation-test
results. To create candidate rankings with CSLS from a saved checkpoint, add
`--csls` to the submission command:

```powershell
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\clip_layers6_12_videomae_rerun\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_clip_layers6_12_videomae_csls --csls
```

CSLS uses the other test EEG trials for local-density correction; add the flag
to compare it with the default column-normalized ranking.

A validation-selected follow-up experiment disables cross-subject trial
averaging while keeping the same CLIP layer 6/12, VideoMAE, and CLAP features.
In one matched run, its held-out training-video rank score was 0.746 versus
0.735 for the default augmentation. On labeled subjects, its CSLS average
top-1/top-5 was 12.25%/35.54%, compared with 12.01%/34.56% for the existing
CSLS checkpoint; this is a small single-seed gain, not yet evidence of a
reliable improvement. Reproduce it without overwriting prior runs:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clap.npz work\emb_test__clap.npz --set train.xsubj_avg.prob=0 --out rankall_no_xsubjavg_20261017
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\rankall_no_xsubjavg_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_rankall_no_xsubjavg_20261017 --csls
```

### Audio-free, two-head CSLS late fusion

The default late-fusion setup uses one shared EEG backbone with separate
normalized output heads for VideoMAE and CLIP (including its combined
intermediate-layer features). Each head learns against its own feature space.
At submission time, CSLS is computed separately for each head and the score
matrices are combined using normalized weights before ranking. Weights
default to 1:1 and can be adjusted at inference with `--head-weights`.

Train with the VideoMAE and CLIP feature archives and select just those two
heads explicitly:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz --set model.multi_head=true model.head_names=[videomae,clip] train.xsubj_avg.prob=0 --out late_fusion_videomae_clip_20261017
```

Score labeled subjects 01–04 and write ranked candidate files:

```powershell
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\late_fusion_videomae_clip_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_late_fusion_videomae_clip_20261017 --head-weights videomae=1 clip=1
```

Both visual heads always use CSLS with `k=10`; their score matrices are
combined after CSLS, so the checkpoint can rank candidates without depending
on embedding dimensions matching. Test labels are only used to report the
submission scores, not to train the encoder.

#### Optional CLAP head

CLAP remains excluded by default. To test it as an independent head, include
its train/test archives and configure the explicit head list. It can be used
as a third head without AlexNet or as a fourth head with the default
VideoMAE+CLIP+AlexNet ensemble. This requires training a new checkpoint. The
command below combines CLAP with the SE channel-attention experiment; omit
`model.channel_attention=true` if you want to test audio without that change.
The score-fusion weight is independently adjustable at submission:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__clap.npz work\emb_test__clap.npz --set model.multi_head=true model.head_names=[videomae,clip,clap] model.channel_attention=true train.xsubj_avg.prob=0 --out spatial_attention_videomae_clip_clap_20261017
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\spatial_attention_videomae_clip_clap_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_spatial_attention_videomae_clip_clap_20261017 --head-weights videomae=1 clip=1 clap=1
```

Omit the CLAP archives and `model.head_names` setting for the default
audio-free two-head experiment. To disable CLAP's contribution at scoring time
without retraining, use `--head-weights videomae=1 clip=1 clap=0`. For a fair
comparison, train a two-head and three-head model with the same seed, attention
setting, and other training parameters, and compare on the same subjects.

### Optional EEG test-time augmentation

The challenge loader and current preprocessing config provide EEG from
0–1.0 s at 128 Hz, not a 2.8-second window. TTA therefore uses overlapping
0.8-second crops starting at 0, 100, and 200 ms, averages their normalized
predictions (independently per visual head), renormalizes, then applies CSLS.
Enable it for the two-head late-fusion run with:

```powershell
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\late_fusion_videomae_clip_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_late_fusion_videomae_clip_tta_20261017 --head-weights videomae=1 clip=1 --tta-offsets 0 0.1 0.2 --tta-duration 0.8
```

Omit `--tta-offsets` to use the original full-window single-pass prediction.
Offsets and duration are seconds relative to the start of the available
preprocessed window; each crop must fit completely within that window.

### Optional squeeze-and-excitation channel attention

`model.channel_attention=true` adds a squeeze-and-excitation gate immediately
after the first temporal EEG convolution. It learns weights over the first
convolution's feature maps. The current challenge data has 17 EEG channels but
does not include channel names, so this experiment cannot reliably apply a
fixed occipital/parietal electrode mask; the learned gate is over convolutional
features rather than named scalp electrodes. This changes the model
architecture, so train a fresh checkpoint:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz --set model.multi_head=true model.head_names=[videomae,clip] model.channel_attention=true train.xsubj_avg.prob=0 --out spatial_attention_videomae_clip_20261017
```

Score the labeled subjects without TTA first, to isolate the attention
change:

```powershell
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\spatial_attention_videomae_clip_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_spatial_attention_videomae_clip_20261017 --head-weights videomae=1 clip=1
```

### Optional multi-scale temporal filter bank

`model.multi_scale_temporal=true` replaces EEGNet's first temporal
convolution with parallel 64/32/16-sample branches defined at a 250 Hz
reference rate. At the challenge's 128 Hz preprocessing rate, the kernels are
scaled to 33/17/9 samples, then concatenated before the depthwise spatial
convolution. This changes the architecture and requires training a fresh
checkpoint. The command enables both this filter bank and the SE attention
above; omit `model.channel_attention=true` to isolate the temporal-filter
change:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz --set model.multi_head=true model.head_names=[videomae,clip] model.multi_scale_temporal=true model.channel_attention=true train.xsubj_avg.prob=0 --out multiscale_attention_videomae_clip_20261017
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\multiscale_attention_videomae_clip_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_multiscale_attention_videomae_clip_20261017 --head-weights videomae=1 clip=1
```

The multi-scale block currently supports `arch=eegnet`. To compare its
contribution independently from channel attention, train a second run with
`model.channel_attention` omitted and otherwise matching settings.

## 2. Extract video-audio embeddings

The audio extractor uses LAION CLAP by default:

```powershell
python "src\extract embeddings\extract_audio_embeddings.py" --videos stimuli_train --out work\train_clap.npz --backend laion_clap --model HTSAT-tiny --device cuda --silence-rms 0.0001
python "src\extract embeddings\extract_audio_embeddings.py" --videos stimuli_test --out work\test_clap.npz --backend laion_clap --model HTSAT-tiny --device cuda --silence-rms 0.0001
```

Videos without an audio stream or whose decoded waveform RMS is below
`--silence-rms` are retained with a zero audio vector and their IDs/RMS
threshold are recorded in the archive. The default silence threshold is
`1e-4` for normalized float audio. Decode or model errors stop extraction with
the video path in the error. Use `--device cpu` when CUDA is unavailable.

Both extractors save `video_ids` and `embeddings` arrays, along with model
metadata. The current `configs/xsubj.yaml` uses the existing combined
CLIP+CLAP archives `work\emb_train__clip-clap.npz` and
`work\emb_test__clip-clap.npz`. These are legacy archives keyed by video/frame
IDs; `src.xsubj` pools those frames and also accepts the newer
`video_ids`/`embeddings` format. If you regenerate separate image and audio
archives with the commands above, update `data.video_emb` in the config to
list those four files. Separate model spaces are normalized independently and
concatenated at equal norm, and the final vector is normalized as well. If a
video is missing a modality, its available embedding retains unit norm.
Existing output archives are not overwritten; add `--overwrite` when
regenerating an existing path.

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

To train separate EEG projection heads with a video contrastive loss and an
audio contrastive loss, then retrieve using their equal-weight fused
embedding, run:

```powershell
python -m src.xsubj.eval --config configs/xsubj.yaml --models encoder --mode official --set model.dual_projection=true
```

This option expects a combined embedding bank with equal-sized video and audio
spaces concatenated in that order. `train.audio_loss_weight` defaults to 1.0
and can be changed with `--set`; the default single-projection training remains
unchanged.

To learn gated fusion between visual features (CLIP intermediate layers plus
VideoMAE) and CLAP audio features, first extract new silence-filtered CLAP
archives, then train and submit with those separate archives:

```powershell
python "src\extract embeddings\extract_audio_embeddings.py" --videos stimuli_train --out work\emb_train__clap_silenceaware.npz --backend laion_clap --model HTSAT-tiny --device cuda --silence-rms 0.0001
python "src\extract embeddings\extract_audio_embeddings.py" --videos stimuli_test --out work\emb_test__clap_silenceaware.npz --backend laion_clap --model HTSAT-tiny --device cuda --silence-rms 0.0001
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clap_silenceaware.npz work\emb_test__clap_silenceaware.npz --set model.gated_fusion=true model.fusion_dim=512 --out gated_clip_layers_videomae_clap_20261017
python -m src.xsubj.submit --config configs/xsubj.yaml --model encoder --checkpoint models\xsubj\checkpoints\gated_clip_layers_videomae_clap_20261017\model.pt --subjects sub-01 sub-02 sub-03 sub-04 --out work\submissions_gated_clip_layers_videomae_clap_20261017 --csls
```

The gate and separate visual/audio projections are trained jointly with the
EEG encoder. At submission time, the checkpoint transforms the candidate
features with the same learned gate. Gated fusion requires distinct visual
and audio archives; legacy pre-concatenated CLIP+CLAP archives cannot be split
reliably and are rejected for this mode. `model.fusion_dim` sets the shared
retrieval-space size. This is an opt-in alternative to the default equal-norm
feature concatenation. In the first silence-filtered gated run, mean CSLS
top-1/top-5 on sub-01..04 was 8.82%/26.72%, below the
`rankall_no_xsubjavg_20261017` reference at 12.25%/35.54%. Keep the reference
checkpoint as the current best; gated fusion is experimental and should not
replace it without a repeatable gain.

## 4. Train the final EEG encoder

After choosing the model settings, train on the labeled training trials from
all six subjects:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --out branchmerge_20261017
```

This run saves to
`models\xsubj\checkpoints\branchmerge_20261017\model.pt`. Without `--out`,
the checkpoint is saved to `models\xsubj\checkpoints\all\model.pt`.
Configuration overrides for an experiment can be passed to evaluation with
`--set`, for example:

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
| 2026-10-18 | Historical experiment: mean-rank fusion of complementary spaces reached top-1 11.76% / top-5 38.24% on the sub-01 holdout, but checkpoint-level rank fusion is no longer part of the supported workflow. |
|---|---|
