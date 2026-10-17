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

When enabled, Euclidean alignment estimates a trace-normalized mean spatial
covariance separately for each subject and split, including each held-out
subject's unlabeled test trials, and applies its inverse square root. Changing
this preprocessing requires retraining the encoder; its cache key is versioned
to avoid reusing arrays produced by the previous covariance formula. The cache
also tracks challenge test-label contents, so adding labels after preprocessing
makes them visible to evaluation without changing EEG normalization.

## Components

- `eeg/adapter.py`: challenge and synthetic EEG data loading.
- `eeg/preprocess.py`: per-subject/split signal preprocessing and caching.
- `eeg/dataset.py`: trial/label alignment with the video embedding bank.
- `video/embeddings.py`: standardized archive loading, model-space fusion,
  and video-ID lookup.
- `nets/encoders.py`: EEGNet, shallow, and MLP encoders plus subject-specific
  channel mixing, optional squeeze-and-excitation attention and multi-scale
  temporal filters, and separate VideoMAE/CLIP projection heads.
- `nets/ridge.py`: Ridge decoder and ERP/bandpower/covariance features.
- `train.py`: contrastive EEG encoder training and checkpoint handling.
- `eval.py`: labeled retrieval evaluation.
- `submit.py`: ranked candidate generation.

The contrastive logit scale is learnable, initialized from `train.temperature`
(default `0.07`), and bounded so its effective temperature stays in
`[0.01, 0.10]`.

The default config points at the existing combined CLIP+CLAP archives in
`work/`. The loader accepts both these legacy per-video/frame-key archives
(pooling frames as needed) and the newer standardized `video_ids` plus
`embeddings` format. If using separate standardized image/audio archives,
list the train and test files for each model under `data.video_emb`; distinct
model spaces are normalized independently and concatenated at equal norm.
The final vector is normalized too, so a missing modality does not reduce the
available embedding's norm.

For late fusion, `model.multi_head=true` defaults to separate VideoMAE, CLIP,
and AlexNet fc6 heads. Set `model.head_names` explicitly to choose another
combination, such as `[videomae,clip]` or `[videomae,clip,clap]`, and supply a
separate archive for each feature space and split. The submission path computes
CSLS in each head's own space, then combines the score matrices with normalized
configurable weights; see the root README for feature extraction, retraining,
and scoring commands.

The VideoMAE extractor can also save one selected intermediate transformer
block per archive using `--intermediate-layers N`. These archives are exposed as
independent heads named `videomae_layerN`; include both train and test archives
and list each name in `model.head_names`. This lets final-layer VideoMAE and
earlier/middle representations (for example, layers 4 and 8) participate
independently in late fusion.

## Joint pretraining and subject-adapter fine-tuning

The first stage jointly trains the shared encoder and one channel-mixing
adapter per subject. It reserves a global set of training video IDs for
validation. Stage two fine-tunes only the selected subject's adapter at a
smaller learning rate, using that same globally held-out video set for early
stopping; the shared encoder and other subject adapters remain frozen. This
checkpoint metadata is required for fine-tuning, so create the joint checkpoint
with the current `train` command before running stage two.

Example using the VideoMAE, CLIP, and AlexNet target spaces:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --video-emb work\emb_train__videomae.npz work\emb_test__videomae.npz work\emb_train__clip-vitb32-layers6-12.npz work\emb_test__clip-vitb32-layers6-12.npz work\emb_train__alexnet-fc6.npz work\emb_test__alexnet-fc6.npz --set model.multi_head=true model.head_names=[videomae,clip,alexnet] train.xsubj_avg.prob=0 train.xsubj_contrastive_weight=0.1 --out eb_joint_stage1
python -m src.xsubj.train --finetune models\xsubj\checkpoints\eb_joint_stage1\model.pt --subjects all --finetune-lr 0.00005 --finetune-epochs 50 --finetune-patience 8 --out eb_joint_stage2_all
python -m src.xsubj.submit --model encoder --checkpoint models\xsubj\checkpoints\eb_joint_stage2_all\model.pt --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 --out work\submissions_eb_joint_stage2_all
```

In stage one, each batch includes complete video groups so that the auxiliary
loss can align same-video embeddings from different subjects and repel
different-video embeddings. Stage two adapts each subject using the original
video-retrieval objective because a single-subject pass has no cross-subject
positive pairs. The default auxiliary weight is zero; set
`train.xsubj_contrastive_weight=0.1` to enable it. Stage-two test labels are
never used for training.

Subject-adapter dropout and EEG-prototype alignment are optional training
guards/objectives. Adapter dropout routes a configurable fraction of stage-one
training trials through the mean (unseen-subject) adapter while retaining the
true subject IDs for the cross-subject loss. EEG prototypes are built from
training trials only, averaging same-video EEG from other subjects while
excluding all trials of the target subject; validation videos and `tst` trials
never contribute to prototype targets. Both features default off.

Example E-E run using the existing combined CLIP+CLAP target space:

```powershell
python -m src.xsubj.train --config configs/xsubj.yaml --set model.eeg_prototype_head=true train.eeg_prototype_weight=0.1 train.subject_dropout=0.1 train.xsubj_contrastive_weight=0.1 --out eb_ee_prototypes
```

To late-fuse an encoder with independent frozen checkpoint scores and a Ridge
decoder trained on a frozen encoder's shared EEG features:

```powershell
python -m src.xsubj.submit --model encoder --checkpoint models/xsubj/checkpoints/eb_joint_stage2_all/model.pt --fusion-checkpoints models/xsubj/checkpoints/shallow_cross_subject_tau007/model.pt --ridge-feature-checkpoint models/xsubj/checkpoints/shallow_cross_subject_tau007/model.pt --subjects sub-01 sub-02 sub-03 sub-04 sub-05 sub-06 --out work/submissions_ed_score_fusion
```

This E-D path requires exactly matching EEG preprocessing and data-source
settings across checkpoints. It standardizes each member's similarity scores
per trial before a weighted sum; it does not combine rank values or use Borda
counting. The `shallow_cross_subject_tau007` checkpoint is the locally trained
replacement used in this workspace; it is not the teammate checkpoint at
`models/xsubj/checkpoints/all/model.pt`.

The optional `model.channel_attention=true` flag gates learned feature maps
after the first temporal convolution and requires a fresh checkpoint. It does
not select named scalp electrodes because the challenge files have no channel
name metadata. `model.multi_scale_temporal=true` replaces EEGNet's first
temporal convolution with parallel filters, scaled to the configured sampling
rate, and also requires a fresh checkpoint; see the root README for the
reference kernels and experiment commands.
