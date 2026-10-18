# EEG Mind Reading: which video was the person watching?

National AI Hackathon 2026 (SchubKI), challenge *Mind Reading Using Machine Learning and EEG*. For each
3 s EEG recording (17 channels, 500 Hz), the pipeline ranks the 102 candidate videos from most to least
likely to have caused it.

**Result on the development subjects sub-01..04** (102 recordings × 102 candidates each, models trained
on training splits only, every setting chosen by leave-one-subject-out):

| | top-1 | top-5 | mean rank |
|---|---|---|---|
| **Final submission** | **44.9 %** | **76.7 %** | **4.36** |
| Organisers' baseline (AlexNet forward model, sub-01) | 3.9 % | 11.8 % | – |
| Chance | 1.0 % | 4.9 % | 51.5 |

## Approach

The EEG is mapped into the embedding spaces of frozen, public video and audio models, and each candidate
is scored by cosine similarity. Seven votes are summed and re-ranked jointly:

```
                   frozen public models                 EEG decoders (trained here)
video ─► CLIP ViT-B/32 ─────────► 9 cross-subject encoders + pooled & per-subject ridge ─┐ 0.75
video ─► AlexNet conv3 ─────────► pooled & per-subject ridge ─────────────────────────────┤ 1.0
video ─► VideoMAE layer 8 ──────► pooled & per-subject ridge ─────────────────────────────┤ 1.0
sound ─► CLAP ──────────────────► pooled & per-subject ridge ─────────────────────────────┤ 0.3   cosine terms
sound ─► loudness envelope ─────► pooled & per-subject ridge ─────────────────────────────┤ 0.5 ─► weighted sum ─► Sinkhorn ─► ranking
video ─► AlexNet fc6 ───────────► per-subject InfoNCE EEGNet, 3 seeds ────────────────────┤ 0.5
video ─► VideoMAE layer 8 ──────► per-subject InfoNCE EEGNet, 3 seeds ────────────────────┘ 0.25
```

Main findings (details in the reports):

- The decodable signal is the **phase-locked onset response, 50–600 ms**. Power and covariance features
  carry nothing.
- **Mid-level visual features** (AlexNet conv3–fc7, VideoMAE layer 8) are better targets than CLIP. A
  single ridge into them beats the whole CLIP ensemble.
- **The sound is decoded too.** Loudness and CLAP add independent information.
- **Per-subject models matter**: a ridge on one subject's ~950 trials beats a pooled ridge on 5,650.
- **Less preprocessing is better**: only a 40 Hz low-pass, label-free per-split normalisation and
  Euclidean alignment help.
- **Sinkhorn re-ranking** uses that every candidate occurs once per test split. It is
  label-free, but a property of this benchmark.

## Reports

- [docs/report_eeg_encoding.md](docs/report_eeg_encoding.md): EEG features, encoders, preprocessing,
  video and audio targets, second-round scoring, final configuration, limitations.
- [docs/report_infonce_votes.md](docs/report_infonce_votes.md): per-subject InfoNCE votes, ablation of
  the final pipeline, variants that did not help, how to choose weights, per-subject signal quality.
- [src/xsubj/README.md](src/xsubj/README.md): module map and all commands.

## Setup (Python 3.11)

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Linux/macOS: .venv/bin/python
```

The code runs on CPU and uses CUDA if available. Put the organisers' download here (`data/` is gitignored):

```
data/eeg-mind-reading-challenge/
├── trn/sub-XX_trn.npz            training EEG, {video id: 17 x 1500}
├── tst/sub-XX_tst.npy            test EEG, 102 x 17 x 1500
├── tst_stim/sub-XX_tst_stim.npy  test labels, sub-01..04 only
└── stimuli_train/  stimuli_test/ videos (only needed to re-extract features)
```

## Reproduce the submission

```bash
python -m src.xsubj.submit
```

This writes `submissions/sub-05_ranked_candidate_list.npy` and `sub-06_…` (int, 102 × 102; row *i* =
*i*-th recording of `tst/sub-XX_tst.npy` in its original order, video IDs 1001–1102 from most to least
likely) and scores sub-01..04 as a check. All model weights and video features are versioned, so no
training or feature extraction is needed. The first run preprocesses the EEG once (cache under
`models/xsubj/cache/`).

Tested from a fresh clone with a fresh Python 3.11.15 environment from `requirements.txt` (CPU, about 5 min):
the dev scores and sub-05 match exactly; sub-06 differs only by two swaps of neighbouring candidates at
ranks 82–85 (floating-point near-ties), with identical top-1 and top-5 in every row.

Retraining (optional): the commands for the 9 CLIP encoders are listed under `submit:` in
[configs/xsubj.yaml](configs/xsubj.yaml). The InfoNCE votes are retrained with:

```bash
python -m src.xsubj.infonce_member --target fc6   --emb models/xsubj/features/rsa/alexnet_fc6.npz --seeds 0 1 2
python -m src.xsubj.infonce_member --target vmae8 --emb models/xsubj/features/rsa/videomae_l8.npz --seeds 0 1 2
```

## Pre-trained models used (public, released before the hackathon)

Only frozen features of these models are used. Their weights are not part of this repository.

| Use | Model | Exact identifier | Library |
|---|---|---|---|
| Image embedding (CLIP target) | CLIP ViT-B/32 | `ViT-B-32`, pretrained `openai` | open_clip_torch 3.3.0 |
| Mid-level image features (conv3, fc6 targets) | AlexNet | `torchvision.models.alexnet`, weights `IMAGENET1K_V1` | torchvision |
| Video features (layer 8 target) | VideoMAE base, Kinetics fine-tuned | `MCG-NJU/videomae-base-finetuned-kinetics` | transformers |
| Audio embedding (CLAP target) | CLAP HTSAT | `laion/clap-htsat-unfused` | transformers |
| Loudness envelope | none (log RMS of the sound track, 100 ms bins) | – | – |

Feature extraction: `src/xsubj/video/clip_features.py` (CLIP), `src/xsubj/video/foundation_features.py`
(CLAP), `eeg_pipeline/rsa_features.py` (AlexNet layers, VideoMAE layers, loudness; additionally needs
`decord`). The extracted features are versioned under `models/xsubj/features/`.

## What is versioned

| Path | Content |
|---|---|
| `models/xsubj/features/` | CLIP, CLAP, AlexNet conv3 / fc6, VideoMAE l8 and loudness features of all 1,102 videos |
| `models/xsubj/checkpoints/{all,shallow_s*,mlp_s*,eegnet_s0}/` | the 9 cross-subject CLIP encoders |
| `models/xsubj/checkpoints/ridge*/` | all ridge models (refitted deterministically by `submit`) |
| `models/xsubj/checkpoints/infonce_{fc6,vmae8}/` | per-subject InfoNCE EEGNets, 6 subjects × 3 seeds each |
| `submissions/` | the submitted rankings for sub-05 and sub-06 |


