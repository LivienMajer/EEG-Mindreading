# Final solution: per-subject InfoNCE votes and ablation

This section adds two per-subject contrastive votes to the second-round ensemble of
[report_eeg_encoding.md](report_eeg_encoding.md) (section 8) and ablates the final pipeline. All numbers
are on the test splits of the four development subjects sub-01..04 (102 recordings each, 102
candidates), with every model trained on the training splits only. Every choice in this section
(votes, weights, Sinkhorn τ, number of seeds) was made by leave-one-subject-out (LOO) over sub-01..04:
chosen on three subjects, scored on the fourth.

## Final pipeline

```
video ─► frozen CLIP ViT-B/32 ──► image vote:  9 cross-subject encoders + pooled ridge + per-subject ridge ─┐
video ─► AlexNet conv3 ─────────► ridge vote (pooled 0.25 + per-subject)                                   │
video ─► VideoMAE layer 8 ──────► ridge vote                                                               ├─► weighted sum of
sound ─► CLAP / loudness ───────► two ridge votes                                                          │   cosine terms
video ─► AlexNet fc6 ───────────► InfoNCE vote: per-subject EEGNet, 3 seeds          (new)                 │   (one global z-score
video ─► VideoMAE layer 8 ──────► InfoNCE vote: per-subject EEGNet, 3 seeds          (new)                 ┘    each) ─► Sinkhorn ─► ranking
```

| Vote | Weight | Model |
|---|---|---|
| CLIP image ensemble | 0.75 | 5× ShallowConvNet, 3× MLP, 1× EEGNet (cross-subject, subject layer) + ridges |
| AlexNet conv3 | 1.0 | ridge, pooled × 0.25 + per-subject |
| VideoMAE l8 | 1.0 | ridge |
| CLAP audio | 0.3 | ridge |
| Loudness envelope | 0.5 | ridge |
| **AlexNet fc6** | **0.5** | **per-subject InfoNCE EEGNet, 3 seeds averaged** |
| **VideoMAE l8** | **0.25** | **per-subject InfoNCE EEGNet, 3 seeds averaged** |

Sinkhorn τ = 0.3. The two InfoNCE weights were chosen on a grid of 0 / 0.25 / 0.5 per vote; all four LOO
folds picked the same pair.

**Result on sub-01..04:** top-1 **44.9 %**, top-5 **76.7 %**, mean rank **4.36** (chance 1.0 % / 4.9 % / 51.5).
The strict LOO estimate equals the in-sample value (every fold chose the same weights). Against the second
round without the InfoNCE votes: 40.9 % / 74.8 % / 4.85.

## The InfoNCE vote

`src/xsubj/infonce_member.py`. One small EEGNet per subject (temporal conv 16, depthwise spatial conv,
separable conv 8, projection head to 256 d) maps a nearly raw trial (per-trial channel z-score, 100 Hz,
50–350 ms) into a PCA-256 space of a video model. The PCA is fitted on the 1,000 training videos only.
It is trained with symmetric InfoNCE (τ 0.1, AdamW 3e-4, cosine schedule, 200 epochs, batch 256) against
the subject's own training trials, and has no cross-subject training and no subject layer. At test time
the 102 candidates are projected through the same PCA, and the cosine similarities of three seeds are
averaged.

Why it adds to the ensemble, although each vote alone is weaker than the ridges (table below):

- It is a different model class: nonlinear, per subject, contrastive on a short early window. The ridges
  are linear and use 50–600 ms, and the CLIP encoders are shared across subjects. Its errors are only
  partly shared with the other votes.
- Its targets are mid-level visual features (fc6, VideoMAE l8). These are the targets that work best
  for these recordings (report section 7).
- **Seed averaging matters.** Single networks are noisy. One seed gives 43.9 / 77.0 / 4.43, three seeds
  give 44.9 / 76.7 / 4.36 in-sample. Under LOO, one seed gives 43.4 / 77.2 / 4.47 against 44.9 / 76.7 /
  4.36 with three seeds.

## Ablation of the final pipeline (sub-01..04)

| Configuration | top-1 % | top-5 % | mean rank | top-1 per subject 01 / 02 / 03 / 04 |
|---|---|---|---|---|
| **Final** | **44.9** | **76.7** | **4.36** | 52.9 / 47.1 / 33.3 / 46.1 |
| without Sinkhorn | 28.9 | 61.0 | 6.96 | 31.4 / 29.4 / 20.6 / 34.3 |
| without CLIP image ensemble | 41.7 | 71.6 | 4.93 | 52.9 / 43.1 / 29.4 / 41.2 |
| without AlexNet conv3 ridge | 39.0 | 73.5 | 4.98 | 47.1 / 41.2 / 26.5 / 41.2 |
| without VideoMAE l8 ridge | 42.9 | 76.5 | 4.44 | 50.0 / 44.1 / 36.3 / 41.2 |
| without CLAP audio | 41.2 | 76.2 | 4.56 | 50.0 / 42.2 / 29.4 / 43.1 |
| without loudness | 42.6 | 74.5 | 4.69 | 52.0 / 41.2 / 35.3 / 42.2 |
| without InfoNCE fc6 | 42.2 | 76.0 | 4.65 | 52.9 / 41.2 / 32.4 / 42.2 |
| without InfoNCE VideoMAE l8 | 41.9 | 75.7 | 4.50 | 52.9 / 42.2 / 29.4 / 43.1 |
| without both InfoNCE votes (second round) | 40.9 | 74.8 | 4.85 | 49.0 / 39.2 / 33.3 / 42.2 |

| Vote alone (with Sinkhorn) | top-1 % | top-5 % | mean rank |
|---|---|---|---|
| CLIP image ensemble | 21.1 | 51.7 | 9.91 |
| … its 9 cross-subject encoders | 15.0 | 39.5 | – |
| … its pooled ridge | 12.7 | 36.7 | – |
| … its per-subject ridge | 17.4 | 46.6 | – |
| AlexNet conv3 ridge | 25.0 | 54.9 | 9.93 |
| VideoMAE l8 ridge | 25.0 | 52.2 | 9.05 |
| InfoNCE fc6 (3 seeds) | 18.6 | 48.3 | 11.46 |
| InfoNCE VideoMAE l8 (3 seeds) | 17.2 | 47.5 | 12.19 |
| CLAP audio ridge | 4.4 | 22.3 | 24.76 |
| Loudness ridge | 1.2 | 10.8 | 31.91 |

| Sinkhorn τ | 0.1 | 0.2 | **0.3** | 0.5 | 1.0 |
|---|---|---|---|---|---|
| top-1 / top-5 % | 42.6 / 75.7 | 44.4 / 76.7 | **44.9 / 76.7** | 44.6 / 76.5 | 41.9 / 75.7 |

Readings:

- **Every vote contributes.** Removing any one costs 2–6 points of top-1. The largest single
  contributions come from Sinkhorn (+16 points), the AlexNet conv3 ridge (+5.9) and the CLIP image ensemble
  (+3.2). The two InfoNCE votes together add +4.0 top-1 and +1.9 top-5.
- **Mid-level visual targets beat the semantic one.** A single ridge into AlexNet conv3 or VideoMAE l8
  space beats the whole CLIP ensemble of nine encoders and two ridges on top-1 (25.0 vs 21.1 %).
- **Sound carries independent information.** CLAP and loudness are weak alone but cost 3.7 and 2.3 points
  of top-1 when removed, consistent with the early auditory response in the RSA analysis.
- **Sinkhorn is a property of this benchmark.** It exploits that every candidate occurs exactly once per
  test split. It is label-free, but a deployed decoder that sees one recording at a time could not use
  it. Without it the same ensemble reaches 28.9 / 61.0 / 6.96.
- **sub-03 is the hardest subject** in almost every configuration (top-1 26–36 %, against 47–53 % for
  sub-01). It also has the weakest stimulus-locked signal among the dev subjects (below).

## What did not help (sub-01..04)

Tested on the earlier first-round ensemble plus four single-seed InfoNCE votes (fc6, VideoMAE l8, AlexNet
features.3, CLIP; per-candidate normalised scores, no Sinkhorn; base 38.0 / 68.9 / 6.3), one change at a
time on the fc6 and VideoMAE votes:

| Variant of the InfoNCE vote | Ensemble top-1 / top-5 / rank | Vote alone (VideoMAE l8) top-1 |
|---|---|---|
| plain InfoNCE (used) | 38.0 / 68.9 / 6.3 | 13.7 |
| soft targets from video self-similarity, α 0.25 / 0.5 | 38.0 / 68.9 / 6.3 · 37.7 / 68.9 / 6.3 | 13.7 |
| same, own video masked out of the soft targets, α 0.1 / 0.25 | 37.7 / 68.4 / 6.3 · 37.5 / 68.1 / 6.3 | 13.7 / 14.0 |
| positive weighting by EEG–video similarity (+ soft α 0.5) | 34.8 / 68.9 / 6.5 | 14.5 |
| joint EEG / video / audio alignment, α 0 | 34.1 / 68.6 / 6.5 | 9.8 |
| joint alignment with cycle-consistent soft targets, α 0.25 | 33.6 / 68.1 / 6.5 | 10.8 |
| + the joint model's EEG→audio vote (weight 0.25 / 0.5) | 38.5 / 68.4 / 6.3 · 37.0 / 66.2 / 6.4 | – |

The variants follow the robust multi-modal alignment of Lerch, Majer et al. (2026, arXiv:2606.02352):
soft targets against faulty negatives, similarity-based weights against faulty positives, and global
alignment over all modality pairs with a frozen anchor.

- **Soft targets do nothing here.** With targets from the video features' self-similarity, the own video
  dominates the softmax, so the targets stay almost the identity. Masking it out moves weight to similar
  videos but does not help either. Faulty negatives are not what limits this task. With ~950 noisy
  trials per subject, the bottleneck is the EEG signal.
- **Positive weighting hurts** (−3.2 top-1). With EEG, all positive similarities are low and close
  together. The weights then mostly favour videos with strong evoked responses instead of discarding
  noisy trials. In this run the weighting was also active from the first epoch, before the similarities
  carry information.
- **Joint EEG / video / audio alignment** weakens the video vote (13.7 → 9.8 %): the small per-subject
  network splits its capacity between two targets. Cycle-consistent soft targets (from the audio–video
  similarity, which has no dominant diagonal) recover part of it (→ 10.8 %). The joint model's EEG→audio
  vote is better than the CLAP ridge alone (7.6 vs 4.7 % top-1) but adds nothing to the ensemble.
- **InfoNCE toward AlexNet conv3** (one seed) as a third vote: +0.2 top-1, less stable weight choice
  across folds. Not used.
- **Further ridge targets** (AlexNet conv4, conv5, fc6, fc7 as additional ridge votes): no LOO gain
  (best: fc7 +0.4 points of the top-1/top-5 criterion, within noise).
- **Cross-subject encoders toward conv3 / VideoMAE l8** instead of CLIP: the VideoMAE l8 encoders lower
  the ensemble (criterion 47.3 → 44.9 / 45.5); conv3 encoders were not finished.

## How weights should be chosen: more subjects beat similar subjects

sub-03 and sub-04 are weaker than sub-01 and sub-02. Is it better to choose weights on subjects that
resemble the target? We chose the five ridge/image weights from a grid of 405 combinations on different
subsets and scored the choice on sub-03/04:

| Weights chosen on … | Scored on sub-03 / sub-04: top-1 / top-5 / rank |
|---|---|
| all four dev subjects (used) | **37.7 / 73.0 / 4.93** |
| three dev subjects including one weak (LOO) | 35.8 / 71.6 / 5.16 |
| the two strong subjects sub-01/02 | 33.8 / 71.1 / 5.56 |
| the other weak subject only (sub-03 → sub-04 and back) | 31.9 / 72.5 / 5.56 |
| (sub-03/04 themselves, in-sample: upper bound) | 38.7 / 74.5 / 5.01 |

With 102 recordings per subject, the amount of data used for the choice matters more than how similar
the subjects are. The chosen weights jump between subsets (audio 0–0.3, conv3 0.5–1.5, loudness 0–1),
so the optimum is flat and noisy. Choosing on all four dev subjects is the most robust option.

## Signal quality per subject (training data only)

Shared stimulus-locked signal: correlation of two subjects' responses to the same training video minus
different videos (50–350 ms, 100 Hz, split mean removed), averaged over the five partners:

| Subject | 01 | 02 | 03 | 04 | 05 | 06 |
|---|---|---|---|---|---|---|
| training trials | 968 | 972 | 961 | 947 | 952 | 850 |
| shared signal r | 0.112 | 0.104 | 0.087 | 0.084 | 0.088 | 0.073 |

sub-01 and sub-02 carry the strongest shared signal, sub-03 and sub-04 less, matching their lower test
accuracy. sub-06 has the weakest signal and the fewest training trials of all six subjects. Test
recordings have 0.6–0.7× the amplitude of training trials and behave like averages of about two
repetitions (feature/eeg-preprocessing, `check_test_averaging.py`). The ridge regularisation is relaxed
for that reason (section 8, A3).

## Limitations

- **Selection on the dev subjects.** Over two days the team compared many members and settings on the
  same 408 dev recordings. LOO protects each weight choice but not the choice of which members exist.
  The CLIP encoders went through the most iterations, and their dev numbers are likely the most
  optimistic. Top-1 on 102 recordings has a standard error of about ±4–5 points per subject.
- **Benchmark property.** Sinkhorn and the per-candidate normalisation use the fact that each test split
  holds every candidate once.
- **Compute.** All InfoNCE members train on a CPU in about 1 min per subject and seed. More seeds or
  members were limited by time, not by a measured plateau.

## Reproduce

```bash
python -m src.xsubj.infonce_member --target fc6   --emb models/xsubj/features/rsa/alexnet_fc6.npz --seeds 0 1 2
python -m src.xsubj.infonce_member --target vmae8 --emb models/xsubj/features/rsa/videomae_l8.npz --seeds 0 1 2
python -m src.xsubj.submit          # reads submit.infonce_votes from configs/xsubj.yaml
```

Checkpoints are versioned under `models/xsubj/checkpoints/infonce_{fc6,vmae8}/sub-XX_s{0,1,2}.pt`, so
`submit` runs without retraining. The robust variants above are options of `infonce_member.train`
(`soft_alpha`, `soft_offdiag`, `pos_weight`), all off by default.
