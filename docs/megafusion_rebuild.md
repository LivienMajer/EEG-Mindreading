# Megafusion: independent rebuild and improvements (17 Oct, Sofiya)

The megafusion (`docs/final_pipeline.md`) was rebuilt from the code alone on a CPU
laptop, without any of Livien's local files. Every member and the final LOO result
reproduce exactly. Two changes then improve it under the same LOO protocol.

## 1. Rebuild (reproducibility check)

```bash
D=<challenge root with trn/ tst/ tst_stim/ stimuli_train/ stimuli_test/>
# video banks in the megafusion's file format (frames averaged at extraction, same
# frames and encoder as src.extract_embeddings; ~1 min CPU for alexnet-f3)
git show origin/feature/clip-lowlevel-forward-model:models/xsubj/features/clip_vitb32.npz > work/clip_vitb32_daniel.npz
python -m src.extract_pooled --convert work/clip_vitb32_daniel.npz --name clip-vitb32 --out-dir work
python -m src.extract_pooled --model alexnet-f3 --stimuli-root $D --out-dir work
# members 1-2 (sub-01..06), 4, 5-7: the team's scripts, unchanged
for sp in clip-vitb32 alexnet-f3; do for s in 01 02 03 04 05 06; do
  python -m src.contrastive_pipeline --data-dir $D --sub $s --emb-train work/emb_train__$sp.npz \
      --emb-test work/emb_test__$sp.npz --epochs 200 --seed 0; done; done
python -m src.bench_embeddings --data-dir $D --emb-train work/emb_train__clip-vitb32.npz \
    --emb-test work/emb_test__clip-vitb32.npz --window snr
python -m src.cross_subject_probe --data-dir $D --emb-train work/emb_train__clip-vitb32.npz \
    --emb-test work/emb_test__clip-vitb32.npz
# member 3 (E-D): Daniel's checkpoint; the megafusion used the version of commit 86fac43.
# Needs his CLIP bank at the path in the checkpoint config (Git Bash: `>` keeps binaries intact)
mkdir -p models/xsubj/features models/xsubj/checkpoints/all_86fac43 models/xsubj/checkpoints/all_5e3154e
git show origin/feature/clip-lowlevel-forward-model:models/xsubj/features/clip_vitb32.npz > models/xsubj/features/clip_vitb32.npz
git show 86fac43:models/xsubj/checkpoints/all/model.pt > models/xsubj/checkpoints/all_86fac43/model.pt
python -m src.ed_member --data-dir $D --checkpoint models/xsubj/checkpoints/all_86fac43/model.pt     --out work/ed_member --dev-sorted-dir work/tst_ranked     # also writes ed_XX_xsubj_s0.npy
python -m src.loo_generalization
```

| Member | Reported (`final_pipeline.md`) | Rebuilt |
|---|---|---|
| 1 InfoNCE CLIP, per subject | 16.7/37.3, 13.7/32.4, 5.9/34.3, 7.8/28.4 | identical |
| 2 InfoNCE alexnet-f3 | 8.8/35.3, 10.8/35.3, 8.8/21.6, 6.9/27.5 | identical |
| 3 E-D (checkpoint 86fac43) | 7.35 / 23.53 | 7.35 / 23.28 (one trial) |
| 4 ridge | 3.43 / 13.73 | identical |
| 5-7 transfers (12 cells) | 3.68 / 12.99, best cell 7.84 | identical (3.68 / 12.99, best cell 7.84) |
| **LOO megafusion** | **22.30 / 48.28**, weights 9:9:5:2 in all folds | **identical** |

Findings on the way:

- The E-D member uses Daniel's **old** checkpoint (0.5-40 Hz band-pass + average
  reference). The retrained one (commit 5e3154e, low-pass only) is stronger solo:
  **11.52 / 31.86** vs 7.35 / 23.28.
- The `src/xsubj` copy on `contrastive-alignment` could not run the new checkpoint
  (`butter` fails on `l_freq: null`). This branch takes `src/xsubj/eeg/preprocess.py` from
  Daniel's branch; it is a backward-compatible superset: the old checkpoint gives
  bit-identical rankings with it.
- `work/tst_ranked/ed_XX_xsubj_s0.npy`, read by `loo_generalization`, was written by no
  script (the label-sorted `work/ed_member/ranked_subXX.npy`); `ed_member --dev-sorted-dir`
  now writes it.

## 2. Improvements (same LOO protocol: weights chosen on 3 dev subjects, scored on the 4th)

New members: the same per-subject InfoNCE EEGNet as members 1-2, trained toward AlexNet
fc6 and VideoMAE layer 8, the most brain-like embeddings in the RSA
(`eeg_pipeline/README.md` on `feature/eeg-preprocessing`):

```bash
python -m eeg_pipeline.rsa_features --stim_dir $D --out_dir work/rsa_features --models alexnet videomae
python -m src.prep_variants --data-dir $D --variants A --emb work/rsa_features/alexnet_fc6.npz --tag _fc6 --subs 01 02 03 04 05 06
python -m src.prep_variants --data-dir $D --variants A --emb work/rsa_features/videomae_l8.npz --tag _vmae8 --subs 01 02 03 04 05 06
# E-D from Daniel's retrained checkpoint
git show 5e3154e:models/xsubj/checkpoints/all/model.pt > models/xsubj/checkpoints/all_5e3154e/model.pt
python -m src.ed_member --data-dir $D --checkpoint models/xsubj/checkpoints/all_5e3154e/model.pt     --out work/ed_member_5e3154e --dev-sorted-dir work/ed_5e3154e
python -m src.loo_generalization --ed-dir work/ed_5e3154e       # new E-D only
python -m src.loo_new_members                                    # + fc6 + VideoMAE-l8, old E-D
python -m src.loo_new_members --ed-dir work/ed_5e3154e           # + both, new E-D
```

`loo_new_members` also prints the paired comparison with the current megafusion.

| Fusion (LOO held-out mean, sub-01..04) | top-1 | top-5 |
|---|---|---|
| current megafusion | 22.30 | 48.28 |
| new E-D checkpoint only | 21.32 | 49.75 |
| + fc6 + VideoMAE-l8 | 22.79 | 52.94 |
| **new E-D checkpoint + fc6 + VideoMAE-l8** | **24.02** | **55.15** |

Last row vs current, paired on the same 408 recordings: top-5 +45 / -17 trials
(exact McNemar p < 0.001), mean rank 12.03 -> 10.12 (better in 205, worse in 99,
Wilcoxon p = 2e-10), top-1 +28 / -21 (p = 0.39, **not significant**). With the old E-D
(third row): top-5 +31 / -12 (p = 0.005), mean rank 10.50, top-1 +12 / -10 (p = 0.83). Chosen weights
9:9:5:2:1:1:1 + fc6 3 : VideoMAE 9 in three folds, 5 : 5 in one.

## 3. Why fc6 and VideoMAE-l8: brain-likeness predicts decoding

```bash
git show origin/feature/eeg-preprocessing:results/rsa/rsa_summary.csv > work/rsa_summary.csv
for f in alexnet_conv2:_conv2 alexnet_fc7:_fc7 videomae_l4:_vmae4 videomae_l12:_vmae12 clap_audio:_clap low_loudness:_loud; do
  python -m src.prep_variants --data-dir $D --variants A --emb work/rsa_features/${f%%:*}.npz --tag ${f##*:} --subs 01 02 03 04; done
python -m src.member_check     --tags _conv2 _fc6 _fc7 _vmae4 _vmae8 _vmae12 _clap _loud --rsa-summary work/rsa_summary.csv
```

Across the 9 target spaces, the RSA peak r of the embedding predicts the decoding
accuracy of the member trained toward it: Spearman rho = 0.69 (top-1, p = 0.038) and
0.70 (top-5, p = 0.036); without the 30-dimensional loudness target (too few dimensions
to tell 102 videos apart) 0.80 (p = 0.017) and 0.81 (p = 0.015).

Seeds (`python -m src.member_check --tags _fc6 --seeds 0 1 2`, needs `prep_variants
--seed 1/2` dumps of the CLIP member): 11.0 / 10.3 / 10.0% top-1; their mean-rank average
9.3 / 35.3% (rank better than seed 0, p = 0.003; top-1 not, p = 0.21).

## 4. Daniel's cross-subject networks toward fc6 / VideoMAE-l8 (best result)

Daniel's pipeline (`src/xsubj`, synced to his current branch; the old copy could not run
his retrained checkpoint, the synced one gives identical E-D rankings), only the video
target changed: `configs/xsubj_fc6.yaml`, `configs/xsubj_vmae8.yaml` are copies of
`configs/xsubj.yaml` with `data.video_emb` = the RSA feature file. 4 encoders each
(ShallowConvNet seeds 0/1, MLP, EEGNet; Daniel's settings), plus his two ridges.

```bash
mkdir -p data && ln -s <challenge root> data/eeg-mind-reading-challenge   # path in the configs
                                                    # (Windows: New-Item -ItemType Junction)
for n in all shallow_s1 shallow_s2 shallow_s3 shallow_s4 mlp_s0 mlp_s1 mlp_s2 eegnet_s0; do
  mkdir -p models/xsubj/checkpoints/$n
  git show origin/feature/clip-lowlevel-forward-model:models/xsubj/checkpoints/$n/model.pt > models/xsubj/checkpoints/$n/model.pt; done
python -m src.xsubj.submit --config configs/xsubj.yaml --out work/xsubj_clip       # Daniel's CLIP ensemble
for t in fc6 vmae8; do c=configs/xsubj_$t.yaml                                     # ~10 min each, CPU
  python -m src.xsubj.train --config $c --out ${t}_shallow_s0
  python -m src.xsubj.train --config $c --set seed=1 --out ${t}_shallow_s1
  python -m src.xsubj.train --config $c --set model.arch=mlp model.input_dropout=0.2 --out ${t}_mlp_s0
  python -m src.xsubj.train --config $c --set model.arch=eegnet --out ${t}_eegnet_s0
  python -m src.xsubj.submit --config $c --out work/xsubj_$t; done
for p in clip:_dclip fc6:_dfc6 vmae8:_dvmae8; do
  python -m src.xsubj_member --src work/xsubj_${p%%:*} --tag ${p##*:} --data-dir $D; done
python -m src.loo_new_members --ed-dir work/ed_5e3154e --tags _fc6 _vmae8 _dfc6 _dvmae8
```

| Daniel-style ensemble (sub-01..04 tst) | encoders | top-1 | top-5 |
|---|---|---|---|
| CLIP (Daniel's submission, reproduced exactly) | 9 | 17.65 | 47.79 |
| **AlexNet fc6** | 4 | **19.36** | **52.45** |
| **VideoMAE layer 8** | 4 | **20.59** | 50.98 |

Megafusion with the new E-D checkpoint + per-subject fc6/VideoMAE members + these two
(LOO, weights for the new members chosen on 3 subjects, scored on the 4th):

| Fusion (LOO held-out mean) | top-1 | top-5 | mean rank |
|---|---|---|---|
| current megafusion | 22.30 | 48.28 | 12.03 |
| + new E-D, fc6, VideoMAE (section 2) | 24.02 | 55.15 | 10.12 |
| **+ Daniel-style fc6 and VideoMAE** | **29.41** | **63.24** | **7.87** |

Per held-out subject: 35.3/66.7, 26.5/58.8, 23.5/60.8, 32.4/66.7. Weights
9:9:5:2:1:1:1 + 5 (fc6) : 9 (VideoMAE) : 9 (Daniel-fc6) : 9 (Daniel-VideoMAE) in three
folds, 9:9:9:5 in one; oracle 30.39 / 62.99. Paired vs current on 408 recordings: top-1
+49 / -20 (exact McNemar p = 0.001), top-5 +78 / -17 (p < 0.001), rank better in 230,
worse in 81 (Wilcoxon p = 9e-22). Most new weights sit at the top of the grid (9).

### Candidate submission for sub-05/06

```bash
python -m src.make_submission --data-dir $D --ed-dir work/ed_member_5e3154e \
    --extra-tags _fc6 _vmae8 _dfc6 _dvmae8 --extra-weights 5 9 9 9 --out-dir work/submission_candidate
python -m src.validate_submission work/submission_candidate/sub-05_ranked_candidate_list.npy $D 05
python -m src.check_alignment --members "work/prep_variants/A_fc6_{s}_s0.npy" "work/prep_variants/A_vmae8_{s}_s0.npy" \
    "work/prep_variants/A_dfc6_{s}_s0.npy" "work/prep_variants/A_dvmae8_{s}_s0.npy" "work/ed_member_5e3154e/ranked_sub{s}.npy"
```

Weights: the LOO choice for 3 dev transfers plus the 4th transfer at 1, as in the current
submission. `check_alignment`: without labels, each new member's rows must agree with
the original members far above chance (e.g. fc6 vs InfoNCE CLIP top-5 overlap 87% vs 24%
with shifted rows); a shuffled member is caught. `make_submission` without the new
options gives bit-identical files to before.

**Expect less on sub-05/06 than on the dev subjects**, mainly sub-06: fewest training
trials (850 vs 947-972) and the least typical responses (RSA consistency with the other
subjects 0.010 / 0.017 at 105 / 225 ms vs 0.019-0.055 for the dev subjects, which
decode better the more typical they are). For the rules, the encoder weights in
`models/xsubj/checkpoints/{fc6,vmae8}_*` (git-ignored here) must be committed if this
submission is used.

## 5. Clean check of the member choice (no test data)

fc6 / VideoMAE-l8 were motivated by the RSA (training data) but their decoding was first
seen on the dev test sets. Check without any tst recording or label:
`contrastive_pipeline --holdout 102` hides 102 training videos per subject and ranks
their trials among them; same hidden videos for every target (split set by --seed).

```bash
python -m src.extract_pooled --convert work/rsa_features/alexnet_fc6.npz --name alexnet-fc6 --out-dir work
python -m src.extract_pooled --convert work/rsa_features/videomae_l8.npz --name videomae-l8 --out-dir work
for t in clip-vitb32 alexnet-fc6 videomae-l8; do for s in 01 02 03 04 05 06; do for k in 0 1 2; do
  python -m src.contrastive_pipeline --data-dir $D --sub $s --emb-train work/emb_train__$t.npz       --emb-test work/emb_test__$t.npz --holdout 102 --seed $k; done; done; done
python -m src.holdout_compare --targets clip-vitb32 alexnet-fc6 videomae-l8
```

6 subjects (incl. sub-05/06) x 3 seeds = 1836 hidden trials:

| target | top-1 | top-5 | mean rank |
|---|---|---|---|
| CLIP | 4.7 | 18.8 | 29.4 |
| AlexNet fc6 | 6.2 | 22.5 | 25.2 |
| VideoMAE-l8 | 6.9 | 22.8 | 24.7 |

Paired vs CLIP: fc6 top-1 +94/-67 (p = 0.04), top-5 +244/-177 (p = 0.001), rank
p = 2e-11; VideoMAE-l8 top-1 +104/-63 (p = 0.002), top-5 +252/-180 (p = 0.0006), rank
p = 2e-13. Lower mean rank than CLIP in 6/6 subjects for both, including sub-05/06,
over 3 seeds. So the member choice holds on data the dev scores never saw. Absolute
values are lower than on tst because training trials are single presentations while
test trials behave like averages of ~2 (eeg_pipeline/README.md).

Caveats: single seed for the fusion members; fc6 and VideoMAE-l8 were picked by the RSA
and their solo scores were first seen on the same dev test sets (the LOO protects the
weights, not the choice of members); section 5 checks that choice without test data.
The original members' weights (9:9:5:2) were chosen on all four dev subjects. For sub-05/06: `make_submission.py` + the two new members (dumps
`work/prep_variants/A_fc6_XX_s0.npy`, `A_vmae8_XX_s0.npy`, original tst order) and the
E-D dumps from checkpoint 5e3154e.

## 6. Daniel's audio member (CLAP ridge): no gain (18 Oct)

Daniel's branch (commit 22c9541) adds an audio vote: a ridge from the EEG to the CLAP
embedding of each video's sound track (`src.xsubj.audio_member`; solo on dev 4.9 / 21.1,
mean rank 25.7). Tested as a 12th member of the candidate fusion (section 4), same LOO
protocol; its weight searched over 0, 1, 2, 3, 5, 9 together with the other new members:

```bash
for s in 01 02 03 04; do git show origin/feature/clip-lowlevel-forward-model:models/xsubj/members/clap_ridge/dev_sorted/clap_${s}_ridge_s0.npy \
    > work/prep_variants/A_dclap_${s}_s0.npy; done
python -m src.loo_clap_member
```

| Fusion (LOO held-out mean, sub-01..04) | top-1 | top-5 | mean rank |
|---|---|---|---|
| candidate (section 4) | 29.41 | 63.24 | 7.87 |
| + Daniel's CLAP member | 30.15 | 62.50 | 7.60 |

Chosen CLAP weight 5 in all four folds. Paired on 408 recordings: top-1 +30 / -27 (exact
McNemar p = 0.79), top-5 +18 / -21 (p = 0.75), rank better in 138, worse in 123 (Wilcoxon
p = 0.55). Not significant on any measure, so the candidate stays without it. The sound
information is not lost: in the RSA, loudness over time is as brain-like as the best
visual features, but the visual members' EEG decoders seem to pick up most of it already.

## 7. Time windows and CLIP's inner layers: no gain (18 Oct)

Two ideas from the RSA, tested with one decoder type so that only the target or the EEG
window changes: Daniel's ridge (`src.ridge_member`: pooled + per-subject ridge, as in his
audio member; with the CLAP target it reproduces his CLAP rankings exactly).

1. **Time hierarchy:** early AlexNet layers match the brain at ~105 ms, deep layers at
   ~225 ms. Does a decoder of an early layer from the early EEG only add to the fusion?
2. **CLIP's inner layers:** in the RSA, CLIP blocks 9 and 12 (r 0.032) are much more
   brain-like than its output (0.026), close to fc6 (0.036). Could they replace AlexNet?

```bash
git show origin/feature/clip-lowlevel-forward-model:models/xsubj/features/clap.npz > work/clap_daniel.npz
python -m eeg_pipeline.rsa_features --stim_dir $D --out_dir work/rsa_features --models clip   # branch feature/eeg-preprocessing
F=work/rsa_features
python -m src.ridge_member --emb work/clap_daniel.npz --tag _rclap        # check: = A_dclap (section 6)
python -m src.ridge_member --emb $F/alexnet_conv2.npz --tmin 0.05 --tmax 0.15 --tag _rconv2e
python -m src.ridge_member --emb $F/alexnet_conv2.npz --tag _rconv2
python -m src.ridge_member --emb $F/alexnet_fc6.npz --tag _rfc6
python -m src.ridge_member --emb $F/alexnet_fc6.npz --tmin 0.15 --tmax 0.35 --tag _rfc6m
python -m src.ridge_member --emb work/clip_vitb32_daniel.npz --tag _rclip
for l in 6 9 12; do python -m src.ridge_member --emb $F/clip_l$l.npz --tag _rclip$l; done
python -m src.loo_extra_member --tags _rconv2e _rconv2 _rclip9 _rfc6
```

Each ridge member alone (sub-01..04 tst):

| target, EEG window | top-1 | top-5 | mean rank |
|---|---|---|---|
| AlexNet fc6, 50-600 ms | **19.4** | **52.2** | **10.7** |
| CLIP block 9, 50-600 ms | 16.9 | 49.5 | 11.4 |
| CLIP output (Daniel's bank), 50-600 ms | 15.9 | 45.6 | 12.5 |
| CLIP block 6 / block 12, 50-600 ms | 14.5 / 11.5 | 45.8 / 40.7 | 11.9 / 13.7 |
| AlexNet conv2, 50-600 ms | 12.3 | 38.5 | 16.3 |
| AlexNet fc6, 150-350 ms only | 10.5 | 31.6 | 17.0 |
| AlexNet conv2, 50-150 ms only | 8.3 | 25.3 | 22.3 |

- **Cutting the window hurts every target** (conv2 12.3 -> 8.3 top-1, fc6 19.4 -> 10.5):
  the ridge already weighs every time point separately, so it uses the time hierarchy
  on its own, and each layer carries information in both waves (in the RSA, fc6 matches
  the brain at 105 ms almost as well as conv2).
- **CLIP block 9 beats CLIP's output** but not fc6; neither difference is significant
  (fc6 vs block 9: top-1 +52/-42, p = 0.35; rank p = 0.10. Block 9 vs output: top-1
  +42/-38, p = 0.74; rank p = 0.09). So no reason to replace AlexNet, nor to retrain the
  encoders toward block 9.

Added to the candidate fusion (LOO as in section 6, extra weight searched over 0-9):

| extra member | top-1 | top-5 | mean rank | paired vs candidate |
|---|---|---|---|---|
| none (candidate) | 29.41 | 63.24 | 7.87 | |
| conv2, 50-150 ms | 27.21 | 63.73 | 7.71 | top-1 +11/-20 (p = 0.15), top-5 p = 0.87, rank p = 0.09 |
| conv2, 50-600 ms | 29.41 | 60.05 | 7.80 | top-5 +12/-25 (p = 0.047, worse) |
| CLIP block 9 | 28.92 | 61.76 | 7.62 | top-1 p = 0.85, top-5 p = 0.33, rank p = 0.19 |
| fc6 ridge | 27.70 | 62.75 | 7.87 | top-1 p = 0.28, top-5 p = 0.85, rank p = 0.84 |

No extra member helps; together with CLAP (section 6) and V-JEPA 2 (RSA), the fusion
seems saturated: new targets reach the same EEG information the members already use.

## 8. Sinkhorn re-ranking: each video appears once per test set (18 Oct)

Branch `experiment/sinkhorn-fusion`. Every subject's tst split has 102 recordings of the
102 candidates, one each (dev: the label-sorted truth is 1001..1102 for every subject;
sub-05/06: 102 recordings too). The fusion ranked every recording on its own, so one
"attractive" video could top many recordings. `src/sinkhorn.py` turns the fused mean ranks
into P = exp(-mean rank / tau) and normalises rows and columns in turn until every
recording and every video sums to 1 (Sinkhorn 1964; Cuturi 2013), then ranks each
recording by its row of P. Label-free; it uses only the shape of the test set (as the
per-candidate normalisation in `src/xsubj/submit.py`). Plug-in: `make_submission
--sinkhorn 3`; without the option the output is bit-identical to before.

```bash
python -m src.loo_sinkhorn          # nested: weights, then tau, chosen on 3 dev subjects
python -m src.make_submission --data-dir $D --ed-dir work/ed_member_5e3154e \
    --extra-tags _fc6 _vmae8 _dfc6 _dvmae8 --extra-weights 5 9 9 9 --sinkhorn 3 --out-dir work/submission_sinkhorn
python -m src.validate_submission work/submission_sinkhorn/sub-05_ranked_candidate_list.npy $D 05
```

| Candidate fusion (LOO held-out, sub-01..04) | top-1 | top-5 | mean rank |
|---|---|---|---|
| plain (section 4) | 29.41 | 63.24 | 7.87 |
| **+ Sinkhorn** | 29.41 | **68.38** | **6.61** |

Paired on 408 recordings: top-1 +41 / -41 (p = 1.0), top-5 +33 / -12 (exact McNemar
p = 0.0025), rank better in 183, worse in 108 (Wilcoxon p = 4e-9). Chosen tau 5, 2, 2, 8
in the four folds. On all four dev subjects with the final weights (optimistic, for the
choice only) tau 2-5 is a plateau (top-1 31.4-32.1, top-5 66.2-68.1, mean rank 6.34-6.41;
plain 30.4 / 63.0 / 7.65), so the submission uses tau = 3. Simpler variants were worse in
a first, non-nested check: z-scoring each candidate's column (top-1 25.5) and a hard
one-to-one assignment (Hungarian, 28.2). Sinkhorn changes the top video for about half of
the sub-05/06 recordings (47% / 52% unchanged).

## 9. Score fusion instead of rank fusion (18 Oct)

Mean-rank fusion keeps only the order of each member's candidates, not how sure the member
is. Every member script now also saves its similarity table next to its ranking
(`<ranking>.scores.npy`, `src/score_dump.py`; same rows, columns 1001..1102;
`src/xsubj/submit.py --save-scores`). Rerunning all members (commands of sections 1, 2, 4,
with `--save-scores` for `src.xsubj.submit`) reproduced all 78 ranking files bit-identically.
Each member's table is z-scored per recording ("row") or also per candidate ("row+col"),
then summed with weights from the same grid; nested LOO as in section 8.

```bash
python -m src.loo_score_fusion
```

| Fusion (LOO held-out, sub-01..04) | top-1 | top-5 | mean rank |
|---|---|---|---|
| ranks | 29.41 | 63.24 | 7.87 |
| ranks + Sinkhorn | 29.41 | 68.38 | 6.61 |
| scores (z row) | 28.92 | 64.22 | 7.48 |
| scores (z row) + Sinkhorn | 31.62 | 67.65 | 6.21 |
| scores (z row+col) | 28.43 | 64.22 | 7.27 |
| scores (z row+col) + Sinkhorn | 30.15 | 66.42 | 6.64 |

Scores alone vs ranks: n.s. (top-1 +20/-22, top-5 +22/-18, rank p = 0.10). Scores +
Sinkhorn vs ranks + Sinkhorn: top-1 +33/-24 (p = 0.29), top-5 +15/-18 (p = 0.73), rank
p = 0.12, not significant, and its tau varied more between folds (0.1-0.5). So the gain
comes from Sinkhorn, not from scores; the submission keeps rank fusion + Sinkhorn
(fewer moving parts: sub-05/06 need no new score dumps).
