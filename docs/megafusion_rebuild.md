# Megafusion: independent rebuild and two improvements (17 Oct, Sofiya)

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

Caveats: single seed everywhere; fc6 and VideoMAE-l8 were picked by the RSA, but their
solo scores were seen on the same dev test sets, so the LOO protects the weights, not
the choice of members. For sub-05/06: `make_submission.py` + the two new members (dumps
`work/prep_variants/A_fc6_XX_s0.npy`, `A_vmae8_XX_s0.npy`, original tst order) and the
E-D dumps from checkpoint 5e3154e.
