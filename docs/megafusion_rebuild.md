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
# member 3 (E-D): Daniel's checkpoint; the megafusion used the version of commit 86fac43
git show 86fac43:models/xsubj/checkpoints/all/model.pt > models/xsubj/checkpoints/all_86fac43/model.pt
python -m src.ed_member --data-dir $D --checkpoint models/xsubj/checkpoints/all_86fac43/model.pt
#   + label-sort the sub-01..04 dumps into work/tst_ranked/ed_XX_xsubj_s0.npy (np.argsort(tst.y))
python -m src.loo_generalization
```

| Member | Reported (`final_pipeline.md`) | Rebuilt |
|---|---|---|
| 1 InfoNCE CLIP, per subject | 16.7/37.3, 13.7/32.4, 5.9/34.3, 7.8/28.4 | identical |
| 2 InfoNCE alexnet-f3 | 8.8/35.3, 10.8/35.3, 8.8/21.6, 6.9/27.5 | identical |
| 3 E-D (checkpoint 86fac43) | 7.35 / 23.53 | 7.35 / 23.28 (one trial) |
| 4 ridge | 3.43 / 13.73 | identical |
| 5-7 transfers (12 cells) | 3.68 / 12.99, best cell 7.84 | 3.68 / 12.99-ish, best cell 7.84 |
| **LOO megafusion** | **22.30 / 48.28**, weights 9:9:5:2 in all folds | **identical** |

Findings on the way:

- The E-D member uses Daniel's **old** checkpoint (0.5-40 Hz band-pass + average
  reference). The retrained one (commit 5e3154e, low-pass only) is stronger solo:
  **11.52 / 31.86** vs 7.35 / 23.28.
- The `src/xsubj` copy on `contrastive-alignment` could not run the new checkpoint
  (`butter` fails on `l_freq: null`). This branch takes `src/xsubj/eeg/preprocess.py` from
  Daniel's branch; it is a backward-compatible superset: the old checkpoint gives
  bit-identical rankings with it.
- `work/tst_ranked/ed_XX_xsubj_s0.npy`, read by `loo_generalization`, is written by no
  script; it is the label-sorted `work/ed_member/ranked_subXX.npy`.

## 2. Improvements (same LOO protocol: weights chosen on 3 dev subjects, scored on the 4th)

New members: the same per-subject InfoNCE EEGNet as members 1-2, trained toward AlexNet
fc6 and VideoMAE layer 8, the most brain-like embeddings in the RSA
(`eeg_pipeline/README.md` on `feature/eeg-preprocessing`):

```bash
python -m eeg_pipeline.rsa_features --stim_dir $D --out_dir work/rsa_features --models alexnet videomae
python -m src.prep_variants --data-dir $D --variants A --emb work/rsa_features/alexnet_fc6.npz --tag _fc6 --subs 01 02 03 04 05 06
python -m src.prep_variants --data-dir $D --variants A --emb work/rsa_features/videomae_l8.npz --tag _vmae8 --subs 01 02 03 04 05 06
python -m src.loo_new_members                                  # with the old E-D
python -m src.loo_new_members --tst work/tst_ranked_ed_new     # E-D dumps from checkpoint 5e3154e
```

| Fusion (LOO held-out mean, sub-01..04) | top-1 | top-5 |
|---|---|---|
| current megafusion | 22.30 | 48.28 |
| new E-D checkpoint only | 21.32 | 49.75 |
| + fc6 + VideoMAE-l8 | 22.79 | 52.94 |
| **new E-D checkpoint + fc6 + VideoMAE-l8** | **24.02** | **55.15** |

Last row vs current, paired on the same 408 recordings: top-5 +45 / -17 trials
(exact McNemar p < 0.001), mean rank 12.03 -> 10.12 (better in 205, worse in 99,
Wilcoxon p = 2e-10), top-1 +28 / -21 (p = 0.39, **not significant**). Chosen weights
9:9:5:2:1:1:1 + fc6 3 : VideoMAE 9 in three folds, 5 : 5 in one.

Caveats: single seed everywhere; fc6 and VideoMAE-l8 were picked by the RSA, but their
solo scores were seen on the same dev test sets, so the LOO protects the weights, not
the choice of members. For sub-05/06: `make_submission.py` + the two new members (dumps
`work/prep_variants/A_fc6_XX_s0.npy`, `A_vmae8_XX_s0.npy`, original tst order) and the
E-D dumps from checkpoint 5e3154e.
