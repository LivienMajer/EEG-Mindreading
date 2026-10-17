# Final pipeline: weighted mean-rank megafusion (dev 22.30% / 48.28%)

Exact recipe behind the current numbers (sub-01..04 tst, LOO-validated).
All training uses only trn data; no model ever saw any tst recording during
training. Weights were selected exclusively on dev subjects via
leave-one-subject-out (`src/loo_generalization.py`).

```
                    ┌─ own InfoNCE encoder  (CLIP-ViT-B/32 space)   w=9 ─┐
                    ├─ own InfoNCE encoder  (alexnet-f3 space)     w=9 ─┤
102 tst recordings ─┼─ teammate cross-subj encoder (E-D, frozen)    w=5 ─┼── weighted
 per subject        ├─ own ridge forward model (CLIP space)       w=2 ─┤   mean-rank
                    ├─ transfer: infonce model of dev-subj A       w=1 ─┤   fusion
                    ├─ transfer: infonce model of dev-subj B      w=1 ─┤
                    └─ transfer: infonce model of dev-subj C      w=1 ─┘   → ranked list
                                                                              of 102 videos
```

## Members (the diversity axes are the point)

| # | Member | EEG preprocessing | Target space | Training | solo top-1/5 |
|---|--------|-------------------|--------------|----------|--------------|
| 1 | own InfoNCE (clip) | raw decimated 100 Hz, NO filtering, window 50–350 ms, per-trial channel z-score | CLIP ViT-B/32, PCA-256, L2 | InfoNCE, ~950 trn trials of the SAME subject, 200 epochs, lr 3e-4, tau 0.1, seed 0 | 11.03 / 33.09 |
| 2 | own InfoNCE (alexnet) | idem | alexnet-fc7? (alexnet-f3), PCA-256, L2 | idem | 8.82 / 29.90 |
| 3 | E-D: teammate cross-subject encoder (frozen ckpt) | 0.5–45 Hz bandpass, avg ref, crop 0–1 s, 100 Hz, remove mean response, clip_std 6, Euclidean alignment | their CLIP bank (z-scored) | JOINT InfoNCE+MSE on trn of ALL 6 subjects, ShallowConvNet + subject layer, tau 0.07 | 7.35 / 23.53 |
| 4 | own ridge | same EEG as 1–2 | CLIP, PCA-256 | ridge EEG→emb on own trn | 3.43 / 13.73 |
| 5–7 | transfers | same EEG as 1–2 | CLIP, PCA-256 (identical to 1) | none — member 1's model of OTHER dev subjects applied to this subject's tst | 3.68 / 12.99 (mean over cells) |

Key complementarity sources: two distinct video-embedding spaces (zero top-1
overlap on holdout), two algorithm families (contrastive vs regression vs
cross-subject joint training), two preprocessing philosophies (raw vs
filtered/aligned), and subject diversity via transfer members.

## Fusion rule

Weighted mean rank (code: `fuse_mean_rank` / `ensemble_rank --weights`):
each member's full 102-length ranking is converted to per-video ranks,
averaged with weights (9, 9, 5, 2, 1, 1, 1, 2), re-sorted. Members must be in
the same row convention (label-sorted for dev, original tst.npy order for
sub-05/06). Members 1–8 as before, plus member 9 = **jointft** (E-A: one
joint InfoNCE over all 6 subjects' trn with a learned subject token
[~5,650 trials], then 30-epoch finetune on the own trn; `src/joint_infonce.py`).
Standalone the joint model only reaches solo level (11.27/33.58 — the
simplification "one joint model instead of fusion" is rejected by this
measurement), but as a 9th diversity member it consistently adds signal.

## Weight selection = the LOO protocol (no oracle)

For each held-out dev subject: grid search w_ed {3,5,7} x w_ridge {1,2} x
w_joint {0, 0.5, 1, 2} maximizing fused top-1 on the OTHER 3 dev subjects
only, then score the held one. Every fold independently selected
**9:9:5:2:1:1:1:2** — w_joint=2 was picked over 0 in 4/4 folds (never zero);
held-out mean **23.04 / 48.28** (per subject: 23.5/54.9, 26.5/51.0,
20.6/42.2, 21.6/45.1) vs the 8-member LOO **22.30 / 48.28**. Honest framing:
+0.7 pp top-1 (~3 hits in 408 trials, within plateau noise), but the
consistent fold-selection is the real signal — the same pattern that made
the 8-member weights trustworthy.

## Performance ladder (dev tst, mean over 4 subjects, seed 0)

| system | top-1 | top-5 |
|---|---|---|
| ridge clip (organizer-baseline family) | 3.43 | 13.73 |
| InfoNCE clip solo | 11.03 | 33.09 |
| 2-space fusion 1:1 (clip+alexnet InfoNCE) | 16.18 | 41.18 |
| + ridge + 3 transfers (9:9:1:1:1:1... i.e. 16.43-config) | 16.43 | 44.12 |
| megafusion + E-D oracle weights 9:9:5:1:1:1:1 | 21.57 | 49.02 |
| 8-member: 9:9:5:2:1:1:1, LOO-validated | 22.30 | 48.28 |
| **final: 9:9:5:2:1:1:1:2 (+jointft), LOO-validated** | **23.04** | **48.28** |

Chance: 0.98 / 4.90. Teammate reference ensemble: ~10.5–11.0 / 33–35.

## For sub-05/06 (the actual submission)

Same members, but (a) own InfoNCE/ridge/jointft members train on sub-05/06 trn
(allowed — only their tst is withheld), (b) E-D already includes 05/06 trn,
(c) transfers come from all FOUR dev models → weights 9:9:5:2:1:1:1:1:2
(the new 4th transfer gets the same band weight; consistent with LOO logic).
Output rows in ORIGINAL tst.npy order; `validate_submission.py` before
hand-in.

**CAUTION (pseudo-labels):** `tst_stim/sub-0{5,6}_tst_stim.npy` files that
appeared locally on 2026-10-17 are teammates' placeholders, NOT organizer
labels. All loaders in our pipeline are gated (`with_labels` only for dev
subjects) so they can never be consumed; the jointft 05/06 dumps were
re-dumped in original order from the checkpoint
(`work/redump_joint0506.py`) after the first run accidentally
argsort-permuted them against the fake labels.

## Caveats to state up front

- Single seed (0) everywhere; seed-averaging pending if time allows.
- Weight grid was coarse (w_ed in {1,3,5,7,9}); fine grid would risk
  overfitting 408 dev trials, coarse grid is the conservative choice.
- 05/06 is unvalidatable: honest expectation 17–25% top-1 driven by
  subject similarity (LOO spread across dev subjects is ±3pp).

## Reproduction

```
# members 1,2 (per subject and space)
python -m src.contrastive_pipeline --data-dir <challenge> --sub 01 \
    --emb-train work/emb_train__clip-vitb32.npz --emb-test work/emb_test__clip-vitb32.npz \
    --holdout 0 --epochs 200 --seed 0
# member 4
python -m src.bench_embeddings ...            # ridge + ranking dumps
# members 5–7 (+ checkpoints for 05/06 transfers)
python -m src.cross_subject_probe --data-dir <challenge> \
    --emb-train work/emb_train__clip-vitb32.npz --emb-test work/emb_test__clip-vitb32.npz
# member 3
python -m src.ed_member --data-dir <challenge>
# fusion + LOO validation
python -m src.loo_generalization
# per-subject fusion / submission writing
python -m src.ensemble_rank --rankings <7 member dumps> --weights 9 9 5 2 1 1 1 --subject 01 --experiment megafusion-final
python -m src.validate_submission submissions/sub-05_ranked_candidate_list.npy
```
