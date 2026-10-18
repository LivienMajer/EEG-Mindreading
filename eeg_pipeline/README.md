# EEG QC + preprocessing

Automatic quality control and preprocessing for the mind-reading hackathon EEG.
One command checks the data, decides the preprocessing from fixed rules, applies it
to all subjects and writes a report explaining every decision.

## Quick start

```bash
pip install -r requirements.txt          # Python 3.11
python -m eeg_pipeline --data_dir data --out_dir prep
```

`data/` must contain the organizers' folders `trn/`, `tst/` and (for sub-01..04)
`tst_stim/`. Runtime is about a minute for all six subjects.

Then, in any notebook or script:

```python
from eeg_pipeline import load_preprocessed

d = load_preprocessed('prep', 1)
d['X_train']      # (n_train, channels, time)  float32
d['train_ids']    # video id of each training trial (sorted)
d['X_test']       # (102, channels, time) in the ORIGINAL file order -> submission rows
d['test_labels']  # true video ids for sub-01..04 (validation only!), None for sub-05/06
d['times']        # seconds from video onset
d['channels']     # indices of the channels kept
d['config']       # the exact settings used
```

## What happens

1. **QC metrics** per subject: integrity (NaN/Inf, shapes, duplicates, clipping),
   channel quality, artefact trials, line noise, existing filtering, trial offsets,
   test-vs-train amplitude. Test labels are never used.
2. **Rules** turn the metrics into **one configuration shared by all subjects**
   (thresholds are constants at the top of `eeg_pipeline/auto.py`):

   | Rule | Applied when | Action |
   |---|---|---|
   | Integrity | NaN/Inf, wrong shape, not 102 test trials | stop with an error |
   | Drop channels | flat or clipped in any subject | drop for all subjects |
   | Drop training trials | max amplitude z > 6 or flat (≤ 5 % per subject) | drop those training trials; test trials never |
   | Notch 50 Hz | 50 Hz peak > 5x neighbours | notch filter |
   | Low-pass 40 Hz | data not already low-passed | low-pass |
   | Per-trial demeaning | trial offsets > 0.3x within-trial signal | subtract each trial's mean |
   | Normalization | test/train amplitude outside 0.8-1.25 | z-score train and test separately, else with training statistics |

   Then: downsample 500 → 100 Hz, keep 0-3 s.
3. **Preprocess and save** `prep/sub-XX.npz`, plus `config.json`,
   `decisions.md` (human-readable), `decisions.json` and `qc_metrics.csv`.

## Optional variants

Some choices cannot be decided by QC; only the model score on the sub-01..04 test
sets can tell whether they help. Create them as separate folders:

```bash
python -m eeg_pipeline --data_dir data --out_dir prep_detrend --set baseline=detrend
python -m eeg_pipeline --data_dir data --out_dir prep_car     --set reference=average
python -m eeg_pipeline --data_dir data --out_dir prep_hp1     --set l_freq=1
python -m eeg_pipeline --data_dir data --out_dir prep_trialz  --set normalize=trial
python -m eeg_pipeline --data_dir data --out_dir prep_1s      --set tmax=1.0
```

Overrides are applied after the rules and listed in `decisions.md`.
Compare variants by the **average over sub-01..04** (top-1, top-5, median rank).

## Full QC report with plots

```bash
python -m eeg_pipeline.qc --data_dir data --output_dir qc_out
```

Per subject: power spectra, channel correlations, trial-variance map, most extreme
trials, train-vs-test comparison. Note: the organizers most likely subtracted the
mean over trials, so the average-response plots (ERP, reliability, latency, GFP) are
not interpretable for this dataset.

## Are the test trials averaged?

```bash
python -m eeg_pipeline.check_test_averaging --data_dir data
```

Compares test/train variance with the correlation between two subjects' responses to
the same video. Correlation ignores scale, so a pure gain change leaves it unchanged,
while averaging k repetitions removes noise and raises it by ~1/sqrt(R_a * R_b)
(R = test/train variance ratio). Uses the sub-01..04 test labels only to pair trials
of the same video.

## Brain vs AI: when and where does the EEG organize videos like which model?

Representational similarity analysis (RSA; Kriegeskorte et al. 2008), applied over time as
in Cichy et al. (2014, 2016). For every pair of videos, how different are
the EEG responses, and how different are the videos for a model? If the two
dissimilarity matrices (RDMs) correlate, brain and model organize the videos the same
way. Done in 50 ms windows (20 ms steps) over 0-800 ms, so it also shows *when*.

**Data:** training trials only, the 695 videos seen by all six subjects. Test recordings
and labels are not used. Preprocessing: baseline 0-50 ms per channel (no visual response
yet; demeaning over the whole trial would leak the later response into early windows),
10 ms bin averages instead of a filter (no smearing to earlier times), subject's mean
response removed. One RDM per subject (correlation distance over 17 channels x 5 samples).

**Models** (`rsa_features.py`; all public, released before the hackathon):

| Feature set | What it captures | Source |
|---|---|---|
| `low_luminance`, `low_contrast`, `low_saturation`, `low_edges`, `low_motion` | simple visual properties over time (16 frames) | computed from the frames |
| `low_loudness` | loudness over time (log RMS, 100 ms bins) | computed from the sound track |
| `alexnet_conv1..5`, `alexnet_fc6/7` | image network, early to deep layers | torchvision AlexNet `IMAGENET1K_V1` |
| `clip_vitb32_daniel` | image-text model (semantic) | OpenCLIP `ViT-B-32` / `openai` (open_clip 3.3.0), mean of 8 frames; Daniel's `models/xsubj/features/clip_vitb32.npz` |
| `clip_l3/l6/l9/l12` | the same CLIP, inside: tokens after transformer blocks 3/6/9/12 of 12, before the final projection | same model, `--models clip`; its output reproduces Daniel's file exactly |
| `videomae_l4/l8/l12` | video model (sees motion), early/middle/top layer | `MCG-NJU/videomae-base-finetuned-kinetics` |
| `vjepa2_l6/l12/l18/l24` | self-supervised video world model (predicts masked video in feature space), layers 6-24 of 24 | `facebook/vjepa2-vitl-fpc64-256` (V-JEPA 2 ViT-L, Meta 2025), 16 frames, mean over tokens |
| `clap_audio` | audio model (sound meaning) | `laion/clap-htsat-unfused` (same weights as `laion_clap` default) |

Statistics: Spearman r between model RDM and each subject's RDM, averaged over subjects;
noise ceiling (Nili et al. 2014; lower bound: each subject vs the mean of the others);
one-sided t-test over subjects, FDR over time windows (Benjamini & Hochberg 1995). Peak
latency differences: jackknife over subjects (Miller et al. 1998). The upper ceiling bound is reported in the CSV but
not interpreted: with single-trial RDMs it is inflated (each subject is partly compared
with itself).

**Reproduce** (CPU is enough; times on an 8-core laptop):

```bash
pip install -r requirements.txt
# 1. CLIP features from Daniel's branch (Git Bash; in PowerShell `>` corrupts binary files)
git show origin/feature/clip-lowlevel-forward-model:models/xsubj/features/clip_vitb32.npz > work/clip_vitb32_daniel.npz
# 2. all other features for the 1102 videos (~45 min, VideoMAE is most of it)
python -m eeg_pipeline.rsa_features --stim_dir data --out_dir work/rsa_features \
    --models lowlevel alexnet videomae audio
#    V-JEPA 2 (~2.5 h, 8.4 s/video; progress is saved every 25 videos, a rerun resumes)
python -m eeg_pipeline.rsa_features --stim_dir data --out_dir work/rsa_features --models vjepa2
#    CLIP inner layers (~8 min); clip_final = Daniel's file, moved out so it is not counted twice
python -m eeg_pipeline.rsa_features --stim_dir data --out_dir work/rsa_features --models clip
mv work/rsa_features/clip_final.npz work/clip_final_check.npz
# 3. channel order check (~30 s)
python -m eeg_pipeline.check_channel_order --data_dir data
# 4. RSA over time + per electrode group (~6 min), robustness without silent videos (~2 min)
python -m eeg_pipeline.rsa --data_dir data --features work/rsa_features work/clip_vitb32_daniel.npz --groups
python -m eeg_pipeline.rsa --data_dir data --features work/rsa_features work/clip_vitb32_daniel.npz --exclude_silent
```

`data` holds `trn/` and `stimuli_train/`, `stimuli_test/`. Outputs in `results/rsa/`:
`rsa_timecourses.png` (figure), `rsa_timecourses.csv` (r per model and window, plus
noise ceiling), `rsa_summary.csv` (peak time, peak r, % of ceiling, significant span),
`rsa_model_overlap.csv` (how much the models agree with each other),
`rsa_hierarchy_test.txt` (early vs deep layers), `rsa_groups.csv`
(per electrode group), and the same with suffix `_with_sound` for the robustness run
(run before V-JEPA 2 and the CLIP layers were added, so without them; adding them leaves
all other rows unchanged).
Verified with Python 3.11.17, numpy 2.4.6, scipy 1.17.1, torch 2.14.1 (CPU),
torchvision 0.29.1, transformers 5.20.1, av 18.1.0, decord 0.6.0, mne 1.13.2.

**Findings** (`results/rsa/`; r = Spearman, % = of the lower noise ceiling at the peak):

- **The brain response has two waves**, ~100 ms and ~200-260 ms (subjects agree most then).
- **Hierarchy in time:** AlexNet conv1-3 match best at 105 ms (r 0.034-0.036, ~100%
  of the ceiling) and fade; conv4-fc7 peak at 225 ms (fc6 r 0.036, 122%) and stay
  ahead to 800 ms. Peaks are stable when any one subject is left out (conv1-3: 105 ms
  in every fold; fc6/fc7: 225-265 ms). Same order as in the network: simple structure
  first, abstract content later. Tested (`results/rsa/rsa_hierarchy_test.txt`, conv1-3
  vs conv5-fc7): early layers win the first wave and deep layers the second in 6/6
  subjects (paired t(5) = 5.60, p = 0.001); deep layers peak 120 ms later (105 vs 225
  ms, jackknife t(5) = 5.69, p = 0.001; leave-one-subject-out 120-140 ms). The same
  early-to-deep order over time as in MEG with images (Cichy et al. 2016).
- **Hierarchy in space:** early layers are strongest over the occipital pole (conv2 r
  0.038 at 85 ms vs 0.018 at the lateral groups); deep layers move to the lateral
  parieto-occipital electrodes at ~205 ms (fc6 0.024-0.027), over lateral
  occipito-temporal cortex, where objects and scenes are processed (Grill-Spector &
  Malach 2004).
- **Simple visual features explain only the first wave:** edges (0.021) and contrast
  (0.019) peak at 105 ms and fade by 300 ms; luminance and motion energy are weak.
  The contrast effect is not significant once the 26 silent videos are removed, so
  only edges count as robust.
- **Video vs image models:** VideoMAE's middle layer (l8, r 0.035 at 225 ms) is as good
  as AlexNet's deep layers; its top layer (l12, tuned to action classes) is the weakest
  model (0.012). CLIP is in between (0.026 at 265 ms). A video model that sees motion is
  not more brain-like than an image network here, and motion energy alone is weak.
- **CLIP is brain-like inside, not at its output:** blocks 9 and 12 reach r 0.032 at 225
  ms (110% of the ceiling), close to fc6, vs 0.026 at 265 ms for the output embedding the
  team decodes toward; block 3 peaks at 105 ms like AlexNet's early layers (0.024), so
  CLIP shows the same early-to-deep order in time. The output is trained to match captions,
  so it discards visual detail the posterior EEG still carries. For decoding, block 9
  beats the output (ridge 16.9 vs 15.9% top-1) but not fc6 (19.4%), and adds nothing to
  the megafusion (`docs/megafusion_rebuild.md` section 7, branch
  `experiment/eegnet-preprocessing`).
- **A video world model is not more brain-like either:** V-JEPA 2 (ViT-L, self-supervised
  on video) is best at layer 18 (r 0.030 at 225 ms, 101% of the ceiling), below AlexNet
  fc6 (0.036) and VideoMAE-l8 (0.035); layers 6/12/24: 0.022 / 0.027 / 0.024. Its RDMs
  overlap most with VideoMAE-l8 (r 0.62 for l18), less with fc6 (0.47). Given the link
  between RSA and decoding (below), a decoder trained toward it is expected to land
  between CLIP and fc6, so it was not added to the megafusion.
- **Sound:** loudness over time matches the brain as well as the best visual features
  (r 0.027 at 125 ms, ~100% of the ceiling; peak at the latency of the auditory N1,
  Näätänen & Picton 1987),
  and is nearly independent of all visual models (RDM overlap r <= 0.07). Robust
  without the silent videos (0.025 at 105 ms, 76%). CLAP (sound meaning) is weaker (0.020
  at 105 ms): the early response follows how loud, more than what kind of sound.
  Loudness is relatively strongest at the lateral electrodes, closest to the auditory
  cortex (1.5-1.6x the early visual layer there vs 0.7x occipital); significant at
  the right-lateral peak, not at the left-lateral one (subjects vary). Electrical
  pick-up from headphones would also be lateral but has no delay; the signal is weak
  before 45 ms and peaks at 105-125 ms, which argues for a cortical response.
- **Brain-likeness predicts decoding.** Livien's per-subject InfoNCE EEGNet trained toward
  each of 9 of these feature sets (only the target changed): the more brain-like the
  embedding in the RSA, the better the decoder (Spearman rho = 0.69, p = 0.038 for top-1;
  0.80, p = 0.017 without the 30-dimensional loudness target, which has too few
  dimensions to tell 102 videos apart). AlexNet fc6 17.2% / 41.7% and VideoMAE-l8
  13.0% / 44.9% beat the team's CLIP target (11.0% / 33.1%); added to the megafusion
  with Daniel's retrained network: LOO 24.02% / 55.15% vs 22.30% / 48.28% (top-5
  p < 0.001, top-1 n.s.). Code and commands: `docs/megafusion_rebuild.md` on branch
  `experiment/eegnet-preprocessing`.
- **Channel order:** the organizers' names follow the file rows (channel correlation
  falls with scalp distance, rho = 0.93; random orders <= 0.45, p = 0.0002).

Limits: 17 posterior electrodes only (no central/temporal sites); latencies of the
earliest effects may be 10-20 ms too early if the organizers' 0.1-40 Hz filter was
zero-phase (Widmann et al. 2015; peaks are not affected); layers of one network are correlated with each
other (overlap 0.44-0.62 between AlexNet and VideoMAE), so "which layer" comparisons
are about the profile over time, not independent contributions.

## Findings on the challenge data (QC, 17 Oct)

- Channels (organizers; order verified with `check_channel_order`): Pz P3 P7 O1 Oz O2
  P4 P8 P1 P5 PO7 PO3 POz PO4 PO8 P6 P2, all parietal/occipital. Organizer
  preprocessing: band-pass 0.1-40 Hz, blink components removed, 1000 -> 500 Hz.

- Data are clean and already preprocessed (low-pass ~45 Hz, no line noise, almost no
  artefact trials, no bad channels).
- The mean over trials was most likely subtracted upstream (train and test averages
  anti-correlate, r = -0.71 to -0.92 in all subjects).
- Trial offsets are as large as the within-trial signal → per-trial demeaning.
- Test trials are 0.60-0.70x the amplitude of training trials in every subject
  → train and test normalized separately.
- **Test trials behave like averages of ~2 repetitions** (`check_test_averaging`):
  same-video correlation between subjects is 2.17x higher in test than in training
  (95% CI 1.85-2.48; averaging predicts 2.00, a pure gain change 1.00). Power drops
  less at low frequencies (delta ~0.5x) than at high ones (beta/gamma ~0.33x), as
  expected when noise is averaged out and the stimulus response is kept. Same in all
  6 subjects; implied k = 1.8-2.4. Strictly, this shows test trials carry about half
  the noise of training trials; another noise-reduction step upstream would look the
  same and cannot be excluded.

## Rules of the challenge this code respects

- Test trials are never dropped or reordered.
- Test labels (sub-01..04) are only loaded for validation, never used in QC or
  preprocessing decisions.
- Normalizing test data with its own statistics is unsupervised (no labels).

## Tests

```bash
python -m pytest eeg_pipeline/tests -q   # or: python eeg_pipeline/tests/test_pipeline.py
```

Synthetic data built like the challenge data, and data with planted faults (dead
channel, line noise, blink trials). Also checks that the test order is preserved
and that the pipeline stops on NaN values.

## References

Methods

- Kriegeskorte N, Mur M, Bandettini P (2008). Representational similarity analysis -
  connecting the branches of systems neuroscience. *Front Syst Neurosci* 2:4.
  doi:10.3389/neuro.06.004.2008
- Nili H, Wingfield C, Walther A, Su L, Marslen-Wilson W, Kriegeskorte N (2014). A
  toolbox for representational similarity analysis. *PLoS Comput Biol* 10(4):e1003553.
  doi:10.1371/journal.pcbi.1003553 (noise ceiling)
- Cichy RM, Pantazis D, Oliva A (2014). Resolving human object recognition in space and
  time. *Nat Neurosci* 17:455-462. doi:10.1038/nn.3635 (time-resolved RSA)
- Cichy RM, Khosla A, Pantazis D, Torralba A, Oliva A (2016). Comparison of deep neural
  networks to spatio-temporal cortical dynamics of human visual object recognition
  reveals hierarchical correspondence. *Sci Rep* 6:27755. doi:10.1038/srep27755
- Miller J, Patterson T, Ulrich R (1998). Jackknife-based method for measuring LRP onset
  latency differences. *Psychophysiology* 35:99-115. doi:10.1111/1469-8986.3510099
- Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate: a practical and
  powerful approach to multiple testing. *J R Stat Soc B* 57:289-300.
- Widmann A, Schröger E, Maess B (2015). Digital filter design for electrophysiological
  data - a practical approach. *J Neurosci Methods* 250:34-46.
  doi:10.1016/j.jneumeth.2014.08.002 (zero-phase filters and latencies)
- He H, Wu D (2020). Transfer learning for brain-computer interfaces: a Euclidean space
  data alignment approach. *IEEE Trans Biomed Eng* 67:399-410.
  doi:10.1109/TBME.2019.2913914 (Euclidean alignment, tested in experiment/eegnet-preprocessing)

Neuroscience background

- Näätänen R, Picton T (1987). The N1 wave of the human electric and magnetic response
  to sound: a review and an analysis of the component structure. *Psychophysiology*
  24:375-425. doi:10.1111/j.1469-8986.1987.tb00311.x
- Grill-Spector K, Malach R (2004). The human visual cortex. *Annu Rev Neurosci*
  27:649-677. doi:10.1146/annurev.neuro.27.070203.144220

Models (exact checkpoints in the RSA table above)

- Krizhevsky A, Sutskever I, Hinton GE (2012). ImageNet classification with deep
  convolutional neural networks. *NeurIPS* 25. (AlexNet)
- Radford A et al. (2021). Learning transferable visual models from natural language
  supervision. *ICML*, PMLR 139:8748-8763. (CLIP; weights via OpenCLIP, Ilharco G et al.
  2021, doi:10.5281/zenodo.5143773)
- Tong Z, Song Y, Wang J, Wang L (2022). VideoMAE: masked autoencoders are data-efficient
  learners for self-supervised video pre-training. *NeurIPS* 35.
- Wu Y, Chen K, Zhang T, Hui Y, Berg-Kirkpatrick T, Dubnov S (2023). Large-scale
  contrastive language-audio pretraining with feature fusion and keyword-to-caption
  augmentation. *ICASSP*. doi:10.1109/ICASSP49357.2023.10095969 (CLAP)

Software

- Gramfort A et al. (2013). MEG and EEG data analysis with MNE-Python. *Front Neurosci*
  7:267. doi:10.3389/fnins.2013.00267 (electrode positions in check_channel_order)
