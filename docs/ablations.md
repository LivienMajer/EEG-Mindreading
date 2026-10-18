# Ablations of the submission (18 Oct)

The submission is described in `docs/report_eeg_encoding.md`, section 9: CLIP image ensemble, AlexNet conv3, VideoMAE l8, CLAP and loudness ridge votes, cosine terms + Sinkhorn, ridge alpha × 0.2, pooled ridge × 0.25.

How to read the tables:
- **Ablations:** each row removes or changes one component and keeps everything else fixed (weights IMG 0.75 / CLAP 0.3 / conv3 1 / l8 1 / loudness 0.5, Sinkhorn τ = 0.3, no re-tuning).
- Every cell is top-1 % / top-5 % / mean rank.
- Changes are against the full submission (ablations) or the previous step (build-up). In brackets: recordings gained / lost on top-1 and top-5, with exact McNemar p-values.

Noise: top-1 has a standard error of about ±2.3 points on dev.

## Ablations (one component removed; dev = sub-01..04; top-1 / top-5 / mean rank)

| Configuration | dev | dev change vs full (top-1 hits gained/lost; top-5) |
|---|---|---|
| **full submission** | 40.9 / 74.8 / 4.85 |
| - Sinkhorn (cosine terms ranked directly) | 25.0 / 57.4 / 7.58 | -15.9 / -17.4 (20/85 p=0.00; 18/89 p=0.00) |
| - Sinkhorn, per-candidate z-scores instead (old scoring) | 33.8 / 68.4 / 5.75 | -7.1 / -6.4 (24/53 p=0.00; 21/47 p=0.00) |
| Sinkhorn on per-candidate z-scores | 35.0 / 74.0 / 5.23 | -5.9 / -0.7 (10/34 p=0.00; 15/18 p=0.73) |
| - ridge alpha x 0.2 (alpha x 1) | 36.8 / 69.6 / 5.71 | -4.2 / -5.1 (19/36 p=0.03; 18/39 p=0.01) |
| - pooled ridge x 0.25 (pooled x 1) | 37.7 / 71.8 / 5.28 | -3.2 / -2.9 (20/33 p=0.10; 20/32 p=0.13) |
| - loudness vote | 36.8 / 72.5 / 5.15 | -4.2 / -2.2 (12/29 p=0.01; 10/19 p=0.14) |
| - CLAP vote | 39.0 / 73.5 / 4.98 | -2.0 / -1.2 (15/23 p=0.26; 8/13 p=0.38) |
| - both audio votes | 32.4 / 70.6 / 5.61 | -8.6 / -4.2 (19/54 p=0.00; 18/35 p=0.03) |
| - AlexNet conv3 vote | 33.8 / 67.9 / 6.01 | -7.1 / -6.9 (14/43 p=0.00; 15/43 p=0.00) |
| - VideoMAE l8 vote | 39.0 / 69.6 / 5.41 | -2.0 / -5.1 (26/34 p=0.37; 14/35 p=0.00) |
| - both mid-level votes | 25.2 / 55.4 / 8.92 | -15.7 / -19.4 (27/91 p=0.00; 15/94 p=0.00) |
| - CLIP image ensemble | 36.5 / 69.6 / 5.83 | -4.4 / -5.1 (16/34 p=0.02; 11/32 p=0.00) |
| - neural encoders (CLIP vote = ridges only) | 37.5 / 73.0 / 5.21 | -3.4 / -1.7 (7/21 p=0.01; 6/13 p=0.17) |

## Single votes (with Sinkhorn)

| Vote alone | dev |
|---|---|
| CLIP image ensemble | 20.8 / 50.7 / 9.96 |
| CLIP ridges only | 19.1 / 49.5 / 11.85 |
| AlexNet conv3 ridge | 25.0 / 54.9 / 9.93 |
| VideoMAE l8 ridge | 25.0 / 52.2 / 9.05 |
| CLAP ridge | 4.2 / 22.8 / 24.67 |
| loudness ridge | 0.7 / 10.3 / 31.88 |

## Build-up from the previous submission (fixed weights, tau 0.3)

| Step | dev | dev change vs previous step |
|---|---|---|
| previous submission (per-candidate z-scores) | 28.7 / 65.9 / 7.20 |
| + cosine terms and Sinkhorn | 32.8 / 66.4 / 6.27 | +4.2 / +0.5 (42/25 p=0.05; 23/21 p=0.88) |
| + ridge alpha x 0.2 | 34.8 / 70.8 / 5.64 | +2.0 / +4.4 (26/18 p=0.29; 28/10 p=0.01) |
| + loudness vote (0.5) | 37.7 / 71.8 / 5.28 | +2.9 / +1.0 (21/9 p=0.04; 14/10 p=0.54) |
| + pooled ridge x 0.25 (= submission) | 40.9 / 74.8 / 4.85 | +3.2 / +2.9 (33/20 p=0.10; 32/20 p=0.13) |

## Reading

- **Cosine terms need Sinkhorn.** Ranked directly, raw cosine terms are much worse than the old per-candidate z-scores, because nothing corrects the "hub" candidates. Together with Sinkhorn they beat the old scoring by 7.1 points of top-1.
- **The audio votes are complementary.** Loudness alone is near chance (0.7 % top-1), yet removing it costs about 4 points; both audio votes together add 7–9 points. Loudness carries information the image votes lack: when and how strongly the sound starts.
- **Smaller contributions:** the CLIP ensemble, its neural encoders and the pooled weight each add 3–4 points on dev. VideoMAE l8 mainly helps top-5.

The numbers were computed with the evaluation scripts used for report section 8, from cached similarities of the submission's votes. They are not part of the pipeline.
