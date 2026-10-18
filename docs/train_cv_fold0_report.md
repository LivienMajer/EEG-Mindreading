col_norm: True

## Single votes, Train-CV (top-1 / top-5 / mean rank; 1/3 folds x <= 102 trials per subject, candidates = fold videos)

| vote | sub-01 | sub-02 | sub-03 | sub-04 | sub-05 | sub-06 | mean |
|---|---|---|---|---|---|---|---|
| clip | 8.8 / 28.4 / 21.7 | 12.7 / 31.4 / 20.6 | 5.9 / 24.5 / 27.3 | 3.9 / 21.6 / 25.1 | 6.9 / 21.6 / 25.2 | 4.1 / 13.3 / 31.6 | 7.1 / 23.5 / 25.3 |
| audio | 3.9 / 18.6 / 32.9 | 5.9 / 18.6 / 32.3 | 3.9 / 15.7 / 31.7 | 3.9 / 16.7 / 30.9 | 5.9 / 16.7 / 31.3 | 1.0 / 13.3 / 31.3 | 4.1 / 16.6 / 31.7 |
| alexnet_conv3 | 11.8 / 37.3 / 17.5 | 6.9 / 28.4 / 19.6 | 4.9 / 22.5 / 27.9 | 6.9 / 27.5 / 20.6 | 2.9 / 22.5 / 21.8 | 5.1 / 21.4 / 26.9 | 6.4 / 26.6 / 22.4 |
| videomae_l8 | 11.8 / 38.2 / 19.2 | 12.7 / 30.4 / 20.1 | 3.9 / 27.5 / 25.4 | 7.8 / 32.4 / 20.0 | 9.8 / 26.5 / 22.7 | 1.0 / 21.4 / 25.5 | 7.8 / 29.4 / 22.2 |

## Weight grid: 1715 combos (clip [0.0, 0.25, 0.5, 0.75, 1.0], each extra [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]); selection metric = (top-1 + top-5) / 2, ties -> lower mean rank


### Best weights according to Train-CV (CV top-1 / top-5 / mean rank at that point)

| clip vote | subject | best weights | CV score | CV score at submission default |
|---|---|---|---|---|
| clip | sub-01 | clip=0.25 audio=0.5 alexnet_conv3=1.25 videomae_l8=1.25 | 21.6 / 44.1 / 15.0 | 22.5 / 39.2 / 14.8 |
| clip | sub-02 | clip=0.5 audio=1.25 alexnet_conv3=1.5 videomae_l8=1.25 | 17.6 / 42.2 / 16.3 | 12.7 / 40.2 / 15.6 |
| clip | sub-03 | clip=0.25 audio=0.5 alexnet_conv3=0.25 videomae_l8=1 | 12.7 / 35.3 / 22.1 | 10.8 / 33.3 / 22.3 |
| clip | sub-04 | clip=1 audio=0.75 alexnet_conv3=1.5 videomae_l8=0.75 | 14.7 / 41.2 / 17.1 | 12.7 / 38.2 / 17.5 |
| clip | sub-05 | clip=0.5 audio=0.5 alexnet_conv3=1.5 videomae_l8=0.25 | 14.7 / 34.3 / 18.0 | 12.7 / 33.3 / 18.0 |
| clip | sub-06 | clip=0 audio=1 alexnet_conv3=1.25 videomae_l8=1.25 | 8.2 / 32.7 / 22.7 | 8.2 / 23.5 / 23.7 |
| clip | pooled | clip=0.25 audio=0.5 alexnet_conv3=1.25 videomae_l8=1 | 13.8 / 35.8 / 18.6 | 13.3 / 34.6 / 18.6 |

Top-10 for sub-05 (clip): [clip=0.5 audio=0.5 alexnet_conv3=1.5 videomae_l8=0.25] 24.5; [clip=0 audio=0.75 alexnet_conv3=1.5 videomae_l8=1] 24.5; [clip=0 audio=0.75 alexnet_conv3=1 videomae_l8=0.75] 24.5; [clip=0.5 audio=0.5 alexnet_conv3=1.25 videomae_l8=1] 24.0; [clip=0.75 audio=0.25 alexnet_conv3=1.25 videomae_l8=0.75] 24.0; [clip=0.25 audio=0.25 alexnet_conv3=0.75 videomae_l8=0.25] 24.0; [clip=0.5 audio=0.5 alexnet_conv3=1.5 videomae_l8=0.5] 24.0; [clip=0.25 audio=0.25 alexnet_conv3=1 videomae_l8=0.25] 24.0; [clip=0 audio=0.5 alexnet_conv3=1.25 videomae_l8=0.5] 24.0; [clip=0 audio=1 alexnet_conv3=1.25 videomae_l8=1] 24.0

Top-10 for sub-06 (clip): [clip=0 audio=1 alexnet_conv3=1.25 videomae_l8=1.25] 20.4; [clip=0 audio=0.75 alexnet_conv3=1 videomae_l8=1] 19.9; [clip=0 audio=0.5 alexnet_conv3=0.75 videomae_l8=0.5] 19.9; [clip=0 audio=1 alexnet_conv3=1.5 videomae_l8=1] 19.9; [clip=0 audio=0.75 alexnet_conv3=1.5 videomae_l8=1] 19.9; [clip=0 audio=1 alexnet_conv3=1 videomae_l8=0.25] 19.9; [clip=0 audio=0.5 alexnet_conv3=0.75 videomae_l8=0.75] 19.4; [clip=0 audio=1 alexnet_conv3=1.5 videomae_l8=1.5] 19.4; [clip=0 audio=1 alexnet_conv3=1.5 videomae_l8=1.25] 19.4; [clip=0 audio=1.25 alexnet_conv3=1.25 videomae_l8=1.5] 19.4