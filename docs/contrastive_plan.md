# Contrastive Alignment — Arbeitsplan (Branch `contrastive-alignment`)

Themenbereich: **Shared-Embedding / Contrastive Loss**.
Der Ridge-Forward-Encoder (Organizer) ist der zu schlagende Baseline.
Interne Proxy-Metrik solange kein Embedding der Teammates da ist:
alexnet-f3 (liegt in `work/`).

## Setup

```bash
git fetch && git checkout contrastive-alignment
python -m src.contrastive_pipeline --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge \
    --sub 01 --emb-train work\emb_train__alexnet-f3.npz --emb-test work\emb_test__alexnet-f3.npz
```

CPU-only reicht (EEGNet ~100k Parameter, 968 Trials -- wenige Min./Subject).

## Datenlage (bereits gemessen, s. README findings)

- KEIN Filtering (raw gewinnt), Fenster 50–350 ms @100 Hz (`T_WIN=(5,35)`)
- Pro Kanal z-scoring pro Trial (erlaubt auch auf Test)
- PCA(256) Target-Space, fit nur auf Train-Video-Embeddings (kein Leakage)

## Experiment-Roadmap (Reihenfolge = Aufwand/Nutzen)

| # | Experiment | Warum | Status |
|---|---|---|---|
| 1 | InfoNCE Baseline (symmetrisch, τ=0.1), sub-01 | lauffähiger Anfang, train-top-1 als Sanity | ⬜ |
| 2 | Video-Sub-Set-Validierung: trainiere ohne 102 zufällige Train-Videos, matche gegen diese als Pseudo-102-Kandidaten | echte Generalisierung messbar, dev-label-frei | ⬜ |
| 3 | Temperatur-Sweep τ ∈ {0.05, 0.1, 0.3, 0.7} | schärfste Design-Entscheidung bei InfoNCE | ⬜ |
| 4 | Probs weighted loss / Bootstrapped-hard negatives | verhindert Collapse auf populäre Cluster | ⬜ |
| 5 | Subject-conditioning (subject token / per-subject head) | 6 Subjects, gleiche Videos -> gemeinsamer Raum + Subjekt-Offset | ⬜ |
| 6 | Multi-positive InfoNCE (alle Frames desselben Videos) | sobald CLIP-Embeddings da sind: 8 statt 1 Positiv pro Trial | ⬜ |
| 7 | Ensemble: cos-sim von InfoNCE-Encoder + Ridge-Forward | meist der größte Sprung: zwei unabhängige Fehlerquellen | ⬜ |

## Wichtige Regeln, um nicht zu scheitern

- **NIE** auf tst-Daten trainen; tst_stim von sub-01–04 NUR für finale Metrik
- Ein Hyperparameter pro Run ändern, Ergebnisse ins README-Log
- `work/` ist git-ignored:结果 (ranked npy) dokumentieren via README/scores

## Erfolgskriterium

Schlage die Ridge-Baseline (top-1/top-5 auf Sub-01–04) mit >1െσ-Abstand;
sonst: Ensemble-Rolle für den Contrastive-Teil.
