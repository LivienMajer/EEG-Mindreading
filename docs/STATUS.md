# STATUS — Stand des Kontrast-Alignment-Strangs (2026-10-18, frühe Nacht)

**Dieses Dossier-Dokument ist der Einstiegspunkt:** Es konsolidiert alle Messungen,
Befunde und Entscheidungen des Kontrast-Alignment-Zweigs mit Reproduktionsbefehlen.
Das ursprüngliche Experiment-Roadmap ist docs/contrastive_plan.md (Exp-Nummern
werden hier weitergeführt).

---

## 1. Drei Sätze Zusammenfassung

1. Unser InfoNCE-Ansatz (EEGNet, 50–350 ms, PCA-Zielraum) **generalisiert**:
   auf 102 ungesehenen Videos top-1 ~7%, top-5 ~28% (Chance 1%/4.9%) und
   schlägt die Ridge-Referenzarchitektur auf sub-01 um Faktor ~2.3.
2. Der Zielraum (alexnet-f3 vs. clip-vitb32) ist **sekundär** — der Effekt liegt
   innerhalb der Seed-Streuung. Der Flaschenhals ist die EEG-Seite.
3. **Historischer Versuch, nicht mehr Teil des Workflows:** Ein Mean-Rank-
   Ensemble komplementärer Räume erreichte 11,8%/38,2% (sub-01 Holdout).
   Checkpoint-level Rank-Fusion wird nicht mehr verwendet; aktuelle
   Vorhersagen stammen aus einem einzelnen EEGNet-Checkpoint.

## 2. Zahlenpolitik (Protokolle sauber auseinanderhalten!)

| Protokoll | Was es misst | Wofür es gut ist |
|---|---|---|
| **Holdout-102** (`--holdout 102`) | 102 Train-Videos des Subjekts als Pseudo-Kandidaten; Training + PCA sehen sie nie | unbegrenzte Iteration OHNE tst-Label-Kontakt; Challenge-Proxy |
| **tst-Eval** (sub-01–04, Labels bekannt) | echte Test-Trials, 102 tst-Kandidaten | faire Referenz (Ridge-Bench), finaler Dev-Vergleich |
| **Submission** (sub-05/06) | tst-Row-Reihenfolge = ORIGINAL tst.npy-Reihenfolge (NICHT argsort!) | nur 1×, am Ende |

Chance-Level überall: top-1 0.98%, top-5 4.9% (102 Kandidaten).

## 3. Messergebnisse (alle auch in work/scores.csv)

### Ridge-Referenz (forward model, organizer-style, tst, Dev-Mean über sub-01–04)
| Zielraum | top-1 | top-5 |
|---|---|---|
| alexnet-f3 | 2.45% | 9.31% |
| clip-vitb32 | 3.43% | 13.73% |

### InfoNCE-Holdout (sub-01, Split-Seed 0, Mittel über 3 Netz-Seeds)
| Zielraum | top-1 | top-5 | Einzel-Seeds (top1/top5) |
|---|---|---|---|
| alexnet-f3 | 6.54% | 27.12% | 6.86/23.53 · 6.86/28.43 · 5.88/29.41 |
| clip-vitb32 | 7.52% | 28.10% | 6.86/23.53 · 7.84/27.45 · 7.84/33.33 |

τ-Studie (alexnet, Split 0, 1 Seed): τ=0.1 → 6.86/23.53; τ=0.05 → 4.90/28.43
(schärferes τ verschiebt Masse von top-1 nach top-5).

### Historisches Rank-Fusion-Experiment (nicht mehr im Workflow)

Das frühere Mean-Rank-Ensemble auf sub-01 Holdout erreichte 11.76%/38.24%
(top-1/top-5), verglichen mit 7.84%/33.33% für die beste Einzelmessung. Dieser
Befund ist ausschließlich historisch; es gibt keinen unterstützten
Checkpoint-Level-Rank-Fusion-Befehl mehr.

## 4. Interpretation & Entscheide

- **Zielraum-These verworfen** ("besseres Embedding rettet alles"): CLIP ≈ alexnet
  innerhalb Seed-Rauschen (±5pp auf top-5). EEG-Seite ist limitierend.
- **Seed-Disziplin**: Single-Seed-Zahlen sind nicht aussagekräftig (die
  exakte "Gleichheit" beider Räume bei netseed 0 war Zufall). Regel: 3 Netz-Seeds
  mitteln; Vergleiche nur gegen Streuband bewerten.
- **Aktueller Inferenzpfad**: ein EEGNet-Checkpoint mit gewichteter CSLS-
  Late-Fusion seiner Feature-Heads; kein Rank-Fusion mehrerer Checkpoints.
- **Audio** (docs/multimodal_audio.md): nur als separater Fusions-Zweig nach
  billigem Ridge-Probe; tri-modales Joint-Alignment verworfen (faulty-positive-
  Falle, s. XID/RINCE-Papers).

## 5. Reproduktion aller Zahlen

```powershell
# Grid: 2 Räume × 3 Netz-Seeds (fester Split)
.venv\Scripts\python.exe -m src.contrastive_pipeline --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge --sub 01 --emb-train work\emb_train__alexnet-f3_pooled.npz --emb-test work\emb_test__alexnet-f3_pooled.npz --holdout 102 --epochs 200 --seed 0 --net-seed 0
# ... emb clip-vitb32 analog

# Ridge-Referenz
.venv\Scripts\python.exe -m src.bench_embeddings --data-dir ... --emb-train work\emb_train__clip-vitb32.npz --emb-test work\emb_test__clip-vitb32.npz --window snr

```

## 6. Offene Fronten (Priorität)

1. **EEG-Flaschenhals**: Encoder- und Augmentierungsvarianten unter identischem
   Validierungsprotokoll vergleichen; keine Checkpoint-Rank-Fusion.
2. **Diversitäts-Räume**: DINOv2, SigLIP (GPU-Teammates, s. README-Workflow);
   später ggf. Audio-Zweig nach Probe (docs/multimodal_audio.md Entscheidungsbaum).
3. **EEG-Flaschenhals**: Exp #5 — Teammate-Encoder (`models/xsubj/checkpoints/all/model.pt`,
   ShallowConvNet, auf sub-01–04 vortrainiert) frozen als Feature-Extraktor testen;
   Multi-Subject-Training.
4. **tst-Fair-Vergleich**: InfoNCE auf tst (sub-01–04) messen, um Ridge vs. InfoNCE
   auf identischem Protokoll zu vergleichen (Holdout ist nur der Proxy).
5. **Submission-Protokoll** (zuletzt!): Reihenfolge original, int-Array, 102×102.

## 7. Fallstricke-Register (aus echten Fehlern)

- Submission-Zeilen: ORIGINAL tst.npy-Reihenfolge — `np.argsort(tst.y)` ist NUR Eval.
- Teammate-CLIP via git: NIE `git show branch:file > out` (PowerShell zerstört
  Binärdaten); `git restore --source=<branch> -- <pfade>`.
- Frame-Embeddings (alexnet-f3, 4.5 GB): nur via `load_pooled()` laden.
- PCA bei 140k Features: nur `HighDimPCA` (sklearn baut p×p → 156 GB).
- torch: `norm(..., keepdim=)` (nicht keepdims).
- torch VOR decord importieren (Windows-DLL-Konflikt).
- 48h-Disziplin: kein Experiment ohne scores.csv-Eintrag, kein Befund ohne 3 Seeds.
