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
3. **Größter Hebel bislang: Rank-Fusion komplementärer Räume** — mean-rank
   Ensemble zweier Räume: top-1 11.8%, top-5 38.2% (sub-01 Holdout).

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

### FAIRER tst-Vergleich (gleiches Protokoll: tst, sub-01–04) — Kernergebnis
| System (Zielraum clip-vitb32) | top-1 | top-5 |
|---|---|---|
| Ridge (forward model) | 3.43% | 13.73% |
| **InfoNCE (EEGNet, 50–350ms)** | **11.03%** | **33.09%** |
| → Verhältnis | **3.2×** | **2.4×** |

Pro Subject InfoNCE-tst: 16.7/37.3 · 13.7/32.4 · 5.9/34.3 · 7.8/28.4.
Zusatz: Holdout-Proxy unterschätzt tst systematisch (sub-01: Holdout ~7% vs.
tst 16.7% top-1) — Holdout-Iteration ist konservativ, echte Zahlen tendieren höher.

### tst-Rank-Fusion: die „vergleichbare Mitglieder"-Lektion
| Fusions-Kandidat (tst, mean sub-01–04) | top-1 | top-5 |
|---|---|---|
| InfoNCE solo (clip) | 11.03% | 33.09% |
| + Ridge 1:1 (equal weight) | 7.35% | 28.43% |
| + Ridge gewichtet 4:1 | 10.29% | 32.35% |
| **+ Ridge gewichtet 9:1** | **12.75%** | 33.09% |
| + Ridge gewichtet 19:1 | 12.01% | 33.58% |
| + 3 Transfers 9:1:1:1 (xsubj-Sonde) | 11.52% | 36.03% |
| **+ Ridge + 3 Transfers 9:1:1:1:1 (GRAND)** | **13.24%** | **36.77%** |
| 2-Raum clip+alexnet InfoNCE 1:1 | 16.18% | 41.18% |
| **MEGAFUSION: 2 Räume (9:9) + Ridge + 3 Transfers (1:1:1:1:1)** | **16.43%** | **44.12%** |

**Lektion 1 (Pitfall):** Rank-Fusion mit einem *deutlich schwächeren* Member im
Equal-Weight zieht den Starken Richtung Rauschen — auf Holdout klappte die Fusion,
weil beide Räume vergleichbar stark waren (6.5 vs 7.5).
**Lektion 2:** Stark herabgewichtete Fusion (9:1–19:1, breites Plateau, kein
Spikes-Overfit) bringt robust +1–2 pp top-1; Feinmuster: Ridge hilft schwachen
Subjects (sub-03: 5.9→9.8, sub-04: 7.8→12.8 bei 9:1) und kostet starke nur wenig
(sub-01: 16.7→14.7 bei 9:1). Bei 102 Trials/Subject sind ±2–3 Treffer Rauschen —
die Gewichte wurden per Dev-Subject-Auswahl gewählt und sind damit legitime
Modell-Selection, aber mit Konfidenzintervall behaupten.

### E-C Cross-Subject-Sonde (sub-01–04, clip, seed 0) — GO für Cross-Subject
| Zelle-Typ | top-1 | top-5 |
|---|---|---|
| Diagonale (Modell Y → tst Y; sanity) | 11.03% | 33.09% |
| **Off-Diagonale (Transfer; 12 Zellen)** | **3.68%** | **12.99%** |
| → Vielfaches über Chance (0.98/4.9) | **3.7×** | **2.6×** |
| Referenz: Per-Subject-Ridge tst | 3.43% | 13.73% |

Interpretation: Transfer ≈ Ridge-Niveau — der *generalisierende* Anteil eines
per-subject-trainierten Modells trägt bereits Ridge-Äquivalent über Subjects
hinweg; der Leistungssprung 3.4→11.0 (Ridge→InfoNCE diagonal) kommt aus dem
idiosynkratischen plus dem gemeinsam nutzbaren Anteil. Joint-Training (E-A)
sollte beides gleichzeitig nutzen. Ausnahmezelle: Modell sub-04 → tst sub-03:
7.84 (besser als diag 5.88!) → Subjekt-Ähnlichkeit ist strukturiert, nicht
uniform — ein Hinweis auf Subjekt-Cluster, nutzbar für sub-05/06-Fusion.
Grand-Fusion (9:1:1:1:1): **13.24% / 36.77%** — neuer tst-Bestwert; alle Subjects
profitieren, die schwächsten am meisten (sub-03 5.88→10.78, +4.9 pp top-1).
Checkpoints: `work/xsubj_probe/model_subXX_*.pt` — Wiederverwendung für
sub-05/06-Transfer-Rankings ohne Retraining.

### 2-Raum-Fusion + MEGAFUSION (tst, seed 0) — neuer Bestwert 16.43%/44.12%
Alexnet-tst-InfoNCE (Kette abgeschlossen): 8.82/35.29, 10.78/35.29, 8.82/21.57,
6.86/27.45 → mean 8.82%/29.90% (clip: 11.03/33.09; vergleichbar stark).
| System (mean sub-01–04) | top-1 | top-5 |
|---|---|---|
| clip oder alexnet solo | 8.8–11.0% | 29.9–33.1% |
| clip+alexnet 1:1 | 16.18% | 41.18% |
| + Ridge + 3 Transfers (9:9:1:1:1:1) | **16.43%** | **44.12%** |

Die Holdout-Evidenz (zero top-1-Overlap zwischen Räumen) überträgt sich 1:1 auf
tst: vergleichbar starke, komplementäre Räume fusionieren besser als alles
zuvor (+5.2pp top-1 / +8.1pp top-5 über dem besseren Solo). Die Transfers
addieren v.a. top-5 (+2.9). Per-Subject Megafusion: 19.6/45.1, 23.5/50.0,
9.8/42.2, 12.8/39.2./sub-03-Top-1 sinkt leicht durch die Transfers (12.8→9.8),
top-5 +5.9.

### Rank-Fusion (sub-01, Holdout, Split 0)
| System | top-1 | top-5 |
|---|---|---|
| beste Einzelmessung (clip, netseed 2) | 7.84% | 33.33% |
| **mean-rank Ensemble alexnet+clip** | **11.76%** | **38.24%** |

Kernbefund der Fusionsanalyse: Die top-1-Treffer der beiden Räume überlappen sich
**NULL** (6 vs. 8 Treffer, 0 gemeinsam); top-5-gelöste Trials: 30 vs. 34, nur 13
gemeinsam, **Union = 51%** aller Trials. Die Räume sind komplementär → Fusion
ist kein Tuning-Trick, sondern strukturell begründet.

## 4. Interpretation & Entscheide

- **Zielraum-These verworfen** ("besseres Embedding rettet alles"): CLIP ≈ alexnet
  innerhalb Seed-Rauschen (±5pp auf top-5). EEG-Seite ist limitierend.
- **Seed-Disziplin**: Single-Seed-Zahlen sind nicht aussagekräftig (die
  exakte "Gleichheit" beider Räume bei netseed 0 war Zufall). Regel: 3 Netz-Seeds
  mitteln; Vergleiche nur gegen Streuband bewerten.
- **Strategiewechsel**: Statt "besten Raum suchen" → **Diversität sammeln**.
  Jeder *andersartige* Raum lohnt für die Fusion (auch wenn er solo schwächer ist),
  solange er andere Trials löst.
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

# Rank-Fusion
.venv\Scripts\python.exe -m src.ensemble_rank --rankings work\holdout\ranked_sub01_emb_train__alexnet-f3_s0_n*.npy work\holdout\ranked_sub01_emb_train__clip-vitb32_s0_n*.npy --truth work\holdout\y_sub01_s0.npy
```

## 6. Offene Fronten (Priorität)

1. **Fusionspool vergrößern**: 3 Netz-Seeds × beide Räume als 6er-Pool (läuft/nächste) +
   Ridge-Rankings mitfusionieren (Ridge hat anderen Bias → vermutlich komplementär).
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
  Automatischer Check: `python -m src.validate_submission <file> <data-dir> <sub>` — prüft
  Format UND die Zeilenreihenfolge-Falle (bei Dev-Subjects kollabiert eine argsortierte
  Datei auf Chance und wird FAILED; verifiziert mit sub-01 in beiden Varianten).
- Teammate-CLIP via git: NIE `git show branch:file > out` (PowerShell zerstört
  Binärdaten); `git restore --source=<branch> -- <pfade>`.
- Frame-Embeddings (alexnet-f3, 4.5 GB): nur via `load_pooled()` laden.
- PCA bei 140k Features: nur `HighDimPCA` (sklearn baut p×p → 156 GB).
- torch: `norm(..., keepdim=)` (nicht keepdims).
- torch VOR decord importieren (Windows-DLL-Konflikt).
- 48h-Disziplin: kein Experiment ohne scores.csv-Eintrag, kein Befund ohne 3 Seeds.
