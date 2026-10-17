# Cross-Subject-Information: Reflexion & Experiment-Menü (Status: Ideenpapier, nichts umgesetzt)

## 1. Die Idee in einem Satz

Wir trainieren jedes Modell derzeit nur mit EEG + Video des EINEN Subjects — dabei liegt
für ~95 % der Train-Videos die Reaktion von 4–6 verschiedenen Subjects vor; diese
zusätzlichen Beobachtungen könnten Training und Fusion stärken.

## 2. Gemessene Datenlage (Warum die Idee realistisch ist)

Pro Subject ~950 (850–972) trn-Trials, **ein Trial pro Video**, hoher Video-Overlap
zwischen allen 6 Subjects (sub-01 vs. andere: 82–94 %). Gemessen mit
`python -m src.cross_subject_stats`:

- Union der Train-Videos über alle 6 Subjects: **exakt 1000** (= alle Train-Videos).
- **Im Schnitt 5,65 EEG-Beobachtungen pro Video** (min 3, max 6); **100 % der Videos
  haben mindestens 2 Subject-Beobachtungen.**
- Ein Joint-Training über alle Subjects => **~5.650 Trials pro Epoche statt ~950**
  (6× Daten für den InfoNCE-Batch; gleichzeitig ~1.000 statt ~866 härtere Negatives).
- `load_all_train()` existiert bereits in `src/challenge_data.py` — die Datenpipeline
  für Joint-Training wäre trivial.

**Passung zu unserer Diagnose:** Der Engpass ist laut Befund das EEG (Batch-Top-1
flaut bei ~45 % ab, tst nur 11–17 %). Die einzigen fundamental NEUEN
Informationsquellen, die wir noch nicht nutzen: (a) **andere Subjects** (dieses
Papier), (b) Audio (decidiert: nur cheaper Probe, s. multimodal_audio.md), (c)
reichere Video-Räume (DINOv2/SigLIP — blockiert auf Teammate-GPU). Cross-Subject ist
billiger als (b)+(c) und methodisch am besten belegt.

## 3. Vorläufige Evidenz aus dem eigenen Repo

| Befund | Quelle | Lesart |
|---|---|---|
| Per-Subject-Ridge (unser Bench): 3,4 % top-1 tst | scores.csv | ohne Cross-Subject-Info |
| Cross-Subject-Ridge (Teammate xsubj): 9,8 % | submit.py-Docstring | mit Joint-Training + Subject-Layer → fast 3× |
| Teammate-Ensemble (Ridge+Encoder, beide cross-sub): 11,0/33,1 | submit.py-Docstring | ≈ unser per-Subject-InfoNCE solo (11,0/33,1) |

Interpretation: Joint-Training hilft schwache Modelle massiv (Ridge 3→10), aber es
ersetzt nicht automatisch ein starkes per-Subject-Modell — Subjekt-Heterogenität
ist real (unsere per-Subject-Top-1-Streuung: 5,9–16,7 %). Das deutet auf dem
Literatur-Rezept „gemeinsames Pretraining + Subject-Adaption" (s. §6).

## 4. Experiment-Menü (sortiert nach erwartetem Nutzen / Aufwand)

**E-A — Joint-Pretraining + Subject-Adapter, dann Per-Subject-Finetuning**
(`--subs all`, Subject-Embedding-Layer; Stage 2: kleines LR nur auf Zielsubject).
Erwartung: am höchsten (6× Daten gegen den EEG-Overfit; Adaption fängt Heterogenität).
Aufwand: moderat (Pipeline-Erweiterung ~eine Stunde + Rechenzeit).
Literatur-Anschluss: ATMS/Tuning-then-Fusion — genau dieses Rezept dominiert
THINGS-artige Benchmarks.

**E-B — Cross-Subject-Kontrast als Nebenziel (Subjekt y als Info, nicht nur als Datenmeer)**
Positivpaare: verschiedenes Subject, gleiches Video; Negatives: verschiedene Videos.
Das belohnt Repräsentationen, die intersubjektiv stabil sind = evozierte, antwortliche
Komponente des Signals; idiosynkratisches Rauschen wird unterdrückt. Kann als
Nebenbedingung (λ·L_xsubj) in dasselbe Training wie E-A.
Aufwand: klein (Batch-Konstruktion + Maske). Neuheitswert: hoch (kaum direkt
so implementiert in den THINGS-Papers — dort meist über Hyperalignment/ISC-Statistik).

**E-C — BILLIGSTER TEST, kein Training: Cross-Subject-Ranking-Sonden (heute machbar)**
Wir haben bereits 8 InfoNCE-Modelle (4 Subjects × 2 Räume) + 4 Ridge-Modelle.
Ranke einfach die tst-Trials von Subject X mit dem auf Subject Y trainierten Modell:
- Falls über Chance: Cross-Subject-Signal existiert und diese Rankings sind
  sofort legitime, diversitäts-starke Fusion-Members (die 9:1-Lektion lehrte:
  Fusion braucht vergleichbare Stärke — fremde Subjects könnten die Lücke füllen).
- Falls auf Chance: E-A/E-B funktionieren vermutlich nur via Joint-Training, nicht via
  Transfer ohne Adaption.
Reine Inferenz, Minuten an CPU — idealer erster Go/No-Go vor jedem Umbau.

**E-D — Exp #5 wie geplant: Teammate-Encoder frozen als Feature-Extraktor**
Ihr Cross-Subject-Checkpoint (ShallowConvNet, sub-01–06 trn, τ=0.07) existiert
bereits (models/xsubj/checkpoints/all/model.pt). Verwende ihn (1) als zusätzlicher
Fusion-Member, (2) als EEG-Feature-Layer für unseren Ridge. Kostet fast nichts —
und GEGEN dessen Ridge (9,8 %!) ist unser InfoNCE (11,0 %) erstmals ein
STÄRKEVERGLEICHBARER Fusionspartner — genau die Konstellation, unter der Fusion
laut Holdout- und tst-Evidenz gewinnt.

**E-E — Other-Subject-Antworten als Zieldenoising (EEG-zu-EEG)**
Durchschnitts-EEG-Prototypen pro Video über Subjects = rauschärmere Targets als
Einzelaufnahmen; per-Subject-Trials darauf alignieren. Methodisch reizvoll, aber
wir übernehmen damit ein selbstgebautes Target statt des bewährten CLIP-Raums —
nur angehen, wenn E-A/E-B keine Früchte tragen.

## 5. Fallstricke (VOR Umsetzung lesen)

1. **False Negatives im Joint-Batch:** Mit 6 Subjects im Batch können zwei Trials
   desselben Videos (verschiedene Subjects) als InfoNCE-Negatives Paare bilden —
   der Loss würde semantisch identische Targets auseinanderziehen. Ab dem ersten
   Joint-Batch braucht die Loss-Maske einen same-video-Ausschluss (auch über Subjects
   hinweg!). Das ist der wahrscheinlichste klassische Fehler bei der Umsetzung.
2. **Subject-Token als Leck:** Der Token kann Subjekt-Identität absorbieren und der
   Encoder nach Subject clustern statt nach Video — Gegenmittel: Subject-Dropout
   (Token zufällig nullen) oder adversarialen Subjekt-Klassifikator.
3. **Heterogenität:** Cross-Subject allein (ohne Adaption, E-C-Sonde) kann auf
   Chance landen, obwohl Joint-Training hilft — falsches No-Go-Risiko bei der
   Sonde. Sonde negativ ≠ E-A sinnlos, nur ohne Adaption.
4. **Vergleichbarkeit:** Cross-Subject-Modelle nicht gegen eine per-Subject-Baseline
   auf unterschiedlichen Fenstern vergleichen (Teammate nutzt 50–600 ms, wir
   50–350 ms) — Protokoll-Detail vorher festlegen.
5. **Regelecheck:** trn von sub-05/06 darf verwendet werden (nur ihre tst ist
   tabu). Their tst bleibt unberührt — keine Grauzone.

## 6. Literatur-Anschluss (für die Präsentation wertvoll)

- **ATMS / Tuning-then-Fusion** (THINGS-Challenge-Gewinner-Rezept): Joint-Pretraining
  über Subjects + gezielte Feinabstimmung → überwindet die Domänenlücke zwischen
  Subjects und bringt Zielleistungs-Gewinn.
- **Hyperalignment / ISC-Linie**: derselben Film ansehen erzeugt intersubjektiv
  korrelierte Muster — genau die Information, die E-B belohnt.
- **Prog-EEG / Zero-Shot-Subject-Alignment**: Subject-Embeddings in einen gemeinsamen
  Raum — Option für sub-05/06, falls E-A bessere Encoder liefert.

## 7. Empfehlung (Reihenfolge)

1. **E-C-Sonde sofort** (null Training, klare Go/No-Go-Info über Transfer),
2. **E-D/Fusion mit dem Teammate-Ridge** (nächster konkreter Schritt nach der
   laufenden alexnet-Kette, vermutlich sofort nutzbar),
3. **E-A + E-B** als der eigentliche Ausbildungsschritt, falls der Hackathon-Zeitplan
   es hergibt — E-B ist zugleich unser bester Kandidat für den Neuheitswert in der
   30 %-Qualitäts-Wertung.

Nicht implementiert — dieses Papier ist die Sortierung für die Team-Diskussion.
