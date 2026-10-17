# Text-Anchor-Kette: VLM-Captions als neuer Zielraum (E-T1–E-T4)

## Hypothese

Die bisherigen Zielräume sind Bild-Embedding-Räume (CLIP-image, AlexNet).
Ein über **Text** ankerter Raum — VLM-generierte Beschreibungen der Videos,
encodiert vom *frozen* CLIP-Text-Tower — ist *andersartig* (anderer Kegel,
semantisch-statistische statt pixelige Struktur). Zwei mögliche Payoffs:

1. **Fusion** (eigentlicher Hebel, s. STATUS.md §4: Diversität schlägt
   Solo-Stärke — zero top-1 overlap zwischen Räumen): komplementäre Räume
   fusionsgewinn-stark, kostengünstig, risikofrei testbar (E-T2/E-T3).
2. **Finetuning-Anker** (E-T4, teuer, nur bei positiver E-T2/E-T3-Evidenz):
   CLIP-Image-Tower via LoRA auf die Videos ziehen, mit frozenem Text-Tower
   als Ziel. Anti-Kollaps: Ziel bleibt semantisch statt one-hot.

## Regeln-Status (aus Challenge-PDF geprüft)

- Captions der **1000 trn-Videos**: frei (Training/Finetuning erlaubt).
- Captions der **102 tst-Videos**: **inference-only** — nur frozen encoden
  für Kandidaten-Features (identisch zum Organizer-Baseline-Encoding der tst
  Videos). Keine Gradients, keine PCA/Scaler-Fit-Statistik auf tst-Captions.
- API-Modelle sind erlaubt (public, pre-hackathon, accessible); Captions-JSON
  **ins Repo committen** ( reproduzierbares Artefakt, ~1000 kurze Texte) +
  Modellname/Version + Prompt im Report dokumentieren.

## Experimente (Entscheidungstore, kein Skippen)

| Id | Was | Skript | Gate |
|---|---|---|---|
| E-T1 | Captions generieren (3/V: SCENE/ACTION/MOOD) | `src/caption_videos.py` | — (Artefakt) |
| E-T2 | frozen CLIP-Text → npz → **Ridge-Bench** (sub-01..04, beide Fenster) | `src/encode_captions.py` + `src.bench_embeddings` | top-1 >= ~2%? (alexnet-Level) |
| E-T3 | InfoNCE-Zielraum + tst-Rank-Fusion gegen clip-Raum; **Overlap-Analyse** top-1/top-5 | `src.contrastive_pipeline` + `src.ensemble_rank` | Fusionsgewinn > +1pp UND niedriger Overlap? |
| E-T4 | CLIP-Finetuning (LoRA, Drift-Anker zu Original-Embedding, nur trn) | neu | nur wenn E-T2+E-T3 positiv |

## Befunde (laufend ergänzt)

- (E-T2-Vortest, Dummy-Captions): CLIP-Text-Kegel bestätigt — pairwise cos
  ~0.97–0.99 im Dummy; real vermutlich breiter. → Scaler+PCA vor Ridge sind
  Pflicht (macht bench ohnehin), aber **Raum-Hypothese nicht aus rohem
  Cosine ableiten**.

### E-T1 ersetzt: BOLD Moments Public Captions (2026-10-17)

Statt einer VLM-API: Das Challenge-Datenset ist das **BOLD Moments Dataset**
(OpenNeuro `on008257`), dessen `derivatives/stimuli_metadata/
llm_frame_annotations.json` **fertige Captions für exakt unsere 1102 Videos**
mitliefert (1000 trn + 102 tst, IDs deckungsgleich; GIT git-large-coco,
Wang et al. 2022, 5 Captions/Video vom mittleren Frame). Public + pre-hackathon
→ rulesafe, null API-Kosten. `src/convert_bold_captions.py` überführt das
BOLD-Format in das `caption_videos.py`-Schema.

- **Gate 0, Alignment (`src/check_caption_alignment.py`)**: Unsere CLIP-Image-
  Embeddings retrieven die passende Caption: trn top-1 **56,7%**, top-5 85,8%
  (1000 Kandidaten); tst top-1 **91,2%**, top-5 99,0% → ID-Zuordnung bewiesen.
- **E-T2, Ridge-Bench (512-d CLIP-Text, mean-pooled)**: dev mean **3.19/12.25**
  (sub-01..04: 3.92/3.92/2.94/1.96 top-1) — über Alexnet-Ridge (2.45/9.31),
  nahe CLIP-Ridge (3.43/13.73). Gate ≥2% → **GO für E-T3**.
- **E-T3a, Caption-`ridge` als 10. Member (`src/loo_captions.py`)**: LOO
  wählt `w_cap=0` in 4/4 Folds — der Ridge-Member fällt durch
  (held 22.55 ≈ 9-member-Referenz; Abweichung kommt aus w_joint=3 in Fold 02,
  nicht vom Caption-Member). Erwartbar: Ridge ist der schwächste Member-Typ.
- **Diversität des Raums**: Caption-Ridge top-1-Übereinstimmung mit clip-ridge
  nur 3–9%, rescue ~0–1% — der Raum ist maximal divers zum CLIP-Bildraum,
  nur die Solo-Stärke fehlt. Deshalb E-T3b: **InfoNCE auf dem Caption-Raum**
  (beim Bild-CLIP war InfoNCE 3,2× stronger als Ridge: 11.03 vs 3.43;
- **E-T3b, Caption-`infonce` als 10. Member (`src/loo_captions.py`)**: Solo
  4.90/20.59 dev mean (sub-01: 6.86/23.53, sonst 3.9–4.9%) — InfoNCE nur
  1.5× Ridge (beim Bild-CLIP: 3.2×), die ~8–10%-Extrapolation zog nicht.
  LOO wählt `w_cap=0` in 3/4 Folds; der eine Fold mit w_cap=0.5 verliert auf
  dem gehaltenen Subject (19.61 vs 20.61). Held-out 22.30/48.28 vs 9-member-
  Referenz 23.04/48.28 → **NO-GO. 9-Member-Fusion und Submissions bleiben
  unverändert.**
- **E-T Gesamtfazit (Report/Presentation-Material)**: Der Text-Anker-Raum
  ist nachweisbar valide (Gate 0: img→txt top-1 91.2% tst) und maximal divers
  (3–9% top-1-Übereinstimmung mit allen Bild-Raum-Members), aber die
  EEG→Text-Repräsentation ist zu schwach (4.9% solo vs 11.0% Bild-CLIP):
 vernunftgemäßes „EEG beschreibt *Was* das Subjekt sah" funktioniert
  schlechter als „EEG codet *Wie* das Video aussieht". Diversity schlägt
  Solo-Stärke nur bis zur Solo-Untergrenze ~Ridge-Niveau; darunter hilft
  auch null Overlap nicht mehr. Sauber geloggte Negativkette E-T2→E-T3a→E-T3b
  in `work/scores.csv` (experiment `loo-heldout-caption`).

## Ausführung

```powershell
# E-T1 — smoke test (3 Videos, zeigt Captions):
$env:KIT_API_KEY = "..."          # NICHT committen
$env:OPENAI_BASE_URL = "https://<kit-toolbox-endpoint>/v1"
.venv\Scripts\python.exe -m src.caption_videos --stimuli-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge\stimuli_train --out work\captions_train.json --model <vlm-name> --limit 3

# E-T1 full (resumable, ~4 parallele Calls):
.venv\Scripts\python.exe -m src.caption_videos --stimuli-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge\stimuli_train --out work\captions_train.json --model <vlm-name>
# Kandidaten (inference-only!):
.venv\Scripts\python.exe -m src.caption_videos --stimuli-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge\stimuli_test --out work\captions_test.json --model <vlm-name>

# E-T2 — encoden + Ridge-Bench:
.venv\Scripts\python.exe -m src.encode_captions --captions work\captions_train.json --out work\emb_train__captions.npz
.venv\Scripts\python.exe -m src.encode_captions --captions work\captions_test.json --out work\emb_test__captions.npz
.venv\Scripts\python.exe -m src.bench_embeddings --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge --emb-train work\emb_train__captions.npz --emb-test work\emb_test__captions.npz --window snr

# E-T3 — InfoNCE auf Text-Raum + Fusion (analog clip-Befehle aus STATUS.md §5)
```

## Pitfalls

- Captions-JSON committen, Key NIE (env `KIT_API_KEY`).
- `work/` ist .gitignore'd — Captions-JSON nach Fertigstellung nach
  `docs/captions/` o.ä. verschieben ODER gitignore adden (Team entscheiden).
- tst-Captions nie in Gradients/PCA-Fit (s. Regeln-Status oben).
- Temperature 0.2 im Captions-Call (deterministisch-ish, keine Halluzination-
  Kreativität). Beim Re-Run doppelte Captions vermeiden: JSON ist
  key-dedupliziert (resume überschreibt nicht).
