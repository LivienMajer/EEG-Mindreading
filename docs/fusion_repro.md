# 9-Member-Fusion reproduzieren (Anleitung für das Team)

Stand: commit `e70e9f3` (Branch `contrastive-alignment`).
Ergebnis, das reproduziert werden soll (alles Stand >= commit `e70e9f3`,
Branch `contrastive-alignment`):

Die 9 Mitglieder: [1] eigener clip-InfoNCE, [2] eigener alexnet-InfoNCE,
[3] E-D-Teammate-Encoder, [4] eigene Ridge (clip), [5–8] 4 Dev-Transfers,
[9] jointft (E-A: joint InfoNCE über alle 6 Subjects + 30-Epoch-Finetune).

Zielzahlen:

- LOO held-out (dev, sub-01–04): **23.04 % top-1 / 48.28 % top-5**,
  Gewichte **9:9:5:2:1:1:1:2**, jeder Fold wählt w_joint=2 in 4/4 Fällen.
- Submissions: `submissions/sub-{05,06}_ranked_candidate_list.npy`
  (Gewichte 9:9:5:2:1:1:1:1:2, Original-tst.npy-Reihenfolge).

**Wichtig vorab:** `*.npy`, `*.npz`, `*.pt` sind im `.gitignore` — nach einem
frischen Clone existieren KEINE Dumps/Embeddings/Checkpoints. Wer nur
nachvollziehen will, nimmt Variante A (Dump-Paket). Wer alles selbst
durchrechnen will, nimmt Variante B (Full-Recompute, ~3–4 h CPU plus ggf.
GPU-Embedding-Extraktion).

⚠️ **Pseudo-Label-Falle:** `tst_stim/sub-0{5,6}_tst_stim.npy` bei uns lokal
sind Teammate-Platzhalter (17.10. erstellt), KEINE Veranstalter-Labels.
Alle unsere Loader sind dagegen ab commit `e70e9f3` ge-gated
(`with_labels` nur für sub-01–04). Niemals diese Dateien als Wahrheit
benutzen — auch nicht für Modell-Selektion (Regel: Selektion nur auf
sub-01–04 tst).

---

---

## Variante A — Dumps mitbringen, nur fusern (Minuten)

### A1. Einmaliges Setup

```powershell
git clone git@github.com:LivienMajer/EEG-Mindreading.git
cd EEG-Mindreading
git checkout contrastive-alignment   # >= e70e9f3
git pull
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
```

Daten: Challenge-Daten in den Ordner
`provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge`
legen (gleiche Struktur wie auf der Master-Maschine).

Zusätzlich (nicht im Git, von der Master-Maschine kopieren):

1. **Dump-Paket** `work\` — ca. 12 MB, enthält alle Member-Rankings:
   - `work\tst_ranked\*` (alle `infonce_*`, `joint*_`, `ridge_*`, `ed_*`,
     `truth_*` `.npy`)
   - `work\xsubj_probe\` (4 Dev-Checkpoints `.pt` + 24 `ranked_tr*_te*.npy`)
   - `work\ed_member\ranked_sub*.npy`
2. **Video-Embeddings** (für Ridge/Transfer-Berechnung in
   `make_submission`, und für `loo_joint` nicht nötig — das liest nur Dumps):
   - `work\emb_train__clip-vitb32.npz`, `work\emb_test__clip-vitb32.npz`
   - (nur bei Full-Retrain von alexnet-Mitgliedern: `emb_*__alexnet-f3.npz`,
     4.3 GB, oder selbst via Variante B Schritt B2 extrahieren)
3. **E-D-Encoder** (Teammate-Seite):
   - `models\xsubj\checkpoints\all\model.pt`
   - `models\xsubj\features\clip_vitb32.npz`
   (nur nötig, wenn `ed_member.py` neu laufen soll; zum puren Fusionieren
   reichen die `ranked_subXX.npy`-Dumps)

Integrität der Embeddings prüfen (SHA256):

```
236E90EB0F2DA7F5CB7708E6D0B41A831868B7AFD73BA823C4E5DD3D7E2BDCE4  emb_train__clip-vitb32.npz
20C33438055BA7C22639D35DE403712240F4EBD7908BD0467F4587294E819019  emb_test__clip-vitb32.npz
06E576DE695F985E00C67380024FD0EF6F2FDE6BCFB12676620B3A1958CE3346  emb_train__alexnet-f3.npz
CD2A06C3AB30A282FBAFA50631B56CE4798666EFCED8DAEC7B19EACFFE96C174  emb_test__alexnet-f3.npz
```

```powershell
Get-FileHash work\emb_train__clip-vitb32.npz -Algorithm SHA256
```

### A2. LOO ausführen (Sekunden–Minuten, reines NumPy)

```powershell
.venv\Scripts\python.exe -m src.loo_joint
```

Erwarteter Output (bis auf Gleitkomma-Rundung identisch):

```
hold 01: selected w_ed=5 w_ridge=2 w_joint=2 ... held top1 0.2353  top5 0.5490
hold 02: selected w_ed=5 w_ridge=2 w_joint=2 ... held top1 0.2647  top5 0.5098
hold 03: selected w_ed=5 w_ridge=2 w_joint=2 ... held top1 0.2059  top5 0.4216
hold 04: selected w_ed=5 w_ridge=2 w_joint=2 ... held top1 0.2157  top5 0.4510
LOO held-out mean (with joint member): top-1 0.2304  top-5 0.4828
```

### A3. Submissions neu bauen + validieren (~10 min, CPU)

```powershell
.venv\Scripts\python.exe -m src.make_submission --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge
.venv\Scripts\python.exe -m src.validate_submission submissions\sub-05_ranked_candidate_list.npy provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge 05
.venv\Scripts\python.exe -m src.validate_submission submissions\sub-06_ranked_candidate_list.npy provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge 06
```

`make_submission` rechnet Ridge (05/06) und die 4 Transfer-Rankings selbst
aus (braucht Daten + clip-npz + Dev-Checkpoints), alles andere kommt aus
Dumps. Erwartet: `fused 9 members with weights [9, 9, 5, 2, 1, 1, 1, 1, 2]`
und vom Validator `format checks: OK` + `ALL OK`.

Referenz-Vergleich mit der Master-Maschine (sollte bit-identisch sein,
löst man via `fuse_mean_rank` über die 9 Member-Dumps je Subject — s.
`work\check_submission.py` als Vorlage).

---

## Variante B — alles selbst rechnen (~3–4 h CPU, ohne Embedding-Extraktion)

Reihenfolge einhalten — spätere Schritte überschreiben frühere Dumps
(siehe „Fallstricke" unten).

### B1. Repo + Setup wie A1 (Dump-Paket diesmal NICHT nötig,
nur: E-D-Checkpoint + Video-Bank von Teammate-Seite, Daten, `models\`).

### B2. Video-Embeddings (GPU-Maschine, ~1 h)

```powershell
python -m src.extract_embeddings --models clip-vitb32 alexnet-f3 `
    --stimuli-dir <daten>\stimuli_train --out work\emb_train.npz --device cuda
python -m src.extract_embeddings --models clip-vitb32 alexnet-f3 `
    --stimuli-dir <daten>\stimuli_test --out work\emb_test.npz --device cuda
```

(`src/extract_embeddings.py` speichert per-Frame; die Pipeline-Loader
poolen via `load_pooled`. Unix note: auf Linux `python3 -m ...`.)

### B3. Ridge-Referenz + truth-Files (dev, ~2 min)

```powershell
.venv\Scripts\python.exe -m src.bench_embeddings --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge --emb-train work\emb_train__clip-vitb32.npz --emb-test work\emb_test__clip-vitb32.npz --window snr
```

Schreibt `work\tst_ranked\ridge_{01..04}_emb_train__clip-vitb32_s0.npy`
(label-sorted) und `truth_{01..04}_s0.npy`.

### B4. Eigene InfoNCE-Modelle (12 Läufe à ~13 min = ~2.5 h, CPU)

Pro Subject und Raum (hier clip für sub-01; `--sub` und npz analog bis 04,
alexnet-f3 gleicher Satz):

```powershell
.venv\Scripts\python.exe -m src.contrastive_pipeline --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge --sub 01 --emb-train work\emb_train__clip-vitb32.npz --emb-test work\emb_test__clip-vitb32.npz --epochs 200 --seed 0
```

Schreibt `infonce_{01..04}_{emb_tag}_s0.npy` (label-sorted) + truth.
**Ebenso für sub-05 und sub-06 in BEIDEN Räumen** (clip + alexnet): deren
Dumps laufen über den no-labels-Pfad automatisch in Original-Reihenfolge.

### B5. Dev-Checkpoints + Transfer-Rankings (4 Läufe à ~13 min)

```powershell
.venv\Scripts\python.exe -m src.cross_subject_probe --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge --emb-train work\emb_train__clip-vitb32.npz --emb-test work\emb_test__clip-vitb32.npz
```

Schreibt `work\xsubj_probe\model_sub{01..04}_*.pt` (Checkpoints) und die
16 Dev-Zellen `ranked_tr*_te*_*.npy` (label-sorted, Overlap mit B8-Dumps
kein Problem: gleiche Convention). Die 05/06-Transfer-Rankings werden
NICHT hier erzeugt, sondern später durch `make_submission` (Original-Order!).

### B6. E-D-Member (Inferenz-only, ~Minuten)

```powershell
.venv\Scripts\python.exe -m src.ed_member --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge
```

Schreibt `work\ed_member\ranked_sub{01..06}.npy` (Original-Order).
LOO braucht die Dev-Subjecte label-sorted — einmalig:

```powershell
.venv\Scripts\python.exe -c "import numpy as np; from pathlib import Path; from src.challenge_data import load_test_subject
root = Path(r'provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge')
for s in ('01','02','03','04'):
    y = load_test_subject(root, s, with_labels=True).y
    r = np.load(f'work/ed_member/ranked_sub{s}.npy')[np.argsort(y)]
    np.save(f'work/tst_ranked/ed_{s}_xsubj_s0.npy', r)
print('ed dev dumps label-sorted geschrieben')"
```

### B7. Joint-Modell (E-A, ~50 min: 100 Epochen joint + 6×30 Finetune)

```powershell
.venv\Scripts\python.exe -m src.joint_infonce --data-dir provided_material\eeg-mind-reading-challenge\eeg-mind-reading-challenge --emb-train work\emb_train__clip-vitb32.npz --emb-test work\emb_test__clip-vitb32.npz --epochs 100 --finetune-epochs 30 --log-every 10
```

Schreibt Checkpoint + `joint_*`/`jointft_*` Dumps für ALLE 6 Subjecte:
dev label-sorted (LOO-fähig), 05/06 automatisch in Original-Order
(der with_labels-Gate verhindert die Pseudo-Label-Falle).
Erwartete trainings-Metriken: batch top-1 ~0.19 am Ende, dev mean
joint ≈ 10.0/30.9, jointft ≈ 11.3/33.6 (solo-Niveau — das ist der
messbare Beleg, dass die Vereinfachung „ein Joint-Modell statt Fusion"
nicht reicht).

### B8. LOO + Submissions wie A2/A3

`src.loo_joint` (erwartet 23.04/48.28, w_joint=2 in 4/4 Folds), danach
`src.make_submission` (rechnet Ridge 05/06 + 4 Transfers selbst, braucht
also Daten + clip-npz + Dev-Checkpoints aus B5), dann Validierung.

---

## Fallstricke (aus echten Fehlern, alle kosten Stunden gekostet)

1. **Pseudo-Labels 05/06:** nie konsumieren; Loader sind ab `e70e9f3`
   ge-gated. Bei fremdem/eigene Code: `load_test_subject(root, s,
   with_labels=s in ("01","02","03","04"))`.
2. **Zeilenkonventionen:** dev-Dumps label-sorted (nur für LOO/Fusion
   auf dev!), ALLE 05/06-Dumps + Submissions in Original-tst.npy-Order.
   `np.argsort(tst.y)` ausschließlich für die Auswertung, nie vorm Speichern
   von 05/06-Rankings.
3. **Dump-Überschreiber:** `make_submission.ridge_member` und
   `transfer_member` überschreiben `work/tst_ranked/ridge_*` bzw.
   `work/xsubj_probe/ranked_tr*_te*`. Referenz-Dumps VOR dem Neu-Laufen
   wegsichern, danach dev-Dumps wieder label-sorted restaurieren.
4. **Windows:** PowerShell nimmt `;` statt `&&`, `.venv\Scripts\python.exe`
   (64-bit), `torch` VOR `decord` importieren, `git show branch:file > out`
   zerstört Binärdaten (stattdessen `git restore --source=`).
5. **Alexnet-Frame-npz:** 4.3 GB — nur via `load_pooled()` laden; Auswertung
   der 05/06-Rankings gegen Teammate-Pseudo-Labels liefert ~1 % (Chance)
   und ist bedeutungslos.

## Zeit-Budget Überblick

| Schritt | Dauer |
|---|---|
| Variante A (fusern) | < 10 min |
| B2 Embeddings (GPU) | ~1 h (einmal pro Maschine) |
| B4 InfoNCE 12× | ~2.5 h |
| B5 Probe 4× | ~55 min |
| B7 Joint | ~50 min |
| B3/B6 + Fusion | ~15 min |
