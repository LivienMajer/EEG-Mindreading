# Multimodal (Audio) — kritische Bewertung für unsere Challenge

Quellen: Morgado et al., *Robust Audio-Visual Instance Discrimination* (CVPR'21),
Chuang et al., *Robust Contrastive Learning against Noisy Views* (RINCE, CVPR'22).

## TL;DR

1. Die Papers sind fuer uns primaer als **Loss-Rezepturen fuer EEG<->video**
   wertvoll (EEG = extrem "noisy view"), nicht als Argument fuer ein
   tri-modales Alignment.
2. Audio-Zweig **nur per Late Fusion** und NUR wenn ein billiger Ridge-Probe
   auf Dev-Subjects Signale zeigt. Sonst verwerfen (und das Verwerfen als
   Befund praesentieren!).
3. RINCE = Drop-in fuer unser InfoNCE. XID-Gewichtung = unser Original-Beitrag:
   Cross-Subject-d' als Positiv-Gewicht.

## Transfer der Paper-Ideen auf EEG<->video

- RINCE: ersetzt `infonce()` 1:1; q-Sweep {1.0, 0.7, 0.5} = Exp #4.
- XID faulty positives: Frames desselben Videos sind NICHT alle gleich
  gute Positives (Schnitte, Credits) -> Multi-Positive-Loss mit Gewichten.
- XID weight estimation auf uns gemuenzt: Videos, deren EEG cross-subject
  konsistent ist (unsere d'-Metrik!), sind verlaessliche Positives:
  w_i = max(0, d'-Konsistenz). Original, machbar, gut erzaehlbar.

## Audio: Entscheidungsbaum

1. CLAP (oder ImageBind-Audio) Embeddings fuer trn+tst Videos (GPU-Teammate,
   ~Minuten).
2. `bench_embeddings` mit Audio-Embeddings als Target -> top-1/top-5 der
   Audio-Zweigs allein (Dev-Subjects 01-04).
3. top-1 >= 2x Chance -> Late Fusion probieren:
   S_total = (1-w) * S_visual + w * S_audio, w auf Dev gesucht (Grid 0..0.5).
4. top-1 ~Chance -> Audio dokumentiert verwerfen (Failure-Analysis-Punkt
   fuer die Praesentation: 17-Kanal-Visual-Montage, fruehes Zeitfenster).

## Was wir NICHT tun (in 48h)

- Tri-modales joint InfoNCE (EEG+Audio+Video in einem Space): faulty-positive-
  Falle + zu wenig Daten (~900 Trials/Subject) + keine Zeit.
- Audio-seitiges Trainings-Upgrade (SoundNet etc.) bevor Schritt 2 ein
  klares Signal liefert.

## Cargo-Cult-Warnung

CLAP-Spaces sind(audio-lastig. Wenn wir Audio verwenden, dann als
SEPARATEN Zweig mit eigener Sim-Matrix -- nie audio-dominante
Vorberechnet-Spaces (z.B. ImageBind joint) als einzige Target-Raeume,
weil deren Audio-Dims die visuelle Struktur ueberlagem koennten.
