"""Blend CLIP image + caption-text embeddings (E-T4-lin: LoRA-free probe).

E-T4 idea (docs/text_anchor_captions.md): finetune the CLIP image tower
toward the caption cone (LoRA + drift anchor) to get a space that stays
visually decodable but decorrelates from stock CLIP. This module is the
zero-training, first-order version of that pull: because both embeddings
come from the SAME ViT-B/32 openai CLIP (image vs. text tower), the blend

    v = normalize( normalize(img) + alpha * normalize(txt) )

lives in the native joint space and alpha IS the drift dial
(alpha=0: stock image space, alpha=inf: text cone). If EEG->blend decodes
near clip level (11%) while decorrelating from stock-clip rankings, the
full E-T4 LoRA run has a confirmed sweet spot to aim at.

Rule-safe: frozen public encoders only; trn material used freely;
tst videos/captions inference-only (they only pass through the frozen
blend, exactly like the organizer baseline encoding).

Usage (per split, e.g. alpha 0.35):
  python -m src.make_caption_blend --img work\emb_train__clip-vitb32.npz \
      --txt work\emb_train__captions.npz \
      --out work\emb_train__capblend035.npz --alpha 0.35
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def blend(img_path: Path, txt_path: Path, out_path: Path,
          alpha: float) -> None:
    zi = np.load(img_path)
    zt = np.load(txt_path)
    keys = sorted(zi.files)
    missing = [k for k in keys if k not in zt.files]
    if missing:
        raise KeyError(f"{len(missing)} keys missing in text npz, "
                       f"e.g. {missing[:5]}")
    out: dict[str, np.ndarray] = {}
    for k in keys:
        a = zi[k].astype(np.float64)
        b = zt[k].astype(np.float64)
        a /= np.linalg.norm(a)
        b /= np.linalg.norm(b)
        v = a + alpha * b
        v /= np.linalg.norm(v)
        out[k] = v.astype(np.float32)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    print(f"alpha={alpha}: {len(out)} videos -> {out_path}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--img", required=True, help="CLIP image npz")
    p.add_argument("--txt", required=True, help="caption text npz")
    p.add_argument("--out", required=True, help="blended npz out")
    p.add_argument("--alpha", type=float, required=True,
                   help="text pull strength (0=stock, 1=equal)")
    a = p.parse_args()
    blend(Path(a.img), Path(a.txt), Path(a.out), a.alpha)


if __name__ == "__main__":
    main()
