"""Encode captions with a FROZEN CLIP text encoder -> video-level npz (E-T2).

Takes the JSON produced by src.caption_videos and encodes all captions per
video with the CLIP text tower (no training, no gradients -- rule-safe even
for tst captions), then saves mean-pooled per-video embeddings as an npz with
keys "{vid:05d}". This is exactly the video-level format that
bench_embeddings.load_pooled() expects, so the caption space plugs directly
into:

  - Ridge reference:   python -m src.bench_embeddings --emb-train
                       work/emb_train__captions.npz --emb-test
                       work/emb_test__captions_npx.npz --window snr
  - InfoNCE target:    python -m src.contrastive_pipeline --emb-train
                       ...captions.npz ... (same npz works)
  - rank fusion:       bench/pipeline dump rankings as usual

Default text tower: ViT-B-32 "openai" (same checkpoint family as our
clip-vitb32 image embeddings -- the caption space is directly comparable to
the corresponding image space, which is the interesting scientific contrast:
EEG -> text-anchored vs. EEG -> image-anchored CLIP).

Usage:
  python -m src.encode_captions --captions work/captions_train.json \
      --out work/emb_train__captions.npz
  python -m src.encode_captions --captions work/captions_test.json \
      --out work/emb_test__captions.npz
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

# torch before anything decord-adjacent (Windows DLL order, STATUS.md pitfall)
import torch  # noqa: E402
import open_clip  # noqa: E402

TEXT_MODELS = {
    # key: (open_clip model, pretrained tag, emb dim)
    "clip-vitb32-text": ("ViT-B-32", "openai", 512),
    "clip-vitl14-text": ("ViT-L-14", "openai", 768),
}


def load_text_encoder(key: str, device: str):
    spec, tag, _ = TEXT_MODELS[key]
    model, _, _ = open_clip.create_model_and_transforms(spec, pretrained=tag)
    model = model.eval().to(device)
    tokenizer = open_clip.get_tokenizer(spec)
    dim = model.token_embedding.embedding_dim * 0  # unused; keep dim below
    return model, tokenizer, {"clip-vitb32-text": 512,
                             "clip-vitl14-text": 768}[key]


def encode_captions(captions_path: Path, out_path: Path, key: str,
                    device: str, batch_size: int = 128) -> None:
    data = json.loads(captions_path.read_text(encoding="utf-8"))
    if not data:
        raise ValueError(f"{captions_path} is empty "
                         "(run src.caption_videos first)")
    model, tokenizer, dim = load_text_encoder(key, device)

    ids = sorted(data)  # zero-padded string keys -> stable order
    per_video: list[list[str]] = []
    for k in ids:
        caps = data[k]["captions"] if isinstance(data[k], dict) else data[k]
        caps = [c.strip() for c in caps if c and c.strip()]
        if not caps:
            raise ValueError(f"video {k}: no usable captions "
                             f"({data[k]!r:.120})")
        per_video.append(caps)

    print(f"{len(ids)} videos, {sum(len(c) for c in per_video)} captions, "
          f"encoder={key} ({dim}-d), device={device}")

    out: dict[str, np.ndarray] = {}
    flat = [c for caps in per_video for c in caps]
    owner = np.repeat(np.arange(len(ids)), [len(c) for c in per_video])
    embs = []
    with torch.no_grad():
        for i in range(0, len(flat), batch_size):
            toks = tokenizer(flat[i:i + batch_size]).to(device)
            e = model.encode_text(toks)
            embs.append(e.float().cpu().numpy())
    E = np.concatenate(embs)  # (n_captions_total, D)

    # mean-pool captions per video (frozen encoder -> rule-safe for tst)
    sums = np.zeros((len(ids), E.shape[1]), dtype=np.float64)
    np.add.at(sums, owner, E)
    counts = np.bincount(owner, minlength=len(ids)).astype(np.float64)
    pooled = (sums / counts[:, None]).astype(np.float32)

    for n, k in enumerate(ids):
        out[k] = pooled[n]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    norms = np.linalg.norm(pooled, axis=1)
    print(f"caption space -> {out_path}  dim={pooled.shape[1]} "
          f"(norm min/mean/max: {norms.min():.1f}/{norms.mean():.1f}/"
          f"{norms.max():.1f})")
    # quick distinctness sanity: if all captions were identical, the space
    # would collapse; std across videos should be clearly > 0
    print(f"cross-video std (should be >> 0): "
          f"{pooled.std(axis=0).mean():.3f}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--captions", required=True, help="captions JSON")
    p.add_argument("--out", required=True, help="output npz")
    p.add_argument("--model", default="clip-vitb32-text",
                   choices=list(TEXT_MODELS))
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()
    encode_captions(Path(args.captions), Path(args.out), args.model,
                    args.device)


if __name__ == "__main__":
    main()
