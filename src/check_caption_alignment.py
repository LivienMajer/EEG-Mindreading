"""Sanity: are the BOLD Moments captions really OUR videos? (E-T2 gate 0)

Cross-modal retrieval check: for each video, compare its CLIP *image*
embedding (work/emb_*__clip-vitb32.npz, extracted from our mp4s) against
the CLIP *text* embeddings of the BOLD captions (emb_*__captions.npz).
If the annotation ids line up with our video ids, image->text retrieval
must succeed far above chance (5/102 or 50/1000); if they were shuffled
or from a different stimulus set, retrieval collapses to chance.

Prints per-split top-1/top-5/top-10 image->text accuracy. Gate: train
top-5 >= ~15% (CLIP img->txt is noisy even for perfect pairs; random
would be 0.5%/2.5%).

Usage:
  python -m src.check_caption_alignment \
      --img-train work/emb_train__clip-vitb32.npz \
      --img-test work/emb_test__clip-vitb32.npz \
      --txt-train work/emb_train__captions.npz \
      --txt-test work/emb_test__captions.npz
"""

from __future__ import annotations

import argparse

import numpy as np


def load_npz(path: str) -> tuple[np.ndarray, np.ndarray]:
    z = np.load(path)
    ids = np.array(sorted(int(k) for k in z.files))
    E = np.stack([z[f"{i:05d}"] for i in ids]).astype(np.float32)
    return ids, E


def topk_acc(S: np.ndarray, k: int) -> float:
    return float((np.argsort(-S, axis=1)[:, :k] == np.arange(len(S))[:, None]
                  ).any(1).mean())


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--img-train", required=True)
    p.add_argument("--img-test", required=True)
    p.add_argument("--txt-train", required=True)
    p.add_argument("--txt-test", required=True)
    a = p.parse_args()

    for split, ip, tp in (("train", a.img_train, a.txt_train),
                          ("test", a.img_test, a.txt_test)):
        img_ids, Img = load_npz(ip)
        txt_ids, Txt = load_npz(tp)
        assert np.array_equal(img_ids, txt_ids), \
            f"{split}: id mismatch {img_ids[:3]} vs {txt_ids[:3]}"
        Img = Img / np.linalg.norm(Img, axis=1, keepdims=True)
        Txt = Txt / np.linalg.norm(Txt, axis=1, keepdims=True)
        S = Img @ Txt.T
        print(f"{split}: n={len(img_ids)}  "
              f"image->text top1 {topk_acc(S, 1)*100:.1f}%  "
              f"top5 {topk_acc(S, 5)*100:.1f}%  "
              f"top10 {topk_acc(S, 10)*100:.1f}%  "
              f"(chance top5 = {100*5/len(img_ids):.1f}%)")


if __name__ == "__main__":
    main()
