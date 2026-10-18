"""Video features of single frames at fixed times, for decoding the EEG after 600 ms.

All members decode 50-600 ms after onset toward one embedding per video (the mean over its
frames). The videos keep changing for 3 s, so the EEG at t + 50..600 ms may follow the frame
shown at time t. This writes, per time t, the embedding of the frame closest to t:

  work/frame_features/<model>_t<t>.npz   ids (1001 videos... 1..1102), emb (n, d)

Models: AlexNet fc6 (torchvision IMAGENET1K_V1, as eeg_pipeline.rsa_features) and the CLIP
image embedding (OpenCLIP ViT-B-32 / openai, L2-normalised, as Daniel's clip_vitb32.npz).

    python -m src.frame_features --stim-dir <challenge root>
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch  # before decord (Windows DLL clash)
from decord import VideoReader, cpu

TIMES = (0.0, 0.5, 1.0, 1.5, 2.0, 2.5)


def encoders():
    import open_clip
    from PIL import Image
    from torchvision.models import AlexNet_Weights, alexnet
    w = AlexNet_Weights.IMAGENET1K_V1
    net, prep_a = alexnet(weights=w).eval(), w.transforms()
    clip, _, prep_c = open_clip.create_model_and_transforms("ViT-B-32", "openai")
    clip.eval()

    @torch.no_grad()
    def encode(frames):
        x = prep_a(torch.from_numpy(frames).permute(0, 3, 1, 2))
        x = torch.flatten(net.avgpool(net.features(x)), 1)
        fc6 = net.classifier[2](net.classifier[1](net.classifier[0](x)))
        c = clip.encode_image(torch.stack([prep_c(Image.fromarray(f)) for f in frames]))
        return {"alexnet_fc6": fc6.numpy(), "clip": torch.nn.functional.normalize(c, dim=-1).numpy()}
    return encode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stim-dir", required=True)
    ap.add_argument("--out-dir", default="work/frame_features")
    args = ap.parse_args()
    root = Path(args.stim_dir)
    paths = sorted((root / "stimuli_train").glob("*.mp4")) + sorted((root / "stimuli_test").glob("*.mp4"))
    encode = encoders()
    feats = {}
    for i, p in enumerate(paths):
        vr = VideoReader(str(p), ctx=cpu(0))
        idx = [min(len(vr) - 1, int(round(t * vr.get_avg_fps()))) for t in TIMES]
        for name, e in encode(vr.get_batch(idx).asnumpy()).items():
            for t, v in zip(TIMES, e):
                feats.setdefault((name, t), []).append(v.astype(np.float32))
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(paths)} videos", flush=True)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.array([int(p.stem) for p in paths])
    for (name, t), rows in feats.items():
        np.savez(out / f"{name}_t{t:.1f}.npz", ids=ids, emb=np.stack(rows))
    print(f"saved {len(feats)} files to {out}")


if __name__ == "__main__":
    main()
