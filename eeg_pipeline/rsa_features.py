"""
Video features for the brain-vs-AI comparison (RSA), one file per feature set.

    python -m eeg_pipeline.rsa_features --stim_dir data --out_dir work/rsa_features --models lowlevel alexnet videomae

`stim_dir` holds stimuli_train/ (0001.mp4..1000.mp4) and stimuli_test/ (1001..1102).
Output: <out_dir>/<name>.npz with `ids` (video ids as int) and `emb` (n_videos x d).

Feature sets (all public, released before the hackathon):
  lowlevel   per frame (16 frames over the 3 s): mean luminance, RMS contrast, mean
             saturation, edge energy; motion energy = mean |difference| of consecutive
             frames. Each saved as its own set (time course per video), so the RSA can
             say which simple property the brain tracks.
  alexnet    torchvision AlexNet (IMAGENET1K_V1), ReLU outputs of conv1..conv5 average-
             pooled to 4x4, plus fc6/fc7; mean over 8 frames.
  videomae   MCG-NJU/videomae-base-finetuned-kinetics (VideoMAE, 2022): 16 frames as one
             clip, tokens mean-pooled at transformer layers 4, 8 and 12. The only model
             here that sees motion over time.
  vjepa2     facebook/vjepa2-vitl-fpc64-256 (V-JEPA 2 ViT-L, Meta 2025; a self-supervised video
             "world model"): 16 frames as one clip (256 px, tubelets of 2 frames), encoder
             tokens mean-pooled at layers 6, 12, 18 and 24 (the predictor is not used).
  audio      the 3 s sound track, mono 48 kHz. clap_audio: laion/clap-htsat-unfused (LAION
             CLAP, HTSAT, LAION-Audio-630K; the default checkpoint of laion_clap, as in
             src/extract embeddings/extract_audio_embeddings.py), repeat-padded to 10 s.
             low_loudness: log RMS in 100 ms bins. 34 videos have no audio stream and get
             3 s of silence, so "no sound" is a stimulus property like any other.
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch  # before decord (Windows DLL clash)
import torch.nn.functional as F
from decord import VideoReader, cpu

N_FRAMES_LOW = 16
N_FRAMES_IMG = 8
VIDEOMAE = 'MCG-NJU/videomae-base-finetuned-kinetics'
VIDEOMAE_LAYERS = (4, 8, 12)
VJEPA2 = 'facebook/vjepa2-vitl-fpc64-256'
VJEPA2_LAYERS = (6, 12, 18, 24)
CLAP = 'laion/clap-htsat-unfused'
AUDIO_SR = 48000
AUDIO_S = 3.0


def video_paths(stim_dir, limit=None):
    stim_dir = Path(stim_dir)
    paths = sorted((stim_dir / 'stimuli_train').glob('*.mp4')) + \
        sorted((stim_dir / 'stimuli_test').glob('*.mp4'))
    return paths[:limit] if limit else paths


def read_frames(path, n):
    """n frames evenly spaced over the video, uint8 (n, H, W, 3)."""
    vr = VideoReader(str(path), ctx=cpu(0))
    idx = np.linspace(0, len(vr) - 1, n).astype(int)
    return vr.get_batch(idx).asnumpy()


def read_audio(path):
    """Mono float32 at 48 kHz, exactly 3 s (zeros if the video has no audio stream)."""
    import av
    n = int(AUDIO_SR * AUDIO_S)
    with av.open(str(path), metadata_errors='replace') as c:   # some files have broken metadata
        if not c.streams.audio:
            return np.zeros(n, np.float32)
        rs = av.AudioResampler(format='flt', layout='mono', rate=AUDIO_SR)
        x = np.concatenate([r.to_ndarray().reshape(-1)
                            for f in c.decode(audio=0) for r in rs.resample(f)] or [np.zeros(0)])
    return np.pad(x[:n], (0, max(0, n - len(x)))).astype(np.float32)


# ─────────────────────────────────────────────
# Feature sets
# ─────────────────────────────────────────────

def lowlevel(frames):
    f = frames.astype(np.float32) / 255
    lum = f @ np.array([0.299, 0.587, 0.114], np.float32)            # (n, H, W)
    mx, mn = f.max(-1), f.min(-1)
    sat = np.where(mx > 0, (mx - mn) / np.maximum(mx, 1e-6), 0)
    gy, gx = np.gradient(lum, axis=(1, 2))
    return {
        'low_luminance': lum.mean((1, 2)),
        'low_contrast': lum.std((1, 2)),
        'low_saturation': sat.mean((1, 2)),
        'low_edges': np.sqrt(gx ** 2 + gy ** 2).mean((1, 2)),
        'low_motion': np.abs(np.diff(lum, axis=0)).mean((1, 2)),
    }


def alexnet_encoder():
    from torchvision.models import AlexNet_Weights, alexnet
    weights = AlexNet_Weights.IMAGENET1K_V1
    net = alexnet(weights=weights).eval()
    prep = weights.transforms()
    relu_idx = {'alexnet_conv1': 1, 'alexnet_conv2': 4, 'alexnet_conv3': 7,
                'alexnet_conv4': 9, 'alexnet_conv5': 11}

    @torch.no_grad()
    def encode(frames):
        x = prep(torch.from_numpy(frames).permute(0, 3, 1, 2))
        out = {}
        for i, layer in enumerate(net.features):
            x = layer(x)
            for name, j in relu_idx.items():
                if i == j:
                    out[name] = F.adaptive_avg_pool2d(x, 4).flatten(1).mean(0)
        x = torch.flatten(net.avgpool(x), 1)
        fc = net.classifier
        x = fc[2](fc[1](fc[0](x)))                 # dropout (eval: identity), fc6, relu
        out['alexnet_fc6'] = x.mean(0)
        out['alexnet_fc7'] = fc[5](fc[4](fc[3](x))).mean(0)
        return {k: v.numpy() for k, v in out.items()}
    return encode


def videomae_encoder():
    from transformers import VideoMAEImageProcessor, VideoMAEModel
    proc = VideoMAEImageProcessor.from_pretrained(VIDEOMAE)
    model = VideoMAEModel.from_pretrained(VIDEOMAE).eval()

    @torch.no_grad()
    def encode(frames):
        inputs = proc(list(frames), return_tensors='pt')
        hs = model(**inputs, output_hidden_states=True).hidden_states   # embeddings + 12 layers
        return {f'videomae_l{k}': hs[k][0].mean(0).numpy() for k in VIDEOMAE_LAYERS}
    return encode


def vjepa2_encoder():
    from transformers import AutoModel, AutoVideoProcessor
    proc = AutoVideoProcessor.from_pretrained(VJEPA2)
    model = AutoModel.from_pretrained(VJEPA2).eval()

    @torch.no_grad()
    def encode(frames):
        inputs = proc(torch.from_numpy(frames).permute(0, 3, 1, 2), return_tensors='pt')
        out = model(**inputs, skip_predictor=True, output_hidden_states=True)
        hs = out.hidden_states                                   # embeddings + 24 layers
        return {f'vjepa2_l{k}': hs[k][0].mean(0).numpy() for k in VJEPA2_LAYERS}
    return encode


def audio_encoder():
    from transformers import ClapFeatureExtractor, ClapModel
    fe = ClapFeatureExtractor.from_pretrained(CLAP)
    model = ClapModel.from_pretrained(CLAP).eval()
    hop = int(AUDIO_SR * 0.1)

    @torch.no_grad()
    def encode(x):
        inputs = fe(x, sampling_rate=AUDIO_SR, return_tensors='pt')
        emb = model.get_audio_features(**inputs)
        emb = emb if torch.is_tensor(emb) else emb.pooler_output
        rms = np.sqrt((x.reshape(-1, hop) ** 2).mean(1))
        return {'clap_audio': emb[0].numpy(), 'low_loudness': np.log10(rms + 1e-5)}
    return encode


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    ap.add_argument('--stim_dir', required=True)
    ap.add_argument('--out_dir', default='work/rsa_features')
    ap.add_argument('--models', nargs='+', default=['lowlevel', 'alexnet', 'videomae'],
                    choices=['lowlevel', 'alexnet', 'videomae', 'vjepa2', 'audio'])
    ap.add_argument('--limit', type=int, default=None, help='first N videos only (timing test)')
    args = ap.parse_args(argv)

    torch.set_num_threads(max(1, torch.get_num_threads()))
    paths = video_paths(args.stim_dir, args.limit)
    ids = np.array([int(p.stem) for p in paths])
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for model in args.models:
        encode = {'lowlevel': lowlevel, 'alexnet': alexnet_encoder,
                  'videomae': videomae_encoder, 'vjepa2': vjepa2_encoder,
                  'audio': audio_encoder}[model]
        encode = encode if model == 'lowlevel' else encode()
        n = N_FRAMES_LOW if model in ('lowlevel', 'videomae', 'vjepa2') else N_FRAMES_IMG
        load = read_audio if model == 'audio' else (lambda p: read_frames(p, n))
        feats, t0 = {}, time.time()
        for i, p in enumerate(paths):
            for name, v in encode(load(p)).items():
                feats.setdefault(name, []).append(np.asarray(v, np.float32))
            if (i + 1) % 100 == 0 or i + 1 == len(paths):
                print(f'{model}: {i + 1}/{len(paths)} videos, {time.time() - t0:.0f} s', flush=True)
        for name, rows in feats.items():
            np.savez(out_dir / f'{name}.npz', ids=ids, emb=np.stack(rows))
        print(f'{model}: saved {", ".join(sorted(feats))} '
              f'({(time.time() - t0) / len(paths):.2f} s/video)', flush=True)


if __name__ == '__main__':
    main()
