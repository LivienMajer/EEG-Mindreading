"""Generate per-video captions with an OpenAI-compatible VLM API (E-T1).

Experiment E-T1 of the text-anchor chain (docs/text_anchor_captions.md):
sample N frames per video, send them to a VLM (e.g. the KIT-toolbox or any
OpenAI-compatible endpoint), ask for three complementary captions
(scene/objects, action/motion, style/mood), and store everything in a JSON
file that can be committed (captions are the reproducible artifact; the API
model name + version and this prompt go into the report).

Regulatory note (challenge rules):
  - trn captions (1000 videos): free to use anywhere (training, finetuning).
  - tst captions (102 videos): INFERENCE-ONLY -- they may only be encoded by
    the FROZEN text encoder to build candidate features (same as the
    organizer baseline encodes tst videos). They must never flow into
    gradients, PCA fit statistics, or model selection.

Usage:
  # smoke test (3 videos, prints captions):
  python -m src.caption_videos --stimuli-dir <root>/stimuli_train \
      --out work/captions_train.json --limit 3 --model <vlm-model-name>

  # full run over the 1000 training stimuli (resumable):
  python -m src.caption_videos --stimuli-dir <root>/stimuli_train \
      --out work/captions_train.json --model <vlm-model-name>

  # candidate set (inference-only use downstream!):
  python -m src.caption_videos --stimuli-dir <root>/stimuli_test \
      --out work/captions_test.json --model <vlm-model-name>

Credentials (no key in the repo!):
  - API key:  env KIT_API_KEY  (fallback OPENAI_API_KEY), or --api-key
  - Base URL: env OPENAI_BASE_URL, or --base-url (default: OpenAI)
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Windows DLL order: torch BEFORE decord (see src/extract_embeddings.py).
import torch  # noqa: F401,E402

import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

from .extract_embeddings import extract_frames  # noqa: E402

# ---------------------------------------------------------------------------
# VLM prompt: three captions with distinct foci -> diverse anchors for the
# text space. Line prefixes make parsing robust without json-mode support.
# ---------------------------------------------------------------------------
PROMPT = (
    "You are given {n} frames sampled in temporal order from a 3-second "
    "video (no audio). Describe it in exactly three lines with these "
    "prefixes:\n"
    "SCENE: one plain sentence about the setting, objects and visual "
    "content\n"
    "ACTION: one plain sentence about what happens / moves / changes over "
    "the 3 seconds\n"
    "MOOD: one short sentence about atmosphere, style or color mood\n"
    "Rules: each line starts with its prefix, is a single sentence of at "
    "most 25 words, mentions no frame numbers, and does not mention that "
    "these are frames."
)

PREFIXES = ("SCENE:", "ACTION:", "MOOD:")


def parse_captions(text: str) -> list[str]:
    """Extract the 3 prefixed captions; fall back to raw lines if unprefixed."""
    caps: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        up = line.upper()
        for p in PREFIXES:
            if up.startswith(p):
                caps[p] = line[len(p):].strip(" -\t")
                break
    if caps:
        return [caps.get(p, "") for p in PREFIXES if caps.get(p)]
    # fallback: treat every non-empty line as one caption
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    return lines if lines else [text.strip()]


def frame_to_jpeg_b64(frame: np.ndarray, max_side: int = 512,
                      quality: int = 87) -> str:
    """(H, W, 3) uint8 -> resized base64 JPEG data URI payload."""
    img = Image.fromarray(frame)
    w, h = img.size
    scale = max_side / max(w, h)
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def call_vlm(base_url: str, api_key: str, model: str,
             content: list[dict]) -> str:
    """One chat-completions call against an OpenAI-compatible endpoint.

    content: list of {"type": "text"/"image_url", ...} entries (stdlib only,
    no openai package dependency).
    """
    url = base_url.rstrip("/") + "/chat/completions"
    payload = json.dumps({"model": model, "temperature": 0.2,
                          "max_tokens": 300, "messages": [
                              {"role": "user", "content": content}]}).encode()
    req = urllib.request.Request(
        url, data=payload, method="POST",
        headers={"Content-Type": "application/json",
                 "Authorization": f"Bearer {api_key}"})
    with urllib.request.urlopen(req, timeout=180) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def caption_one(video_path: Path, n_frames: int, base_url: str, api_key: str,
                model: str, max_retries: int = 4) -> dict:
    """Frames -> caption dict for one video. Raises after max_retries."""
    frames = extract_frames(video_path, n_frames)
    if frames.size == 0:
        raise ValueError(f"{video_path.name}: no frames decoded")
    content: list[dict] = [{"type": "text", "text": PROMPT.format(
        n=len(frames))}]
    for f in frames:  # temporal order
        b64 = frame_to_jpeg_b64(f)
        content.append({"type": "image_url", "image_url": {
            "url": f"data:image/jpeg;base64,{b64}"}})

    delay = 5.0
    last_err = None
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            raw = call_vlm(base_url, api_key, model, content)
            caps = parse_captions(raw)
            if not caps:
                raise ValueError(f"unparseable response: {raw[:200]!r}")
            return {"captions": caps, "raw": raw.strip(),
                    "seconds": round(time.time() - t0, 1)}
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                ValueError, KeyError, json.JSONDecodeError) as e:
            last_err = e
            status = getattr(e, "code", None)
            # do not retry client errors other than rate limit
            if status is not None and status != 429 and status < 500:
                raise
            print(f"  [{video_path.name}] attempt {attempt} failed: {e} "
                  f"-- retry in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
            delay = min(delay * 2, 60)
    raise RuntimeError(f"{video_path.name}: giving up after "
                       f"{max_retries} attempts: {last_err}")


def run(stimuli_dir: Path, out_file: Path, model: str, base_url: str,
        api_key: str, n_frames: int, limit: int | None, workers: int) -> None:
    videos = sorted(stimuli_dir.glob("*.mp4"))
    if not videos:
        raise FileNotFoundError(f"no .mp4 under {stimuli_dir}")
    if limit:
        videos = videos[:limit]

    # resumable: skip videos already captioned in a previous run
    done: dict[str, dict] = {}
    if out_file.exists():
        done = json.loads(out_file.read_text(encoding="utf-8"))
        print(f"resuming: {len(done)} videos already captioned")
    todo = [v for v in videos if f"{int(v.stem):05d}" not in done]
    print(f"{len(videos)} videos total, {len(todo)} to caption "
          f"({n_frames} frames each, model={model}, workers={workers})")
    if not todo:
        print("nothing to do.")
        return

    out_file.parent.mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    n_ok, n_fail = 0, 0
    failures: list[str] = []

    def save():
        tmp = out_file.with_suffix(".tmp")
        tmp.write_text(json.dumps(done, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        tmp.replace(out_file)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {pool.submit(caption_one, v, n_frames, base_url, api_key,
                            model): v for v in todo}
        for fut in as_completed(futs):
            v = futs[fut]
            key = f"{int(v.stem):05d}"
            try:
                done[key] = fut.result()
                n_ok += 1
            except Exception as e:  # noqa: BLE001 -- log, keep going
                n_fail += 1
                failures.append(f"{key}: {e}")
                print(f"  FAILED {key}: {e}", file=sys.stderr)
            if (n_ok + n_fail) % 10 == 0:
                save()
                rate = (n_ok + n_fail) / (time.time() - t_start) * 60
                eta = (len(todo) - n_ok - n_fail) / max(rate, 0.01)
                print(f"  {n_ok + n_fail}/{len(todo)} done "
                      f"({rate:.0f}/min, ETA {eta:.0f} min)")
    save()

    print(f"\ncaptions: {n_ok} ok, {n_fail} failed -> {out_file} "
          f"({time.time() - t_start:.0f}s)")
    if failures:
        print("RE-RUN this command to retry failures (resumable). "
              f"{len(failures)} failed:\n  " + "\n  ".join(failures[:20]),
              file=sys.stderr)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--stimuli-dir", required=True,
                   help="stimuli_train or stimuli_test folder")
    p.add_argument("--out", required=True, help="output JSON (resumable)")
    p.add_argument("--model", required=True,
                   help="VLM model name at the endpoint (e.g. gpt-4o-mini)")
    p.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL",
                   "https://api.openai.com/v1"),
                   help="OpenAI-compatible base URL (or env OPENAI_BASE_URL)")
    p.add_argument("--api-key",
                   default=os.environ.get("KIT_API_KEY",
                   os.environ.get("OPENAI_API_KEY")),
                   help="API key (or env KIT_API_KEY / OPENAI_API_KEY)")
    p.add_argument("--n-frames", type=int, default=8,
                   help="frames per video (default 8, same as embeddings)")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--workers", type=int, default=4,
                   help="parallel API calls (default 4)")
    args = p.parse_args()

    if not args.api_key:
        p.error("no API key: set KIT_API_KEY / OPENAI_API_KEY or --api-key")

    run(Path(args.stimuli_dir), Path(args.out), args.model,
        args.base_url.rstrip("/"), args.api_key, args.n_frames, args.limit,
        args.workers)


if __name__ == "__main__":
    main()
