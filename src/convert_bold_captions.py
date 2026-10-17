"""Convert BOLD Moments llm_frame_annotations.json -> caption_videos format (E-T1-alt).

The public BOLD Moments dataset (OpenNeuro on008257, derivatives/
stimuli_metadata/llm_frame_annotations.json) ships five ready-made GIT
(git-large-coco, Wang et al. 2022) middle-frame captions per video --
pre-hackathon, public, no API cost, rule-safe. Keys '0001'..'1102' (4-digit)
map 1:1 onto our stimuli_train (0001..1000) and stimuli_test (1001..1102).

This converts that file into the exact JSON schema src.caption_videos
produces (keys '%05d', value {"captions": [...], "source": ...}), so the
existing E-T chain (src.encode_captions -> bench/pipeline) runs unchanged.

Regulatory note (same as text_anchor_captions.md):
  - trn captions: free to use anywhere.
  - tst captions: inference-only downstream (frozen encoding only, never
    gradients / PCA / scaler fits).

Usage:
  python -m src.convert_bold_captions --in work/llm_frame_annotations.json \
      --train-out work/captions_train.json --test-out work/captions_test.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

KEY_MODEL = "GIT-git-large-coco"


def convert(in_path: Path, train_out: Path, test_out: Path) -> None:
    data = json.loads(in_path.read_text(encoding="utf-8"))
    n_bad = 0
    train: dict[str, dict] = {}
    test: dict[str, dict] = {}
    for k, v in data.items():
        vid = int(k)
        caps = None
        if isinstance(v, dict):
            caps = v.get(KEY_MODEL) or v.get("captions")
        elif isinstance(v, list):
            caps = v
        if not caps or not all(isinstance(c, str) and c.strip() for c in caps):
            n_bad += 1
            continue
        entry = {"captions": [c.strip() for c in caps],
                 "source": f"bold-moments/{KEY_MODEL}/middle-frame"}
        (train if vid <= 1000 else test)[f"{vid:05d}"] = entry
    if n_bad:
        raise ValueError(f"{n_bad} videos had no usable captions")
    assert len(train) == 1000, len(train)
    assert len(test) == 102, len(test)
    train_out.write_text(json.dumps(train, ensure_ascii=False, indent=1),
                         encoding="utf-8")
    test_out.write_text(json.dumps(test, ensure_ascii=False, indent=1),
                        encoding="utf-8")
    print(f"train captions -> {train_out} ({len(train)} videos)")
    print(f"test  captions -> {test_out} ({len(test)} videos)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="in_path", required=True)
    p.add_argument("--train-out", required=True)
    p.add_argument("--test-out", required=True)
    a = p.parse_args()
    convert(Path(a.in_path), Path(a.train_out), Path(a.test_out))


if __name__ == "__main__":
    main()
