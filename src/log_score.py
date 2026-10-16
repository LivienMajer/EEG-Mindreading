"""Append benchmark results to work/scores.csv — one line per run.

Protocol: log every experiment with its exact config + result, so the
findings table in the README can be regenerated from this file.

CSV columns:
  timestamp,branch,experiment,subject,embedding,window,n_components,
  embed_dim,epochs,batch_size,temperature,lr,seed,holdout,top1,top5,note

Usage from a pipeline script:
    from src.log_score import log_score
    log_score(experiment="infonce-holdout", subject="01",
              embedding="alexnet-f3", top1=0.05, top5=0.15, note="tau=0.1")
"""

from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path

FIELDS = ["timestamp", "branch", "experiment", "subject", "embedding",
          "window", "n_components", "embed_dim", "epochs", "batch_size",
          "temperature", "lr", "seed", "holdout", "top1", "top5", "note"]
PATH = Path("work") / "scores.csv"


def log_score(**kw):
    PATH.parent.mkdir(exist_ok=True)
    row = {f: "" for f in FIELDS}
    row["timestamp"] = datetime.now().isoformat(timespec="seconds")
    row.update({k: v for k, v in kw.items() if k in FIELDS})
    new = not PATH.exists()
    with open(PATH, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)
    print(f"score logged -> {PATH}")
