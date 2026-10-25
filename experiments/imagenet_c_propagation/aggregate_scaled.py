"""Merge per-cell scaled_stage_metrics.jsonl into experiment-root JSONL."""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent

rows = []
for cell_dir in sorted(HERE.iterdir()):
    f = cell_dir / "scaled_stage_metrics.jsonl"
    if f.exists():
        for line in open(f):
            rows.append(json.loads(line))

out = HERE / "scaled_stage_metrics.jsonl"
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r) + "\n")
print(f"Merged {len(rows)} rows -> {out}")
