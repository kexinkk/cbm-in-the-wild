"""Stage-metrics sink: append-only JSONL writer for per-stage observations.

Use to capture stage-wise observations from a forward pass — the runner
chooses what stage names and payload fields make sense for the current
analysis. Default ``metrics_sink=None`` on consumers (e.g. ``build_cbm``)
keeps callers byte-identical when no logging is wanted.

Schema per row::

    {"stage": "<name>", **payload}

Where ``payload`` is whatever dict the caller passes — image identifiers,
metric values, condition tags, etc. The sink doesn't validate schema;
discipline lives in the runner.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


class MetricsSink:
    """Append-only JSONL writer. Thread-unsafe (one writer per shard).

    Usage::

        with MetricsSink(Path("out/stage_metrics.jsonl")) as sink:
            sink.write("clip_embedding",
                       {"image_id": "img-7", "cosine_to_clean": 0.93})
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # Append mode so re-running a partial shard doesn't clobber prior rows.
        self._fh = open(self.path, "a", encoding="utf-8")

    def write(self, stage: str, payload: dict[str, Any]) -> None:
        row = {"stage": stage, **payload}
        self._fh.write(json.dumps(row, default=_json_default) + "\n")

    def flush(self) -> None:
        self._fh.flush()

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.flush()
            self._fh.close()

    def __enter__(self) -> "MetricsSink":
        return self

    def __exit__(self, *_exc: Any) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass


def _json_default(obj: Any) -> Any:
    """Stringify torch tensors / numpy scalars to JSON-friendly types."""
    if hasattr(obj, "tolist"):
        return obj.tolist()
    if hasattr(obj, "item"):
        return obj.item()
    raise TypeError(f"not JSON-serializable: {type(obj).__name__}")
