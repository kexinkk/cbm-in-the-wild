"""ImageNet WordNet-ID → human-readable label resolution.

Source mapping fetched from
``https://raw.githubusercontent.com/raghakot/keras-vis/master/resources/imagenet_class_index.json``
on 2026-05-05 and persisted to ``data/imagenet_wnid_to_name.json`` (gitignored).

Use ``wnid_to_name(wnid)`` for single lookups; ``resolve_wnids(wnids)`` to
batch-resolve a list (returns the wnid itself for any unknown entry —
graceful fallback so figure titles never crash).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

from cbm_in_the_wild.paths import DATA_DIR

_WNID_JSON = DATA_DIR / "imagenet_wnid_to_name.json"


@lru_cache(maxsize=1)
def _table() -> dict[str, str]:
    if not _WNID_JSON.exists():
        return {}
    return json.loads(_WNID_JSON.read_text())


def wnid_to_name(wnid: str) -> str:
    """Return ``'tabby'`` for ``'n02123045'``; falls back to the wnid string itself."""
    return _table().get(wnid, wnid)


def resolve_wnids(wnids: list[str]) -> list[str]:
    """Batch resolve. Unknown wnids pass through unchanged."""
    table = _table()
    return [table.get(w, w) for w in wnids]
