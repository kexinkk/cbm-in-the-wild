#!/usr/bin/env python
"""Per-concept shift d_k (+ sev-3 holdout twin)."""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.lfcbm.concept_shift import run

if __name__ == "__main__":
    run(yaml.safe_load((HERE / "config.yaml").read_text()), HERE)
