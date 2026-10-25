#!/usr/bin/env python
"""Capacity-matched prune-vs-random (resumable waves)."""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.lfcbm.ablation import run

if __name__ == "__main__":
    run(yaml.safe_load((HERE / "config.yaml").read_text()), HERE)
