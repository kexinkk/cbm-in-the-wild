#!/usr/bin/env python
"""Stage-wise propagation over all 50 corruption cells (resumable)."""
import sys
from pathlib import Path

import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1] / "src"))

from cbm_in_the_wild.lfcbm.propagation import run_cells

if __name__ == "__main__":
    run_cells(yaml.safe_load((HERE / "config.yaml").read_text()), HERE)
