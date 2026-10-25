"""Reproducibility utilities. Call set_all_seeds() at the top of every experiment."""

import os
import random
import sys
import warnings

import numpy as np
import torch


def _interactive() -> bool:
    """True in a REPL / Jupyter kernel, where re-exec would kill the session."""
    return hasattr(sys, "ps1") or "ipykernel" in sys.modules


def _pin_hashseed(seed: int) -> None:
    """Make PYTHONHASHSEED actually take effect.

    ``PYTHONHASHSEED`` is read only at interpreter startup, so assigning
    ``os.environ["PYTHONHASHSEED"]`` from a running process is a no-op for that
    process's str/bytes hash randomisation — yet ``hash()`` is used for
    reproducible per-image corruption seeds
    (``corruption_eval._build_*_corrupt_fn``: ``seed = hash((image_id, …))``).
    With a random salt those seeds — and hence the sensor-corruption masks —
    differ every run.

    Fix at the source:
    - If a launcher already pinned ``PYTHONHASHSEED`` (e.g. an sbatch
      ``export PYTHONHASHSEED=…``), honour it — no-op.
    - Otherwise re-exec this interpreter with it set to ``seed`` so string
      hashing becomes deterministic. The guard (env var now set) makes the
      re-exec'd process skip this branch, so there is no exec loop.
    - In an interactive session re-exec would destroy the kernel, so warn
      instead and leave hashing salted.

    Call before any work whose result depends on ``hash()``. ``set_all_seeds``
    is documented to run at the top of an experiment script, so the re-exec
    only repeats cheap import/setup, not real compute.
    """
    if os.environ.get("PYTHONHASHSEED") is not None:
        return  # launcher already pinned it — honour that value
    if _interactive():
        warnings.warn(
            "PYTHONHASHSEED is unset and cannot be pinned mid-session; "
            f"hash()-based seeds are non-reproducible. Relaunch with "
            f"PYTHONHASHSEED={seed} (or set it in the environment) for "
            "deterministic string hashing.",
            RuntimeWarning,
            stacklevel=3,
        )
        return
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.execv(sys.executable, [sys.executable, *sys.argv])


def set_all_seeds(seed: int) -> None:
    """Seed every RNG that affects DCBM pipeline output.

    Seeds Python's hash randomisation (PYTHONHASHSEED, via re-exec — see
    ``_pin_hashseed``), ``random``, NumPy, PyTorch CPU, and PyTorch CUDA RNGs,
    plus forces deterministic cuDNN + CUDA algorithms. Call once per
    experiment script before any model construction or data loading.

    Determinism notes:
    - PYTHONHASHSEED must be set before interpreter startup to affect
      ``hash()``; ``_pin_hashseed`` re-execs to achieve that when no launcher
      pinned it. Earlier this module merely assigned the env var (a no-op),
      so ``hash()``-derived corruption seeds were silently non-reproducible.
    - `torch.use_deterministic_algorithms(True, warn_only=True)` flags any
      op without a deterministic CUDA implementation. We start with
      warn_only so an unexpected non-det op in DCBM's internals does not
      crash training. If warnings appear, decide case-by-case whether to
      patch upstream or flip `warn_only=False`.
    - `CUBLAS_WORKSPACE_CONFIG=:16:8` is required for deterministic cuBLAS;
      it MUST be set before the CUDA context is created. Setting it here is
      safe as long as `set_all_seeds` is called before any CUDA op (which
      our discipline already requires).
    - Expect a small wall-clock slowdown from deterministic mode. Acceptable
      trade for 4-decimal reproducibility.
    """
    _pin_hashseed(seed)
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":16:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
