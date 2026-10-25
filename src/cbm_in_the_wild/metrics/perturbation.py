"""Perturbation-stage metrics: Mean Flip Rate (mFR).

mFR scores prediction *consistency* under smooth, video-like perturbation
sequences. It complements accuracy: a model can have high average
accuracy but still flip predictions wildly between consecutive frames
of a tilt / blur / brightness ramp — that's the failure mode mFR
captures (Hendrycks and Dietterich, 2019).

The basic flip rate is defined for prediction sequences (label ↦ label
across consecutive frames). ``stage_wise_flip_rate`` extends this to
arbitrary pipeline-stage tensors via a caller-supplied ``label_fn`` —
e.g. for top-k concept-set flips, pass a function that returns the
frozenset of top-k indices.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Hashable

import torch


def flip_rate(predictions: Sequence[Hashable]) -> float:
    """Fraction of consecutive frame-pairs whose label differs.

    Args:
        predictions: sequence of length ``T``; each entry is a hashable
            label (int class index, frozenset of top-k concepts, etc.).

    Returns:
        Scalar in ``[0, 1]``: ``sum(p_t != p_{t+1}) / (T - 1)``.
        Returns 0.0 if ``T < 2``.
    """
    if len(predictions) < 2:
        return 0.0
    flips = sum(1 for a, b in zip(predictions[:-1], predictions[1:]) if a != b)
    return flips / (len(predictions) - 1)


def mfr_predictions(prediction_sequences: Sequence[Sequence[int]]) -> float:
    """Mean Flip Rate over a batch of prediction sequences.

    Args:
        prediction_sequences: ``M`` sequences (one per perturbation video),
            each of length ``T_m`` containing predicted class indices.

    Returns:
        Mean of per-sequence flip rates. Lower = more consistent.

    Hendrycks's standard mFR also normalizes by a reference-model mFR;
    that ratio form is computed by passing this output to ``mfr_ratio``.
    """
    if len(prediction_sequences) == 0:
        return 0.0
    return float(sum(flip_rate(seq) for seq in prediction_sequences) / len(prediction_sequences))


def mfr_ratio(model_mfr: float, baseline_mfr: float) -> float:
    """Hendrycks's normalized mFR: ``model_mfr / baseline_mfr``.

    Args:
        model_mfr:    output of ``mfr_predictions`` for the evaluated model.
        baseline_mfr: same metric on the reference model, passed in from
                      a literature table.

    Returns:
        Scalar; lower = relatively more consistent than the baseline.

    Raises:
        ValueError: if ``baseline_mfr`` is zero.
    """
    if baseline_mfr == 0.0:
        raise ValueError("baseline_mfr is zero — ratio undefined")
    return float(model_mfr) / float(baseline_mfr)


def stage_wise_flip_rate(
    stage_outputs: Sequence[torch.Tensor],
    label_fn: Callable[[torch.Tensor], Hashable],
) -> float:
    """Flip rate at an arbitrary pipeline stage.

    Args:
        stage_outputs: ``T`` tensors, one per perturbation frame, holding
            this stage's output (e.g. a (D,)-shaped embedding or
            (K,)-shaped concept activation vector). Shapes must match.
        label_fn: maps a stage tensor to a hashable label that defines
            "the same prediction at this stage." Examples:
              - prediction stage: ``lambda t: int(t.argmax().item())``
              - concept top-k:    ``lambda t: frozenset(t.topk(5).indices.tolist())``

    Returns:
        Frame-to-frame flip rate per ``flip_rate`` semantics.
    """
    labels = [label_fn(t) for t in stage_outputs]
    return flip_rate(labels)
