"""Prediction-stage metrics: accuracy, mCE, RmCE.

mCE and RmCE follow Hendrycks and Dietterich (2019). They are ratios
against an external baseline model's errors on the same corruption grid;
the baseline's per-corruption error rates are passed in explicitly so
this module stays free of any model assumption.

mCE is the standard headline robustness metric (corrupted error
normalized by a reference model's corrupted error). RmCE additionally
strips the clean-accuracy confound by comparing corruption-induced
*deltas* — useful when models being compared have different clean
accuracies.
"""

from __future__ import annotations

import torch


def accuracy(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Top-1 accuracy.

    Args:
        pred:   ``(N,)`` int tensor of predicted class indices.
        target: ``(N,)`` int tensor of ground-truth class indices.

    Returns:
        Scalar accuracy in [0, 1].
    """
    if pred.shape != target.shape:
        raise ValueError(f"shape mismatch: {tuple(pred.shape)} vs {tuple(target.shape)}")
    if pred.numel() == 0:
        return 0.0
    return float((pred == target).float().mean().item())


def mce(model_errors: dict[str, list[float]], baseline_errors: dict[str, list[float]]) -> dict:
    """Mean Corruption Error.

    For each corruption type ``c``, ``CE_c = sum_s err_model_{c,s} /
    sum_s err_baseline_{c,s}``. mCE is the unweighted mean of CE_c
    across corruption types. Lower = more robust.

    Args:
        model_errors:    ``{corruption_name: [err_sev1, ..., err_sev5]}``,
                         errors in [0, 1].
        baseline_errors: same shape; reference model's errors.

    Returns:
        ``{"mce": float, "per_corruption": {name: CE_c}}``.

    Raises:
        ValueError: if the corruption-name sets disagree, or if any
            baseline sum is zero.
    """
    if set(model_errors) != set(baseline_errors):
        raise ValueError(
            f"corruption-set mismatch: model has {sorted(model_errors)}, "
            f"baseline has {sorted(baseline_errors)}"
        )
    per: dict[str, float] = {}
    for c in sorted(model_errors):
        m_sum = float(sum(model_errors[c]))
        b_sum = float(sum(baseline_errors[c]))
        if b_sum == 0.0:
            raise ValueError(f"baseline sum-of-errors is zero for corruption {c!r}")
        per[c] = m_sum / b_sum
    return {"mce": float(sum(per.values()) / len(per)), "per_corruption": per}


def rmce(
    model_errors: dict[str, list[float]],
    model_clean_error: float,
    baseline_errors: dict[str, list[float]],
    baseline_clean_error: float,
) -> dict:
    """Relative mean Corruption Error.

    For each corruption type ``c``, ``RCE_c = sum_s (err_model_{c,s} -
    err_model_clean) / sum_s (err_baseline_{c,s} - err_baseline_clean)``.
    Strips the clean-accuracy confound from mCE: a model that's already
    accurate on clean data shouldn't get a free pass on the robustness
    metric.

    Args:
        model_errors:        ``{corruption_name: [err_sev1, ..., err_sev5]}``.
        model_clean_error:   scalar in [0, 1].
        baseline_errors:     same shape as model_errors.
        baseline_clean_error: scalar in [0, 1].

    Returns:
        ``{"rmce": float, "per_corruption": {name: RCE_c}}``.

    Raises:
        ValueError: if corruption-name sets disagree, or if any baseline
            delta sum is zero.
    """
    if set(model_errors) != set(baseline_errors):
        raise ValueError(
            f"corruption-set mismatch: model has {sorted(model_errors)}, "
            f"baseline has {sorted(baseline_errors)}"
        )
    per: dict[str, float] = {}
    for c in sorted(model_errors):
        m_delta = sum(e - model_clean_error for e in model_errors[c])
        b_delta = sum(e - baseline_clean_error for e in baseline_errors[c])
        if b_delta == 0.0:
            raise ValueError(
                f"baseline corruption-induced delta is zero for {c!r} — "
                "RmCE is undefined when the baseline degrades none under this corruption"
            )
        per[c] = float(m_delta) / float(b_delta)
    return {"rmce": float(sum(per.values()) / len(per)), "per_corruption": per}
