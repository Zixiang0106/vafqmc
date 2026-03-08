"""AFQMC statistics and post-processing helpers."""

from __future__ import annotations

from typing import Optional

import numpy as onp


def reject_outliers(
    samples: onp.ndarray,
    obs_index: int,
    *,
    m: float = 10.0,
    min_threshold: float = 1.0e-5,
) -> tuple[onp.ndarray, onp.ndarray]:
    """Reject outliers in ``samples`` by robust dispersion on selected column."""
    target = onp.asarray(samples[:, obs_index], dtype=onp.float64)
    median_val = onp.median(target)
    dist = onp.abs(target - median_val)
    mdev = onp.median(dist)
    q1, q3 = onp.percentile(target, [25, 75])
    iqr = q3 - q1
    normalized_iqr = iqr / 1.349
    dispersion = max(float(mdev), float(normalized_iqr), float(min_threshold))
    score = dist / dispersion
    mask = score < float(m)
    return samples[mask], mask


def blocking_analysis(
    weights: onp.ndarray,
    energies: onp.ndarray,
    *,
    neql: int = 0,
) -> tuple[float, Optional[float]]:
    """Weighted blocking analysis used in ad_afqmc."""
    weights = onp.asarray(weights, dtype=onp.float64).reshape(-1)
    energies = onp.asarray(energies, dtype=onp.float64).reshape(-1)
    if weights.size != energies.size:
        raise ValueError("weights and energies must have same length.")
    if weights.size == 0:
        return onp.nan, 0.0

    neql_i = max(int(neql), 0)
    weights = weights[neql_i:]
    energies = energies[neql_i:]
    if weights.size == 0:
        return onp.nan, 0.0

    weighted_energies = weights * energies
    wsum = weights.sum()
    if wsum <= 0.0:
        return float(onp.mean(energies)), 0.0
    mean_energy = float(weighted_energies.sum() / wsum)

    n_samples = weights.shape[0]
    block_sizes = onp.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 10000])
    prev_error = 0.0
    plateau_error = None
    errors: list[float] = []

    for bsz in block_sizes[block_sizes < n_samples / 2.0]:
        n_blocks = int(n_samples // int(bsz))
        if n_blocks <= 1:
            continue

        blocked_weights = onp.zeros(n_blocks, dtype=onp.float64)
        blocked_energies = onp.zeros(n_blocks, dtype=onp.float64)
        for j in range(n_blocks):
            sl = slice(j * int(bsz), (j + 1) * int(bsz))
            blocked_weights[j] = weights[sl].sum()
            if blocked_weights[j] > 0.0:
                blocked_energies[j] = weighted_energies[sl].sum() / blocked_weights[j]
            else:
                blocked_energies[j] = 0.0

        v1 = blocked_weights.sum()
        if v1 <= 0.0:
            continue
        v2 = (blocked_weights**2).sum()
        mean_b = float((blocked_weights * blocked_energies).sum() / v1)
        denom = (v1 - v2 / v1) * (n_blocks - 1)
        if denom <= 0.0:
            continue
        var = (blocked_weights * (blocked_energies - mean_b) ** 2).sum() / denom
        error = float(onp.sqrt(max(var, 0.0)))
        errors.append(error)
        if error < 1.05 * prev_error and plateau_error is None:
            plateau_error = max(error, prev_error)
        prev_error = error

    if plateau_error is None:
        if errors:
            plateau_error = 2.0 * errors[0]
        else:
            plateau_error = 0.0
    return mean_energy, float(plateau_error)


def analyze_energy_blocks(
    block_energies: onp.ndarray,
    block_weights: Optional[onp.ndarray] = None,
    *,
    reject_energy_outliers: bool = True,
    outlier_m: float = 10.0,
    outlier_min_threshold: float = 1.0e-5,
    neql: int = 0,
) -> tuple[float, float, int]:
    """Post-process block energies with optional outlier rejection and blocking error."""
    energies = onp.asarray(block_energies, dtype=onp.float64).reshape(-1)
    if energies.size == 0:
        return onp.nan, 0.0, 0

    if block_weights is None:
        weights = onp.ones_like(energies)
    else:
        weights = onp.asarray(block_weights, dtype=onp.float64).reshape(-1)
        if weights.size != energies.size:
            raise ValueError("block_weights size mismatch with block_energies.")

    samples = onp.stack((weights, energies), axis=1)
    n_outliers = 0
    if reject_energy_outliers and samples.shape[0] > 3:
        samples_clean, _ = reject_outliers(
            samples,
            1,
            m=outlier_m,
            min_threshold=outlier_min_threshold,
        )
        n_outliers = int(samples.shape[0] - samples_clean.shape[0])
    else:
        samples_clean = samples

    if samples_clean.shape[0] == 0:
        mean = float(onp.mean(energies))
        err = float(onp.std(energies, ddof=1) / onp.sqrt(energies.size)) if energies.size > 1 else 0.0
        return mean, err, int(n_outliers)

    w = samples_clean[:, 0]
    e = samples_clean[:, 1]
    mean, err = blocking_analysis(w, e, neql=neql)
    if err is None or (not onp.isfinite(err)):
        err = float(onp.std(e, ddof=1) / onp.sqrt(e.size)) if e.size > 1 else 0.0
    return float(mean), float(err), int(n_outliers)


__all__ = [
    "analyze_energy_blocks",
    "blocking_analysis",
    "reject_outliers",
]
