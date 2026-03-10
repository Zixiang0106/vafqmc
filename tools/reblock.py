#!/usr/bin/env python3
"""Analyze AFQMC raw.dat after dropping the first N blocks.

Expected raw format (as written by hafqmc.afqmc drivers):
    # block e_blk e_est wsum block_weight elapsed_s
    1 ...
    2 ...

Examples
--------
# Drop first 20 blocks, then print final E +/- err
python tools/trim_raw_blocks.py -i raw.dat -n 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def reject_outliers(
    samples: np.ndarray,
    obs_index: int,
    *,
    m: float = 10.0,
    min_threshold: float = 1.0e-5,
) -> tuple[np.ndarray, np.ndarray]:
    target = np.asarray(samples[:, obs_index], dtype=np.float64)
    median_val = np.median(target)
    dist = np.abs(target - median_val)
    mdev = np.median(dist)
    q1, q3 = np.percentile(target, [25, 75])
    iqr = q3 - q1
    normalized_iqr = iqr / 1.349
    dispersion = max(float(mdev), float(normalized_iqr), float(min_threshold))
    score = dist / dispersion
    mask = score < float(m)
    return samples[mask], mask


def blocking_analysis(
    weights: np.ndarray,
    energies: np.ndarray,
    *,
    neql: int = 0,
) -> tuple[float, float]:
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    energies = np.asarray(energies, dtype=np.float64).reshape(-1)
    if weights.size != energies.size:
        raise ValueError("weights and energies must have same length.")
    if weights.size == 0:
        return np.nan, 0.0

    neql_i = max(int(neql), 0)
    weights = weights[neql_i:]
    energies = energies[neql_i:]
    if weights.size == 0:
        return np.nan, 0.0

    weighted_energies = weights * energies
    wsum = weights.sum()
    if wsum <= 0.0:
        return float(np.mean(energies)), 0.0
    mean_energy = float(weighted_energies.sum() / wsum)

    n_samples = weights.shape[0]
    block_sizes = np.array([1, 2, 5, 10, 20, 50, 100, 200, 300, 400, 500, 1000, 10000])
    prev_error = 0.0
    plateau_error = None
    errors: list[float] = []

    for bsz in block_sizes[block_sizes < n_samples / 2.0]:
        n_blocks = int(n_samples // int(bsz))
        if n_blocks <= 1:
            continue

        blocked_weights = np.zeros(n_blocks, dtype=np.float64)
        blocked_energies = np.zeros(n_blocks, dtype=np.float64)
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
        error = float(np.sqrt(max(var, 0.0)))
        errors.append(error)
        if error < 1.05 * prev_error and plateau_error is None:
            plateau_error = max(error, prev_error)
        prev_error = error

    if plateau_error is None:
        plateau_error = 2.0 * errors[0] if errors else 0.0
    return mean_energy, float(plateau_error)


def analyze_energy_blocks(
    block_energies: np.ndarray,
    block_weights: np.ndarray | None = None,
    *,
    reject_energy_outliers: bool = True,
    outlier_m: float = 10.0,
    outlier_min_threshold: float = 1.0e-5,
    neql: int = 0,
) -> tuple[float, float, int]:
    energies = np.asarray(block_energies, dtype=np.float64).reshape(-1)
    if energies.size == 0:
        return np.nan, 0.0, 0

    if block_weights is None:
        weights = np.ones_like(energies)
    else:
        weights = np.asarray(block_weights, dtype=np.float64).reshape(-1)
        if weights.size != energies.size:
            raise ValueError("block_weights size mismatch with block_energies.")

    samples = np.stack((weights, energies), axis=1)
    n_outliers = 0
    if reject_energy_outliers and samples.shape[0] > 3:
        samples_clean, _ = reject_outliers(
            samples, 1, m=outlier_m, min_threshold=outlier_min_threshold
        )
        n_outliers = int(samples.shape[0] - samples_clean.shape[0])
    else:
        samples_clean = samples

    if samples_clean.shape[0] == 0:
        mean = float(np.mean(energies))
        err = float(np.std(energies, ddof=1) / np.sqrt(energies.size)) if energies.size > 1 else 0.0
        return mean, err, int(n_outliers)

    w = samples_clean[:, 0]
    e = samples_clean[:, 1]
    mean, err = blocking_analysis(w, e, neql=neql)
    if not np.isfinite(err):
        err = float(np.std(e, ddof=1) / np.sqrt(e.size)) if e.size > 1 else 0.0
    return float(mean), float(err), int(n_outliers)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyze AFQMC raw.dat after dropping first N blocks")
    p.add_argument("-i", "--input", default="raw.dat", help="input raw file path")
    p.add_argument("-n", "--drop", type=int, required=True, help="number of leading blocks to drop")
    p.add_argument(
        "--no-outlier-reject",
        action="store_true",
        help="disable outlier rejection in analyze_energy_blocks",
    )
    return p.parse_args()


def _read_raw(path: Path) -> tuple[list[str], list[str]]:
    header_lines: list[str] = []
    data_lines: list[str] = []

    with path.open("r", encoding="utf-8") as f:
        for raw in f:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            if line.lstrip().startswith("#"):
                header_lines.append(line)
            else:
                data_lines.append(line)

    return header_lines, data_lines


def _extract_energy_and_weight(rows: list[str]) -> tuple[np.ndarray, np.ndarray]:
    energies: list[float] = []
    weights: list[float] = []
    for i, row in enumerate(rows, start=1):
        parts = row.split()
        if len(parts) < 5:
            raise ValueError(
                f"Malformed raw line #{i}: expected >=5 columns, got {len(parts)}"
            )
        # Columns: block e_blk e_est wsum block_weight elapsed_s
        energies.append(float(parts[1]))
        weights.append(float(parts[4]))
    return np.asarray(energies, dtype=np.float64), np.asarray(weights, dtype=np.float64)


def main() -> None:
    args = parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"input file not found: {in_path}")

    if args.drop < 0:
        raise ValueError("--drop must be >= 0")

    _headers, rows = _read_raw(in_path)
    n_total = len(rows)

    if args.drop > n_total:
        raise ValueError(f"--drop={args.drop} exceeds number of blocks={n_total}")

    kept = rows[args.drop :]
    if len(kept) == 0:
        raise ValueError("No blocks left after dropping. Reduce --drop.")

    e_blk, w_blk = _extract_energy_and_weight(kept)
    e_mean, e_err, n_outliers = analyze_energy_blocks(
        e_blk,
        w_blk,
        reject_energy_outliers=not bool(args.no_outlier_reject),
    )
    print(f"input={in_path}")
    print(f"blocks_total={n_total} blocks_dropped={args.drop} blocks_used={len(kept)}")
    print(f"E = {float(e_mean):.12f} +/- {float(e_err):.6e} (outliers={int(n_outliers)})")


if __name__ == "__main__":
    main()
