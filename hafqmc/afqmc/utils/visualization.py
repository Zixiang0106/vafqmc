"""Live AFQMC energy visualization helpers."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass
from typing import Optional

import numpy as np


class _NullEnergyVisualizer:
    def update(self, block_idx: int, e_blk: float) -> None:
        del block_idx, e_blk

    def finalize(self, mean: float, err: float) -> None:
        del mean, err


@dataclass
class _RunningStats:
    n: int = 0
    mean: float = 0.0
    m2: float = 0.0

    def update(self, x: float) -> tuple[float, float]:
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.m2 += delta * delta2
        if self.n > 1:
            var = self.m2 / (self.n - 1)
            stderr = math.sqrt(max(var, 0.0)) / math.sqrt(self.n)
        else:
            stderr = 0.0
        return self.mean, stderr


class _MatplotlibEnergyVisualizer:
    def __init__(
        self,
        *,
        logger: logging.Logger,
        title: str,
        refresh_every: int,
        show: bool,
        save_path: Optional[str],
    ) -> None:
        self.logger = logger
        self.refresh_every = max(int(refresh_every), 1)
        self.save_path = save_path
        self.show = bool(show)
        self.final_mean = None
        self.final_err = None

        # Select backend before importing pyplot.
        if not self.show:
            import matplotlib

            matplotlib.use("Agg", force=True)

        import matplotlib.pyplot as plt

        self.plt = plt
        if self.show:
            self.plt.ion()

        self.fig, self.ax = self.plt.subplots(figsize=(9, 5))
        self.ax.set_xlabel("Block")
        self.ax.set_ylabel("Energy (Ha)")
        self.ax.set_title(title)
        self.scatter = self.ax.scatter([], [], s=24, alpha=0.75, c="tab:blue", label="e_blk")
        (self.line_mean,) = self.ax.plot([], [], lw=2.0, c="tab:red", label="running mean")
        self.band = None
        self.summary = self.ax.text(
            0.02,
            0.98,
            "",
            transform=self.ax.transAxes,
            va="top",
            ha="left",
            fontsize=10,
            bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
        )
        self.ax.legend(loc="best")

        self.blocks: list[int] = []
        self.e_blks: list[float] = []
        self.means: list[float] = []
        self.stderrs: list[float] = []
        self.stats = _RunningStats()

    def update(self, block_idx: int, e_blk: float) -> None:
        mean, stderr = self.stats.update(float(e_blk))
        self.blocks.append(int(block_idx))
        self.e_blks.append(float(e_blk))
        self.means.append(float(mean))
        self.stderrs.append(float(stderr))

        if len(self.blocks) % self.refresh_every == 0:
            self._redraw()

    def finalize(self, mean: float, err: float) -> None:
        self.final_mean = float(mean)
        self.final_err = float(err)
        self._redraw(force=True)
        if self.show:
            self.plt.ioff()
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

    def _redraw(self, *, force: bool = False) -> None:
        if not self.blocks:
            return
        if (not force) and (len(self.blocks) % self.refresh_every != 0):
            return

        x = np.asarray(self.blocks, dtype=float)
        y = np.asarray(self.e_blks, dtype=float)
        m = np.asarray(self.means, dtype=float)
        se = np.asarray(self.stderrs, dtype=float)

        self.scatter.set_offsets(np.column_stack((x, y)))
        self.line_mean.set_data(x, m)

        if self.band is not None:
            self.band.remove()
        self.band = self.ax.fill_between(x, m - se, m + se, color="tab:red", alpha=0.18)

        ymin = min(float(np.min(y)), float(np.min(m - se)))
        ymax = max(float(np.max(y)), float(np.max(m + se)))
        span = max(ymax - ymin, 1.0e-8)
        pad = 0.12 * span

        self.ax.set_xlim(0.5, max(float(x[-1]) + 1.0, 2.0))
        self.ax.set_ylim(ymin - pad, ymax + pad)

        msg = (
            f"n={self.stats.n}\n"
            f"E_mean(run)={self.stats.mean:.12f}\n"
            f"E_err(run)={se[-1]:.3e}"
        )
        if self.final_mean is not None and self.final_err is not None:
            msg += (
                f"\nE_mean(final)={self.final_mean:.12f}\n"
                f"E_err(final)={self.final_err:.3e}"
            )
        self.summary.set_text(msg)

        self.fig.canvas.draw_idle()
        if self.show:
            self.fig.canvas.flush_events()
            self.plt.pause(0.001)
        if self.save_path:
            self.fig.savefig(self.save_path, dpi=140, bbox_inches="tight")


def build_energy_visualizer(
    *,
    enabled: bool,
    logger: logging.Logger,
    title: str = "AFQMC Live Energy",
    refresh_every: int = 1,
    show: bool = True,
    save_path: Optional[str] = None,
):
    """Return a live energy visualizer (or no-op fallback)."""
    if not enabled:
        return _NullEnergyVisualizer()

    show_flag = bool(show)
    if show_flag and (os.environ.get("DISPLAY") is None):
        logger.info("Visualization enabled but DISPLAY is unset; switching to headless mode.")
        show_flag = False

    try:
        return _MatplotlibEnergyVisualizer(
            logger=logger,
            title=title,
            refresh_every=refresh_every,
            show=show_flag,
            save_path=save_path,
        )
    except Exception as exc:
        logger.warning("Visualization disabled; failed to initialize matplotlib. error=%s", str(exc))
        return _NullEnergyVisualizer()
