#!/usr/bin/env python3
"""Run an AFQMC command and plot block energy statistics in real time.

Usage:
    python live_energy_plot.py -- python test_sd.py
    python live_energy_plot.py --save-live energy_live.png -- python test_sd.py

The script parses AFQMC logger lines like:
    Block 12/100 e_blk=-108.95 e_est=-108.90 ...
and updates:
    - scatter of e_blk
    - running mean
    - running stderr band (mean +/- stderr)
"""

from __future__ import annotations

import argparse
import math
import os
import re
import subprocess
import sys
from dataclasses import dataclass

import numpy as np


BLOCK_RE = re.compile(
    r"Block\s+(\d+)/(\d+)\s+e_blk=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)
DONE_RE = re.compile(
    r"AFQMC done:\s*E=([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)\s*\+/-\s*([+-]?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?)"
)


@dataclass
class RunningStats:
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


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Live AFQMC block-energy scatter + running mean/stderr plot."
    )
    parser.add_argument(
        "--title",
        default="AFQMC Live Energy Monitor",
        help="Figure title.",
    )
    parser.add_argument(
        "--refresh-every",
        type=int,
        default=1,
        help="Redraw every N parsed blocks.",
    )
    parser.add_argument(
        "--save-live",
        default=None,
        help="If set, overwrite this image file on each refresh.",
    )
    parser.add_argument(
        "--no-show",
        action="store_true",
        help="Disable interactive window (useful on headless machines).",
    )
    parser.add_argument(
        "command",
        nargs=argparse.REMAINDER,
        help="Command to run after '--'. Example: -- python test_sd.py",
    )
    args = parser.parse_args()
    if args.command and args.command[0] == "--":
        args.command = args.command[1:]
    if not args.command:
        parser.error("No command given. Use: live_energy_plot.py -- <your command>")
    if args.refresh_every <= 0:
        parser.error("--refresh-every must be >= 1")
    return args


def _setup_matplotlib(no_show: bool):
    if no_show:
        import matplotlib

        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def main() -> int:
    args = _parse_args()

    no_show = bool(args.no_show)
    if (not no_show) and (os.environ.get("DISPLAY") is None):
        print(
            "[live_energy_plot] DISPLAY is not set; switching to --no-show mode.",
            file=sys.stderr,
        )
        no_show = True

    plt = _setup_matplotlib(no_show)
    if not no_show:
        plt.ion()

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.set_xlabel("Block")
    ax.set_ylabel("Energy (Ha)")
    ax.set_title(args.title)

    scatter = ax.scatter([], [], s=26, alpha=0.75, label="e_blk", c="tab:blue")
    (line_mean,) = ax.plot([], [], lw=2.0, c="tab:red", label="running mean")
    band = None
    summary = ax.text(
        0.02,
        0.98,
        "",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
    )
    ax.legend(loc="best")

    block_ids: list[int] = []
    e_blks: list[float] = []
    means: list[float] = []
    stderrs: list[float] = []
    stats = RunningStats()

    final_mean = None
    final_err = None

    def redraw(force: bool = False) -> None:
        nonlocal band
        if not force and (len(block_ids) % args.refresh_every != 0):
            return
        if not block_ids:
            return

        x = np.asarray(block_ids, dtype=float)
        y = np.asarray(e_blks, dtype=float)
        m = np.asarray(means, dtype=float)
        se = np.asarray(stderrs, dtype=float)

        scatter.set_offsets(np.column_stack((x, y)))
        line_mean.set_data(x, m)

        if band is not None:
            band.remove()
        band = ax.fill_between(x, m - se, m + se, color="tab:red", alpha=0.18)

        ymin = min(np.min(y), np.min(m - se))
        ymax = max(np.max(y), np.max(m + se))
        span = max(ymax - ymin, 1.0e-6)
        pad = 0.12 * span

        xmax = x[-1]
        ax.set_xlim(0.5, max(2.0, xmax + 1.0))
        ax.set_ylim(ymin - pad, ymax + pad)

        msg = f"n={stats.n}\nmean={stats.mean:.12f}\nstderr={stderrs[-1]:.3e}"
        if final_mean is not None and final_err is not None:
            msg += f"\nfinal={final_mean:.12f} +/- {final_err:.3e}"
        summary.set_text(msg)

        fig.canvas.draw_idle()
        if not no_show:
            fig.canvas.flush_events()
            plt.pause(0.001)
        if args.save_live:
            fig.savefig(args.save_live, dpi=140, bbox_inches="tight")

    print(f"[live_energy_plot] Running command: {' '.join(args.command)}")
    proc = subprocess.Popen(
        args.command,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    assert proc.stdout is not None
    for line in proc.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()

        blk = BLOCK_RE.search(line)
        if blk:
            blk_id = int(blk.group(1))
            e_blk = float(blk.group(3))
            mean, stderr = stats.update(e_blk)

            block_ids.append(blk_id)
            e_blks.append(e_blk)
            means.append(mean)
            stderrs.append(stderr)
            redraw(force=False)
            continue

        done = DONE_RE.search(line)
        if done:
            final_mean = float(done.group(1))
            final_err = float(done.group(2))
            redraw(force=True)

    ret = proc.wait()
    redraw(force=True)

    if args.save_live and not os.path.exists(args.save_live):
        fig.savefig(args.save_live, dpi=140, bbox_inches="tight")

    if not no_show:
        plt.ioff()
        plt.show()

    return ret


if __name__ == "__main__":
    raise SystemExit(main())

