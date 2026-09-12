from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "fig1_degradation_icons"

COLOR = "#73A9DB"
DPI = 600
FIGSIZE = (1.55, 0.72)


def setup_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor((1, 1, 1, 0))


def save_scatter_icon(path: Path) -> None:
    points = np.array(
        [
            [0.18, 0.36],
            [0.29, 0.43],
            [0.40, 0.38],
            [0.52, 0.44],
            [0.65, 0.37],
            [0.77, 0.44],
            [0.24, 0.57],
            [0.36, 0.61],
            [0.48, 0.55],
            [0.60, 0.61],
            [0.72, 0.56],
            [0.84, 0.63],
            [0.31, 0.75],
            [0.46, 0.79],
            [0.60, 0.74],
            [0.74, 0.80],
            [0.40, 0.24],
            [0.55, 0.27],
            [0.70, 0.25],
            [0.82, 0.32],
            [0.28, 0.25],
            [0.50, 0.68],
            [0.63, 0.49],
            [0.36, 0.50],
        ]
    )

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)
    setup_axis(ax)
    ax.scatter(points[:, 0], points[:, 1], s=28, c=COLOR, edgecolors="none", alpha=1.0)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_wave_icon(path: Path) -> None:
    x = np.linspace(0.12, 0.88, 500)
    y = 0.50 + 0.23 * np.sin(2 * np.pi * 2.05 * (x - 0.12) / 0.76)

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)
    setup_axis(ax)
    ax.plot(x, y, color=COLOR, linewidth=5.2, solid_capstyle="round", solid_joinstyle="round")
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_dashed_icon(path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)
    setup_axis(ax)
    segments = [(0.08, 0.25), (0.31, 0.39), (0.45, 0.58), (0.64, 0.70), (0.76, 0.92)]
    for x0, x1 in segments:
        ax.plot(
            [x0, x1],
            [0.50, 0.50],
            color=COLOR,
            linewidth=5.0,
            solid_capstyle="butt",
        )
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def save_preview(path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(1.55, 2.24), dpi=DPI)
    fig.patch.set_alpha(0)
    for ax in axes:
        setup_axis(ax)

    points = np.array(
        [
            [0.18, 0.36],
            [0.29, 0.43],
            [0.40, 0.38],
            [0.52, 0.44],
            [0.65, 0.37],
            [0.77, 0.44],
            [0.24, 0.57],
            [0.36, 0.61],
            [0.48, 0.55],
            [0.60, 0.61],
            [0.72, 0.56],
            [0.84, 0.63],
            [0.31, 0.75],
            [0.46, 0.79],
            [0.60, 0.74],
            [0.74, 0.80],
            [0.40, 0.24],
            [0.55, 0.27],
            [0.70, 0.25],
            [0.82, 0.32],
            [0.28, 0.25],
            [0.50, 0.68],
            [0.63, 0.49],
            [0.36, 0.50],
        ]
    )
    axes[0].scatter(points[:, 0], points[:, 1], s=28, c=COLOR, edgecolors="none")

    x = np.linspace(0.12, 0.88, 500)
    y = 0.50 + 0.23 * np.sin(2 * np.pi * 2.05 * (x - 0.12) / 0.76)
    axes[1].plot(x, y, color=COLOR, linewidth=5.2, solid_capstyle="round", solid_joinstyle="round")

    segments = [(0.08, 0.25), (0.31, 0.39), (0.45, 0.58), (0.64, 0.70), (0.76, 0.92)]
    for x0, x1 in segments:
        axes[2].plot(
            [x0, x1],
            [0.50, 0.50],
            color=COLOR,
            linewidth=5.0,
            solid_capstyle="butt",
        )

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.16)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    save_scatter_icon(OUT_DIR / "fig1_icon_scatter_73a9db_600dpi.png")
    save_wave_icon(OUT_DIR / "fig1_icon_wave_73a9db_600dpi.png")
    save_dashed_icon(OUT_DIR / "fig1_icon_dashed_73a9db_600dpi.png")
    save_preview(OUT_DIR / "fig1_icons_preview_73a9db_600dpi.png")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
