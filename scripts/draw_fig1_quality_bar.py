from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "fig1_quality_bar"

DPI = 600
FIGSIZE = (5.8, 0.55)


def rounded_gradient_bar(path: Path) -> None:
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor((1, 1, 1, 0))

    x0, y0, width, height = 0.03, 0.24, 0.94, 0.52
    radius = 0.075

    shadow = FancyBboxPatch(
        (x0, y0 - 0.025),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        facecolor="#5C7D8C",
        alpha=0.16,
        zorder=0,
    )
    ax.add_patch(shadow)

    clip = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=0,
        facecolor="none",
        zorder=2,
    )
    ax.add_patch(clip)

    cmap = LinearSegmentedColormap.from_list(
        "quality_bar",
        [
            (0.00, "#F53242"),
            (0.28, "#FF8E3A"),
            (0.50, "#FFE96A"),
            (0.72, "#75D65A"),
            (1.00, "#00B978"),
        ],
    )
    grad = np.linspace(0, 1, 1600)[None, :]
    im = ax.imshow(
        grad,
        extent=[x0, x0 + width, y0, y0 + height],
        cmap=cmap,
        aspect="auto",
        interpolation="bicubic",
        zorder=1,
    )
    im.set_clip_path(clip)

    highlight = np.ones((180, 1600, 4))
    highlight[:, :, :3] = 1.0
    vertical_alpha = np.linspace(0.30, 0.0, 180)[:, None]
    horizontal_alpha = np.ones((1, 1600))
    highlight[:, :, 3] = vertical_alpha * horizontal_alpha
    hi = ax.imshow(
        highlight,
        extent=[x0, x0 + width, y0 + height * 0.50, y0 + height],
        aspect="auto",
        interpolation="bicubic",
        zorder=3,
    )
    hi.set_clip_path(clip)

    border = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle=f"round,pad=0,rounding_size={radius}",
        linewidth=1.0,
        edgecolor=(1, 1, 1, 0.35),
        facecolor="none",
        zorder=4,
    )
    ax.add_patch(border)

    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rounded_gradient_bar(OUT_DIR / "fig1_quality_gradient_bar_600dpi.png")
    print(OUT_DIR)


if __name__ == "__main__":
    main()
