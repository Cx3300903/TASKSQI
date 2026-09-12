from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "outputs" / "figures" / "fig1_sigmoid_icon"

DPI = 600
FIGSIZE = (1.15, 1.15)
COLOR = "#000000"


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    x = np.linspace(-5.0, 5.0, 800)
    y = 1.0 / (1.0 + np.exp(-x))

    x_plot = 0.16 + 0.68 * (x - x.min()) / (x.max() - x.min())
    y_plot = 0.18 + 0.64 * (y - y.min()) / (y.max() - y.min())

    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.set_facecolor((1, 1, 1, 0))
    ax.plot(
        x_plot,
        y_plot,
        color=COLOR,
        linewidth=5.0,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )

    out_path = OUT_DIR / "fig1_sigmoid_black_600dpi.png"
    fig.savefig(out_path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(out_path)


if __name__ == "__main__":
    main()
