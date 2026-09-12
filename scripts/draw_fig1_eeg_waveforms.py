from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "shhs_c4a1_125hz_epochs.npz"
OUT_DIR = ROOT / "outputs" / "figures" / "fig1_eeg_waveforms"

FS = 125
DPI = 600
FIGSIZE = (6.8, 1.35)
CLEAN_COLOR = "#1E88E5"
DEGRADED_COLOR = "#8E44AD"


def robust_normalize(signal: np.ndarray) -> np.ndarray:
    signal = signal.astype(np.float64)
    signal = signal - np.median(signal)
    scale = np.percentile(np.abs(signal), 95)
    if scale <= 1e-12:
        scale = np.std(signal) + 1e-12
    signal = signal / scale
    return np.clip(signal, -1.8, 1.8)


def choose_epoch(x: np.ndarray, start: int = 1200, n_candidates: int = 256) -> np.ndarray:
    candidates = x[start : start + n_candidates, 0, :]
    activity = np.std(candidates, axis=1) + 0.15 * np.mean(np.abs(np.diff(candidates, axis=1)), axis=1)
    epoch = candidates[int(np.argsort(activity)[len(activity) // 2])]
    return robust_normalize(epoch)


def degrade_signal(clean: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = clean.size
    t = np.arange(n) / FS
    degraded = 0.92 * clean.copy()
    degraded += 0.18 * rng.standard_normal(n)
    degraded += 0.22 * np.sin(2 * np.pi * 0.22 * t + 0.4)
    degraded += 0.10 * np.sin(2 * np.pi * 16.0 * t)

    burst_windows = [(2.6, 5.2), (9.0, 12.2), (23.2, 26.0)]
    for start_s, end_s in burst_windows:
        lo, hi = int(start_s * FS), int(end_s * FS)
        degraded[lo:hi] += 0.24 * rng.standard_normal(hi - lo)

    degraded = robust_normalize(degraded)

    gap_start, gap_end = int(13.8 * FS), int(20.2 * FS)
    degraded[gap_start:gap_end] = np.nan
    return degraded


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 30)
    ax.set_ylim(-1.9, 1.9)
    ax.axis("off")
    ax.margins(x=0, y=0)
    ax.set_facecolor((1, 1, 1, 0))


def save_waveform(path: Path, signal: np.ndarray, color: str, dashed_gap: bool = False) -> None:
    t = np.arange(signal.size) / FS
    fig, ax = plt.subplots(figsize=FIGSIZE, dpi=DPI)
    fig.patch.set_alpha(0)
    style_axis(ax)

    ax.plot(
        t,
        signal,
        color=color,
        linewidth=2.15,
        solid_capstyle="round",
        solid_joinstyle="round",
        antialiased=True,
    )

    if dashed_gap:
        gap = np.flatnonzero(np.isnan(signal))
        if gap.size:
            x0, x1 = t[gap[0]], t[gap[-1]]
            y = 0.0
            ax.plot(
                [x0, x1],
                [y, y],
                color=color,
                linewidth=2.6,
                linestyle=(0, (3.2, 2.7)),
                dash_capstyle="round",
                alpha=0.96,
            )

    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def save_preview(path: Path, clean: np.ndarray, degraded: np.ndarray) -> None:
    t = np.arange(clean.size) / FS
    fig, axes = plt.subplots(2, 1, figsize=(6.8, 2.55), dpi=DPI)
    fig.patch.set_alpha(0)
    for ax, sig, color in zip(axes, [clean, degraded], [CLEAN_COLOR, DEGRADED_COLOR]):
        style_axis(ax)
        ax.plot(t, sig, color=color, linewidth=2.05, solid_capstyle="round", antialiased=True)
    gap = np.flatnonzero(np.isnan(degraded))
    if gap.size:
        axes[1].plot(
            [t[gap[0]], t[gap[-1]]],
            [0.0, 0.0],
            color=DEGRADED_COLOR,
            linewidth=2.5,
            linestyle=(0, (3.2, 2.7)),
            dash_capstyle="round",
        )
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0, hspace=0.18)
    fig.savefig(path, dpi=DPI, transparent=True, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = np.load(DATA_PATH)
    clean = choose_epoch(data["x"])
    degraded = degrade_signal(clean, np.random.default_rng(20260909))

    save_waveform(OUT_DIR / "fig1_clean_eeg_blue_600dpi.png", clean, CLEAN_COLOR)
    save_waveform(OUT_DIR / "fig1_degraded_eeg_purple_600dpi.png", degraded, DEGRADED_COLOR, dashed_gap=True)
    save_preview(OUT_DIR / "fig1_clean_degraded_preview_600dpi.png", clean, degraded)

    print(OUT_DIR)
    print("fig1_clean_eeg_blue_600dpi.png")
    print("fig1_degraded_eeg_purple_600dpi.png")
    print("fig1_clean_degraded_preview_600dpi.png")


if __name__ == "__main__":
    main()
