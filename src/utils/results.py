from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def save_csv(rows, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def plot_risk_coverage(curves: dict[str, list[dict]], out_prefix: str, title: str):
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(5.2, 3.6))
    for name, rows in curves.items():
        pts = sorted(rows, key=lambda r: r["coverage"])
        plt.plot([r["coverage"] for r in pts], [r["risk"] for r in pts], marker="o", label=name)
    plt.xlabel("Coverage")
    plt.ylabel("Risk")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png", dpi=200)
    plt.close()


def plot_corruption_curve(rows, out_prefix: str):
    Path(out_prefix).parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    plt.figure(figsize=(5.2, 3.6))
    for name, sub in df.groupby("corruption"):
        plt.plot(sub["severity"], sub["quality"], marker="o", label=name)
    plt.xlabel("Corruption severity")
    plt.ylabel("Mean TaskSQI")
    plt.title("Corruption Quality Curve")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(out_prefix + ".pdf")
    plt.savefig(out_prefix + ".png", dpi=200)
    plt.close()
