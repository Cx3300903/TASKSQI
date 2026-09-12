from __future__ import annotations

import numpy as np


def risk_coverage_curve(y_true, y_pred, reliability, coverages=(1.0, 0.95, 0.9, 0.85, 0.8, 0.7)):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    reliability = np.asarray(reliability)
    order = np.argsort(-reliability)
    rows = []
    n = len(y_true)
    for cov in coverages:
        k = max(1, int(np.ceil(cov * n)))
        idx = order[:k]
        acc = float((y_true[idx] == y_pred[idx]).mean())
        rows.append({"coverage": float(k / n), "risk": 1.0 - acc, "accuracy": acc, "n": int(k)})
    return rows


def aurc(curve_rows) -> float:
    pts = sorted((row["coverage"], row["risk"]) for row in curve_rows)
    x = np.asarray([p[0] for p in pts], dtype=float)
    y = np.asarray([p[1] for p in pts], dtype=float)
    return float(np.trapz(y, x))


def risk_at_coverage(curve_rows, target_coverage: float) -> float:
    eligible = [row for row in curve_rows if row["coverage"] >= target_coverage]
    if not eligible:
        raise ValueError(f"No curve point reaches coverage {target_coverage}")
    return float(min(eligible, key=lambda row: row["coverage"])["risk"])
