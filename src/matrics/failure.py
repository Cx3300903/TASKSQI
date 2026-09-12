from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def failure_metrics(failure_label, failure_score) -> dict[str, float]:
    y = np.asarray(failure_label).astype(int)
    s = np.asarray(failure_score).astype(float)
    if len(np.unique(y)) < 2:
        return {"auroc": float("nan"), "auprc": float("nan")}
    return {"auroc": float(roc_auc_score(y, s)), "auprc": float(average_precision_score(y, s))}
