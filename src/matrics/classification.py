from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score


def classification_metrics(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    per_class_f1 = f1_score(y_true, y_pred, labels=[0, 1, 2, 3, 4], average=None, zero_division=0)
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "kappa": float(cohen_kappa_score(y_true, y_pred)),
        "f1_wake": float(per_class_f1[0]),
        "f1_n1": float(per_class_f1[1]),
        "f1_n2": float(per_class_f1[2]),
        "f1_n3": float(per_class_f1[3]),
        "f1_rem": float(per_class_f1[4]),
    }
