from __future__ import annotations

import numpy as np


def bootstrap_subject_level(metric_fn, subject_ids, n_boot: int = 1000, seed: int = 42):
    subject_ids = np.asarray(subject_ids).astype(str)
    unique_subjects = np.unique(subject_ids)
    rng = np.random.default_rng(seed)
    subject_to_indices = {sid: np.flatnonzero(subject_ids == sid) for sid in unique_subjects}
    values = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
        indices = np.concatenate([subject_to_indices[sid] for sid in sampled])
        try:
            values.append(float(metric_fn(indices)))
        except ValueError:
            continue
    if not values:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan}
    arr = np.asarray(values, dtype=float)
    return {"mean": float(arr.mean()), "lo": float(np.percentile(arr, 2.5)), "hi": float(np.percentile(arr, 97.5))}
