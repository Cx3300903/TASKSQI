from __future__ import annotations

import numpy as np
import torch
from scipy.signal import welch


def msp_reliability(prob: np.ndarray | torch.Tensor) -> np.ndarray:
    p = prob.detach().cpu().numpy() if torch.is_tensor(prob) else np.asarray(prob)
    return p.max(axis=1)


def predictive_entropy_reliability(prob: np.ndarray | torch.Tensor) -> np.ndarray:
    p = prob.detach().cpu().numpy() if torch.is_tensor(prob) else np.asarray(prob)
    ent = -(p * np.log(p + 1e-12)).sum(axis=1)
    return 1.0 - ent / np.log(p.shape[1])


def spectral_entropy_reliability(x: np.ndarray | torch.Tensor, fs: float = 100.0) -> np.ndarray:
    arr = x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    if arr.ndim == 3:
        arr = arr[:, 0, :]
    vals = []
    for epoch in arr:
        _, pxx = welch(epoch, fs=fs, nperseg=min(512, epoch.shape[-1]))
        pxx = pxx / (pxx.sum() + 1e-12)
        ent = -(pxx * np.log(pxx + 1e-12)).sum() / np.log(len(pxx))
        vals.append(1.0 - ent)
    return np.asarray(vals, dtype=np.float32)
