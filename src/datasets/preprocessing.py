from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy import signal
from tqdm import tqdm

STAGE_TO_ID = {"W": 0, "N1": 1, "N2": 2, "N3": 3, "N4": 3, "REM": 4, "R": 4}
ID_TO_STAGE = {0: "W", 1: "N1", 2: "N2", 3: "N3", 4: "REM"}


def bandpass_filter(x: np.ndarray, fs: float = 100.0, low: float = 0.3, high: float = 35.0) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=fs, output="sos")
    return signal.sosfiltfilt(sos, x, axis=-1).astype(np.float32)


def resample_to_100hz(x: np.ndarray, orig_fs: float) -> np.ndarray:
    if abs(orig_fs - 100.0) < 1e-6:
        return x.astype(np.float32)
    target_len = int(round(x.shape[-1] * 100.0 / orig_fs))
    return signal.resample(x, target_len, axis=-1).astype(np.float32)


def epoch_signal(x: np.ndarray, fs: float = 100.0, seconds: int = 30) -> np.ndarray:
    n = int(fs * seconds)
    usable = (x.shape[-1] // n) * n
    return x[..., :usable].reshape(-1, n).astype(np.float32)


def normalize_by_subject(x: np.ndarray, subject_ids: np.ndarray, verbose: bool = False, desc: str = "normalize subjects") -> np.ndarray:
    out = x.astype(np.float32).copy()
    unique_subjects = np.unique(subject_ids)
    iterator = tqdm(unique_subjects, desc=desc, unit="subject", ascii=True) if verbose else unique_subjects
    for sid in iterator:
        idx = subject_ids == sid
        mu = out[idx].mean()
        sd = out[idx].std() + 1e-6
        out[idx] = (out[idx] - mu) / sd
    return out


def load_epochs_npz(
    path: str | Path,
    normalize: str = "subject",
    subject_filter: set[str] | None = None,
    expected_samples: int | None = 3000,
    verbose: bool = False,
    name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    label = name or str(path)
    if verbose:
        print(f"[{label}] opening NPZ: {path}", flush=True)
    data = np.load(path, allow_pickle=True)
    y = data["y"].astype(np.int64)
    subject_ids = data["subject_ids"].astype(str)
    keep = np.isin(y, [0, 1, 2, 3, 4])
    if subject_filter is not None:
        keep &= np.fromiter((sid in subject_filter for sid in subject_ids), dtype=bool, count=len(subject_ids))
    if verbose:
        print(
            {
                "dataset": label,
                "epochs_in_npz": int(len(y)),
                "epochs_selected": int(keep.sum()),
                "subjects_selected": int(len(np.unique(subject_ids[keep]))),
                "normalize": normalize,
            },
            flush=True,
        )
        print(f"[{label}] loading selected EEG array...", flush=True)
    x = data["x"][keep].astype(np.float32)
    if x.ndim == 2:
        x = x[:, None, :]
    if expected_samples is not None and x.shape[1:] != (1, expected_samples):
        raise ValueError(f"Expected x shape [N,1,{expected_samples}], got {x.shape}")
    y, subject_ids = y[keep], subject_ids[keep]
    if normalize == "subject":
        x = normalize_by_subject(x, subject_ids, verbose=verbose, desc=f"{label} normalize")
    elif normalize == "global":
        if verbose:
            print(f"[{label}] applying global normalization...", flush=True)
        x = ((x - x.mean()) / (x.std() + 1e-6)).astype(np.float32)
    elif normalize not in ("none", None):
        raise ValueError(f"Unknown normalization: {normalize}")
    if verbose:
        print(f"[{label}] ready: x={x.shape} y={y.shape}", flush=True)
    return x, y, subject_ids
