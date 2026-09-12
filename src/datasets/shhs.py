from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import Dataset

from .preprocessing import load_epochs_npz


class SHHSDataset(Dataset):
    def __init__(
        self,
        npz_path: str,
        split_file: str | None = None,
        normalize: str = "subject",
        verbose: bool = True,
        expected_samples: int | None = 3000,
    ):
        subjects = None
        name = Path(npz_path).stem
        if split_file:
            subjects = {line.strip() for line in Path(split_file).read_text(encoding="utf-8").splitlines() if line.strip()}
            name = f"{Path(npz_path).stem}:{Path(split_file).stem}"
            if verbose:
                print(f"[{name}] loaded split file: {split_file} subjects={len(subjects)}", flush=True)
        x, y, subject_ids = load_epochs_npz(
            npz_path,
            normalize=normalize,
            subject_filter=subjects,
            expected_samples=expected_samples,
            verbose=verbose,
            name=name,
        )
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(y)
        self.subject_ids = subject_ids

    def __len__(self) -> int:
        return len(self.y)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx], self.subject_ids[idx]
