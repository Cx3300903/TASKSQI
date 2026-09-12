from __future__ import annotations

from .shhs import SHHSDataset


class HospitalDataset(SHHSDataset):
    """Evaluation-only dataset with the same preprocessed NPZ contract as SHHS."""
