from __future__ import annotations

import torch

from .drift import add_baseline_drift
from .dropout import apply_signal_dropout
from .noise import add_gaussian_noise


def sample_training_corruption(x: torch.Tensor, clean_probability: float = 0.2):
    b = x.shape[0]
    device = x.device
    draw = torch.rand(1, device=device).item()
    if draw < clean_probability:
        return x.clone(), "clean", {}
    choice = torch.randint(0, 3, (1,), device=device).item()
    if choice == 0:
        snr = torch.empty(b, device=device).uniform_(-5.0, 20.0)
        return add_gaussian_noise(x, snr), "noise", {"snr_db": snr}
    if choice == 1:
        f = torch.empty(b, device=device).uniform_(0.05, 0.5)
        amp_mult = torch.empty(b, device=device).uniform_(0.1, 1.5)
        sigma = x.std(dim=-1).squeeze(1).clamp_min(1e-6)
        return add_baseline_drift(x, f=f, amplitude=amp_mult * sigma), "drift", {"f": f, "amplitude_mult": amp_mult}
    drop = torch.empty(b, device=device).uniform_(0.05, 0.50)
    return apply_signal_dropout(x, drop), "dropout", {"drop_ratio": drop}
