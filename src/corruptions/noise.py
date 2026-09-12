from __future__ import annotations

import torch


def add_gaussian_noise(x: torch.Tensor, snr_db: torch.Tensor | float) -> torch.Tensor:
    if not torch.is_tensor(snr_db):
        snr_db = torch.full((x.shape[0],), float(snr_db), device=x.device)
    snr_db = snr_db.to(x.device).view(-1, 1, 1)
    power = x.pow(2).mean(dim=-1, keepdim=True).clamp_min(1e-8)
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = torch.randn_like(x) * noise_power.sqrt()
    return x + noise
