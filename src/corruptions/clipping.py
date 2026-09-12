from __future__ import annotations

import torch


def apply_clipping(x: torch.Tensor, k: float) -> torch.Tensor:
    sigma = x.std(dim=-1, keepdim=True).clamp_min(1e-6)
    limit = float(k) * sigma
    return torch.clamp(x, -limit, limit)
