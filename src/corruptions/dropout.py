from __future__ import annotations

import torch


def apply_signal_dropout(x: torch.Tensor, drop_ratio=None) -> torch.Tensor:
    b, _, n = x.shape
    device = x.device
    if drop_ratio is None:
        drop_ratio = torch.empty(b, device=device).uniform_(0.05, 0.50)
    out = x.clone()
    lengths = (drop_ratio * n).long().clamp(1, n)
    for i, length in enumerate(lengths.tolist()):
        start = torch.randint(0, n - length + 1, (1,), device=device).item()
        out[i, :, start : start + length] = 0.0
    return out
