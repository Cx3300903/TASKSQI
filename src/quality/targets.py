from __future__ import annotations

import numpy as np
import torch


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    m = 0.5 * (p + q)
    kl_pm = (p * (torch.log(p.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
    kl_qm = (q * (torch.log(q.clamp_min(1e-12)) - torch.log(m.clamp_min(1e-12)))).sum(dim=1)
    return 0.5 * (kl_pm + kl_qm)


def severity_to_quality(corruption_name: str, params: dict[str, torch.Tensor | float], batch_size: int, device) -> torch.Tensor:
    if corruption_name == "clean":
        return torch.ones(batch_size, device=device)
    if corruption_name == "noise":
        snr = params["snr_db"] if torch.is_tensor(params["snr_db"]) else torch.tensor(params["snr_db"], device=device)
        severity = (20.0 - snr.to(device)) / 25.0
    elif corruption_name == "drift":
        amp = params["amplitude_mult"] if torch.is_tensor(params["amplitude_mult"]) else torch.tensor(params["amplitude_mult"], device=device)
        severity = (amp.to(device) - 0.1) / 1.4
    elif corruption_name == "dropout":
        drop = params["drop_ratio"] if torch.is_tensor(params["drop_ratio"]) else torch.tensor(params["drop_ratio"], device=device)
        severity = (drop.to(device) - 0.05) / 0.45
    else:
        raise ValueError(f"Unsupported training corruption: {corruption_name}")
    return (1.0 - severity.clamp(0, 1)).view(-1)


def compute_quality_target(
    target_type: str,
    clean_prob: torch.Tensor,
    corrupt_prob: torch.Tensor,
    y: torch.Tensor,
    clean_pred_correct: torch.Tensor | None = None,
    corruption_name: str = "clean",
    corruption_params: dict | None = None,
) -> torch.Tensor:
    if target_type == "retention":
        idx = torch.arange(y.shape[0], device=y.device)
        return torch.minimum(
            torch.ones_like(y, dtype=torch.float32),
            (corrupt_prob[idx, y] + 1e-6) / (clean_prob[idx, y] + 1e-6),
        )
    if target_type == "binary_flip":
        return (corrupt_prob.argmax(dim=1) == y).float()
    if target_type == "js":
        return 1.0 - js_divergence(clean_prob, corrupt_prob) / np.log(2.0)
    if target_type == "severity":
        if corruption_params is None:
            corruption_params = {}
        return severity_to_quality(corruption_name, corruption_params, y.shape[0], y.device)
    raise ValueError(f"Unknown target_type: {target_type}")
