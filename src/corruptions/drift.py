from __future__ import annotations

import torch


def fft_bandpass(x: torch.Tensor, fs: float = 100.0, low: float = 0.3, high: float = 35.0) -> torch.Tensor:
    freqs = torch.fft.rfftfreq(x.shape[-1], d=1.0 / fs).to(x.device)
    mask = ((freqs >= low) & (freqs <= high)).to(x.dtype)
    spectrum = torch.fft.rfft(x, dim=-1)
    filtered = torch.fft.irfft(spectrum * mask.view(*([1] * (x.ndim - 1)), -1), n=x.shape[-1], dim=-1)
    return filtered.to(dtype=x.dtype)


def add_baseline_drift(
    x: torch.Tensor,
    fs: float = 100.0,
    f=None,
    amplitude=None,
    post_bandpass: bool = True,
    bandpass_low: float = 0.3,
    bandpass_high: float = 35.0,
) -> torch.Tensor:
    b = x.shape[0]
    device = x.device
    if f is None:
        f = torch.empty(b, device=device).uniform_(0.05, 0.5)
    if amplitude is None:
        sigma = x.std(dim=-1).squeeze(1).clamp_min(1e-6)
        amplitude = torch.empty(b, device=device).uniform_(0.1, 1.5) * sigma
    phase = torch.empty(b, device=device).uniform_(0, 2 * torch.pi)
    t = torch.arange(x.shape[-1], device=device, dtype=x.dtype).view(1, 1, -1) / fs
    drift = amplitude.view(-1, 1, 1) * torch.sin(2 * torch.pi * f.view(-1, 1, 1) * t + phase.view(-1, 1, 1))
    corrupted = x + drift
    if not post_bandpass:
        return corrupted
    return fft_bandpass(corrupted, fs=fs, low=bandpass_low, high=bandpass_high)
