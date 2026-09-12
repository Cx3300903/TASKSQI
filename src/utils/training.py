from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class EarlyStopping:
    patience: int
    mode: str = "min"
    best: float | None = None
    bad_epochs: int = 0

    def step(self, value: float) -> bool:
        improved = self.best is None
        if self.best is not None:
            improved = value < self.best if self.mode == "min" else value > self.best
        if improved:
            self.best = value
            self.bad_epochs = 0
            return False
        self.bad_epochs += 1
        return self.bad_epochs >= self.patience


def save_checkpoint(path, model, optimizer=None, epoch=0, metric=None, config=None):
    from pathlib import Path

    Path(path).parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": model.state_dict(),
        "epoch": epoch,
        "metric": metric,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    torch.save(payload, path)


def load_model_weights(model, path, map_location="cpu", strict=True):
    ckpt = torch.load(path, map_location=map_location)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    model.load_state_dict(state, strict=strict)
    return ckpt
