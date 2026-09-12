from __future__ import annotations

import torch
from torch import nn

from copy import deepcopy

from third_party.attnsleep_official.model import (
    AttnSleep as OfficialAttnSleep,
    EncoderLayer,
    MRCNN_SHHS,
    MultiHeadedAttention,
    PositionwiseFeedForward,
    TCE,
)


class OfficialSHHSAttnSleep(nn.Module):
    """Official AttnSleep SHHS variant for native 125 Hz, 3750-sample epochs."""

    def __init__(self):
        super().__init__()
        n_layers = 2
        d_model = 100
        d_ff = 120
        h = 5
        dropout = 0.1
        num_classes = 5
        afr_reduced_cnn_size = 30
        self.mrcnn = MRCNN_SHHS(afr_reduced_cnn_size)
        attn = MultiHeadedAttention(h, d_model, afr_reduced_cnn_size)
        ff = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.tce = TCE(EncoderLayer(d_model, deepcopy(attn), deepcopy(ff), afr_reduced_cnn_size, dropout), n_layers)
        self.fc = nn.Linear(d_model * afr_reduced_cnn_size, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_feat = self.mrcnn(x)
        encoded_features = self.tce(x_feat)
        encoded_features = encoded_features.contiguous().view(encoded_features.shape[0], -1)
        return self.fc(encoded_features)


class AttnSleep(nn.Module):
    """Wrapper around the official AttnSleep implementation.

    Upstream: https://github.com/emadeldeen24/AttnSleep

    The official default architecture accepts 30 s single-channel epochs shaped
    [B, 1, 3000] and outputs five sleep-stage logits, matching this project
    after resampling to 100 Hz.
    """

    def __init__(self, num_classes: int = 5, architecture: str = "default"):
        super().__init__()
        if num_classes != 5:
            raise ValueError("The official AttnSleep implementation is fixed to 5 sleep-stage classes.")
        if architecture == "official_shhs":
            self.model = OfficialSHHSAttnSleep()
        elif architecture == "default":
            self.model = OfficialAttnSleep()
        else:
            raise ValueError(f"Unknown AttnSleep architecture: {architecture}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
