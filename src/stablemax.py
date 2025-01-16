# src/stablemax.py

import torch
import torch.nn as nn

class StableMax(nn.Module):
    """
    StableMax is an alternative to Softmax that helps mitigate numerical
    instabilities. It applies an elementwise transform s(x) and normalizes
    over the specified dimension to produce probabilities that sum to 1.

    s(x) = (x + 1) if x >= 0, or 1 / (1 - x) if x < 0.
    logits are clamped to a minimum value (clamp_min) to avoid extreme negatives.
    """
    def __init__(self, dim: int = -1, clamp_min: float = -10.0):
        super().__init__()
        self.dim = dim
        self.clamp_min = clamp_min

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        # Clamp extreme negative logits
        logits = torch.clamp(logits, min=self.clamp_min)
        s_logits = torch.where(logits >= 0, logits + 1, 1 / (1 - logits))
        s_sum = s_logits.sum(dim=self.dim, keepdim=True)
        return s_logits / (s_sum + 1e-9)


class LogStableMax(nn.Module):
    """
    LogStableMax is similar to StableMax but returns log-probabilities (log-space).
    This is useful if you want to use a loss function expecting log-probs
    (like NLLLoss).

    s(x) = (x + 1) if x >= 0, or 1 / (1 - x) if x < 0.
    logits are clamped to a minimum value (clamp_min) to avoid extreme negatives.
    """
    def __init__(self, dim: int = -1, clamp_min: float = -10.0):
        super().__init__()
        self.dim = dim
        self.clamp_min = clamp_min

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        logits = torch.clamp(logits, min=self.clamp_min)
        s_logits = torch.where(logits >= 0, logits + 1, 1 / (1 - logits))
        s_sum = s_logits.sum(dim=self.dim, keepdim=True)
        return torch.log(s_logits / (s_sum + 1e-9) + 1e-9)
