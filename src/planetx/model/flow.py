"""Conditional density estimator q_phi(theta | z) over HPX parameters.

Thin wrapper around zuko's Neural Spline Flow so the rest of the codebase
depends on a stable (log_prob, sample) interface rather than zuko's API
directly. This is the piece that makes training an NPE loss rather than a
point-regression loss -- see README.md, "Point estimate vs distribution".
"""

from __future__ import annotations

import torch
import zuko
from torch import nn


class ConditionalPosteriorFlow(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        context_dim: int,
        hidden_features: tuple[int, ...] = (128, 128),
        transforms: int = 5,
    ):
        super().__init__()
        self.flow = zuko.flows.NSF(
            features=theta_dim,
            context=context_dim,
            hidden_features=list(hidden_features),
            transforms=transforms,
        )

    def log_prob(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.flow(context).log_prob(theta)

    def sample(self, context: torch.Tensor, n: int = 2000) -> torch.Tensor:
        return self.flow(context).sample((n,))
