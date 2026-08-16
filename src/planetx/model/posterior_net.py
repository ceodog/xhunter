"""PosteriorNet: population snapshot -> q_phi(theta | x).

The NPE loss is a plain negative log-likelihood of the true theta under this
conditional density. That loss -- not the transformer architecture -- is
what gives the trained network a calibrated posterior instead of a point
estimate; an MSE loss on the same architecture would converge to E[theta|x]
and lose multimodality/uncertainty entirely.
"""

from __future__ import annotations

import torch
from torch import nn

from planetx.model.encoder import SetTransformerEncoder
from planetx.model.flow import ConditionalPosteriorFlow


class PosteriorNet(nn.Module):
    def __init__(
        self,
        object_feature_dim: int,
        theta_dim: int,
        survey_meta_dim: int = 3,
        d_model: int = 64,
        n_layers: int = 3,
        n_heads: int = 4,
        n_seeds: int = 4,
    ):
        super().__init__()
        self.set_encoder = SetTransformerEncoder(
            in_dim=object_feature_dim, d_model=d_model, n_layers=n_layers,
            n_heads=n_heads, n_seeds=n_seeds,
        )
        self.survey_meta_encoder = nn.Linear(survey_meta_dim, self.set_encoder.out_dim)
        self.flow = ConditionalPosteriorFlow(
            theta_dim=theta_dim, context_dim=self.set_encoder.out_dim
        )

    def embed(
        self, feats: torch.Tensor, key_padding_mask: torch.Tensor, survey_meta: torch.Tensor
    ) -> torch.Tensor:
        z = self.set_encoder(feats, key_padding_mask)
        return z + self.survey_meta_encoder(survey_meta)

    def loss(
        self,
        feats: torch.Tensor,
        key_padding_mask: torch.Tensor,
        survey_meta: torch.Tensor,
        theta_true: torch.Tensor,
    ) -> torch.Tensor:
        z = self.embed(feats, key_padding_mask, survey_meta)
        return -self.flow.log_prob(theta_true, z).mean()

    @torch.no_grad()
    def posterior_samples(
        self,
        feats: torch.Tensor,
        key_padding_mask: torch.Tensor,
        survey_meta: torch.Tensor,
        n: int = 2000,
    ) -> torch.Tensor:
        z = self.embed(feats, key_padding_mask, survey_meta)
        return self.flow.sample(z, n=n)
