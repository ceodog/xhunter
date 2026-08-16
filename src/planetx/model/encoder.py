"""Set Transformer over the population of TNOs/ETNOs.

Deliberately omits positional encoding: x is an unordered set, not a
sequence (see README.md, "x is a set, not a sequence"), so the network must
be permutation-invariant in the objects it attends over. The "transformer"
in this project attends across objects, not across time -- time was
collapsed to a single epoch upstream in planetx.simgen.worker.
"""

from __future__ import annotations

import torch
from torch import nn


class ObjectEncoder(nn.Module):
    """Per-object feature vector -> per-object embedding."""

    def __init__(self, in_dim: int, d_model: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, d_model), nn.GELU(),
        )

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        return self.net(feats)  # [B, N, in_dim] -> [B, N, d_model]


class SelfAttentionBlock(nn.Module):
    """A pre-norm transformer encoder layer, no positional encoding."""

    def __init__(self, d_model: int, n_heads: int, ff_mult: int = 4, dropout: float = 0.0):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult), nn.GELU(),
            nn.Linear(d_model * ff_mult, d_model),
        )

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        h_norm = self.norm1(h)
        attn_out, _ = self.attn(
            h_norm, h_norm, h_norm, key_padding_mask=key_padding_mask, need_weights=False
        )
        h = h + attn_out
        h = h + self.ff(self.norm2(h))
        return h


class PoolingByMultiheadAttention(nn.Module):
    """Permutation-invariant pooling: learned seed vectors attend over the set.

    Produces a fixed-size population embedding regardless of N, so the same
    weights handle a simulated survey yield of hundreds of objects and a real
    catalog of a few dozen.
    """

    def __init__(self, d_model: int, n_heads: int, n_seeds: int = 1):
        super().__init__()
        self.seeds = nn.Parameter(torch.randn(1, n_seeds, d_model) * 0.02)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.n_seeds = n_seeds
        self.d_model = d_model

    def forward(self, h: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        b = h.shape[0]
        seeds = self.seeds.expand(b, -1, -1)
        out, _ = self.attn(seeds, h, h, key_padding_mask=key_padding_mask, need_weights=False)
        out = self.norm(out)
        return out.reshape(b, self.n_seeds * self.d_model)


class SetTransformerEncoder(nn.Module):
    """feats [B, N, in_dim] (padded) + key_padding_mask [B, N] -> z [B, out_dim].

    A persistent, never-masked "null object" token is concatenated to every
    set. This keeps attention well-defined even when a simulated survey (or
    a real catalog subset) yields zero detected objects, and gives the
    network an explicit way to represent "no evidence" rather than relying
    on an all-padding row.
    """

    def __init__(
        self, in_dim: int, d_model: int = 64, n_layers: int = 3,
        n_heads: int = 4, n_seeds: int = 4,
    ):
        super().__init__()
        self.object_encoder = ObjectEncoder(in_dim, d_model)
        self.null_token = nn.Parameter(torch.randn(1, 1, d_model) * 0.02)
        self.sabs = nn.ModuleList(
            [SelfAttentionBlock(d_model, n_heads) for _ in range(n_layers)]
        )
        self.pma = PoolingByMultiheadAttention(d_model, n_heads, n_seeds)
        self.out_dim = d_model * n_seeds

    def forward(self, feats: torch.Tensor, key_padding_mask: torch.Tensor) -> torch.Tensor:
        b = feats.shape[0]
        h = self.object_encoder(feats)
        null = self.null_token.expand(b, -1, -1)
        h = torch.cat([null, h], dim=1)
        mask = torch.cat(
            [torch.zeros(b, 1, dtype=torch.bool, device=feats.device), key_padding_mask], dim=1
        )
        for sab in self.sabs:
            h = sab(h, mask)
        return self.pma(h, mask)
