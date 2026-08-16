"""Sample theta (HPX params) and nuisance latents from a PriorConfig."""

from __future__ import annotations

import numpy as np

from planetx.config import PriorConfig
from planetx.constants import THETA_KEYS


def sample_theta(prior_config: PriorConfig, rng: np.random.Generator) -> dict[str, float]:
    """Sample HPX's own orbital elements + mass. This is the inference target."""
    return {k: prior_config.theta_hpx[k].sample(rng) for k in THETA_KEYS}


def sample_nuisance(prior_config: PriorConfig, rng: np.random.Generator) -> dict[str, float]:
    """Sample latents that are marginalized over, never used as a label.

    Unlike theta, these are never fed back to the network as a target -- they
    exist purely so the simulator's (theta -> x) mapping is trained under
    realistic uncertainty about the primordial disk, rather than one fixed
    assumption.
    """
    return {k: v.sample(rng) for k, v in prior_config.nuisance.items()}


def theta_to_vector(theta: dict[str, float]) -> np.ndarray:
    return np.array([theta[k] for k in THETA_KEYS], dtype=np.float64)


def vector_to_theta(vec: np.ndarray) -> dict[str, float]:
    return {k: float(v) for k, v in zip(THETA_KEYS, vec)}
