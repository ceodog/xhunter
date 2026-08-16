import numpy as np

from planetx.config import PriorConfig
from planetx.constants import THETA_KEYS
from planetx.simgen.priors import (
    sample_nuisance,
    sample_theta,
    theta_to_vector,
    vector_to_theta,
)

PRIOR_PATH = "configs/prior.yaml"


def test_prior_config_loads():
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    assert set(cfg.theta_hpx) == set(THETA_KEYS)
    assert cfg.simulation.n_test_particles > 0


def test_sample_theta_within_bounds():
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(0)
    for _ in range(50):
        theta = sample_theta(cfg, rng)
        assert set(theta) == set(THETA_KEYS)
        for k, v in theta.items():
            dist = cfg.theta_hpx[k]
            assert dist.low <= v <= dist.high


def test_sample_nuisance_within_bounds():
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(1)
    nuisance = sample_nuisance(cfg, rng)
    assert set(nuisance) == set(cfg.nuisance)
    for k, v in nuisance.items():
        dist = cfg.nuisance[k]
        assert dist.low <= v <= dist.high


def test_theta_vector_roundtrip():
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(2)
    theta = sample_theta(cfg, rng)
    vec = theta_to_vector(theta)
    assert vec.shape == (len(THETA_KEYS),)
    back = vector_to_theta(vec)
    for k in THETA_KEYS:
        assert back[k] == theta[k]
