import numpy as np
import pytest

pytest.importorskip("rebound")

from planetx.config import PriorConfig  # noqa: E402
from planetx.simgen.priors import sample_nuisance, sample_theta  # noqa: E402
from planetx.simgen.selection import SimpleSelectionFunction  # noqa: E402
from planetx.simgen.worker import run_one  # noqa: E402

PRIOR_PATH = "configs/prior.yaml"


def test_run_one_smoke():
    """Tiny/short integration: checks the REBOUND wiring end to end, not physical realism."""
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(0)
    theta = sample_theta(cfg, rng)
    nuisance = sample_nuisance(cfg, rng)

    result = run_one(
        theta=theta, nuisance=nuisance, n_test_particles=5,
        integration_years=1.0e4, dt_years=5.0, use_gr=False, seed=0,
    )

    assert result["theta"] == theta
    assert set(result["hpx_final"]) == {"mass", "a", "e", "i", "Omega", "omega", "M"}
    assert isinstance(result["tnos"], list)
    for tno in result["tnos"]:
        assert set(tno) == {"a", "e", "i", "Omega", "omega", "M"}
        assert tno["a"] > 0


def test_run_one_feeds_selection_function():
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(1)
    theta = sample_theta(cfg, rng)
    nuisance = sample_nuisance(cfg, rng)

    result = run_one(
        theta=theta, nuisance=nuisance, n_test_particles=10,
        integration_years=1.0e4, dt_years=5.0, use_gr=False, seed=1,
    )
    fset = SimpleSelectionFunction(sky_fraction=1.0, limiting_mag=30.0).apply(result["tnos"], rng)
    assert fset.n_objects <= len(result["tnos"])
