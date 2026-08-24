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


@pytest.mark.parametrize("disk_backend", ["secular", "hybrid"])
def test_run_one_smoke_alternate_backends(disk_backend):
    """disk_backend="secular"/"hybrid" are opt-in, exploratory (see
    planetx.simgen.secular's module docstring for the accuracy caveats) --
    this only checks the wiring produces well-formed output, not physical
    realism. n_test_particles=30 (vs. 5 for the "rebound" smoke test) so a
    "hybrid" run has a reasonable chance of exercising its resonant-N-body
    branch too, not just the closed-form one.
    """
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(2)
    theta = sample_theta(cfg, rng)
    nuisance = sample_nuisance(cfg, rng)

    result = run_one(
        theta=theta, nuisance=nuisance, n_test_particles=30,
        integration_years=1.0e4, dt_years=5.0, use_gr=False, seed=2,
        disk_backend=disk_backend,
    )

    assert result["theta"] == theta
    assert set(result["hpx_final"]) == {"mass", "a", "e", "i", "Omega", "omega", "M"}
    assert isinstance(result["tnos"], list)
    for tno in result["tnos"]:
        assert set(tno) == {"a", "e", "i", "Omega", "omega", "M"}
        assert tno["a"] > 0
        assert 0.0 <= tno["e"] < 1.0


def test_run_one_rejects_unknown_disk_backend():
    cfg = PriorConfig.from_yaml(PRIOR_PATH)
    rng = np.random.default_rng(3)
    theta = sample_theta(cfg, rng)
    nuisance = sample_nuisance(cfg, rng)

    with pytest.raises(ValueError, match="disk_backend"):
        run_one(
            theta=theta, nuisance=nuisance, n_test_particles=5,
            integration_years=1.0e4, dt_years=5.0, use_gr=False, seed=3,
            disk_backend="nonexistent",
        )
