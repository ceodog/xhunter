"""Fan out many (simulate -> select) runs into a sharded Parquet training set.

Each simulation is independent (embarrassingly parallel); this module uses a
local process pool for a single machine. For cluster-scale generation, swap
the ProcessPoolExecutor loop for a Slurm/cloud-batch job array that each call
`_run_and_select` once per task -- the function signature is deliberately
kept picklable and side-effect-free to make that swap mechanical.
"""

from __future__ import annotations

import concurrent.futures as cf
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from planetx.config import PriorConfig
from planetx.simgen.priors import sample_nuisance, sample_theta, theta_to_vector
from planetx.simgen.selection import SelectionFunction, SimpleSelectionFunction
from planetx.simgen.worker import run_one

logger = logging.getLogger(__name__)


def _run_and_select(args: tuple[PriorConfig, SelectionFunction, int]) -> dict:
    prior_config, selection_fn, seed = args
    rng = np.random.default_rng(seed)

    theta = sample_theta(prior_config, rng)
    nuisance = sample_nuisance(prior_config, rng)
    sim_cfg = prior_config.simulation

    result = run_one(
        theta=theta,
        nuisance=nuisance,
        n_test_particles=sim_cfg.n_test_particles,
        integration_years=sim_cfg.integration_years,
        dt_years=sim_cfg.dt_years,
        use_gr=sim_cfg.use_gr,
        seed=seed,
        disk_backend=sim_cfg.disk_backend,
    )
    fset = selection_fn.apply(result["tnos"], rng)

    return {
        "seed": seed,
        # hpx_final, not the pre-integration `theta` sampled above: the label
        # must be HPX's state at the SAME final epoch as x (result["tnos"]),
        # since real inference wants HPX's *current* parameters (to know
        # where to point a telescope), not its state at some assumed
        # primordial starting epoch -- see README.md, "Single-epoch
        # snapshots, not time series".
        "theta": theta_to_vector(result["hpx_final"]).tolist(),
        "features": fset.features.tolist(),
        "survey_meta": fset.survey_meta.tolist(),
        "n_objects": fset.n_objects,
    }


def generate_dataset(
    prior_config: PriorConfig,
    out_dir: str | Path,
    n_simulations: int,
    shard_size: int = 500,
    n_workers: int = 4,
    seed0: int = 0,
    selection_fn: SelectionFunction | None = None,
) -> None:
    """Write shard_00000.parquet, shard_00001.parquet, ... to out_dir.

    Each row is one (x, theta) training pair: theta (7,), features (N, 11)
    ragged, survey_meta (3,), n_objects. Downstream training reads these
    shards with a streaming loader rather than concatenating them in memory.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_fn = selection_fn or SimpleSelectionFunction()

    seeds = list(range(seed0, seed0 + n_simulations))
    n_shards = (n_simulations + shard_size - 1) // shard_size

    with cf.ProcessPoolExecutor(max_workers=n_workers) as ex:
        for shard_idx in range(n_shards):
            shard_seeds = seeds[shard_idx * shard_size : (shard_idx + 1) * shard_size]
            args = [(prior_config, selection_fn, s) for s in shard_seeds]
            records = list(ex.map(_run_and_select, args))
            df = pd.DataFrame.from_records(records)
            df.to_parquet(out_dir / f"shard_{shard_idx:05d}.parquet")
            logger.info("wrote shard %d/%d (%d sims) -> %s", shard_idx + 1, n_shards, len(records), out_dir)
