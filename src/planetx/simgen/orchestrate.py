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
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from planetx.config import PriorConfig
from planetx.simgen.priors import sample_nuisance, sample_theta, theta_to_vector
from planetx.simgen.selection import SelectionFunction, SimpleSelectionFunction
from planetx.simgen.worker import run_one

logger = logging.getLogger(__name__)

# Per-WORKER-PROCESS completed-simulation counter (not shared across
# processes -- each ProcessPoolExecutor worker gets its own copy of module
# globals, so this correctly accumulates "how many sims has THIS worker
# process run" across the multiple tasks a pool worker is reused for).
_worker_completed = 0


def _init_worker_logging(level: int) -> None:
    """ProcessPoolExecutor initializer: on macOS/Windows (multiprocessing's
    'spawn' start method), a worker process is a fresh interpreter that does
    NOT inherit the parent's already-configured logging handlers -- without
    this, _run_and_select's per-worker progress logging would silently be a
    no-op there (it would appear to work by accident on Linux's 'fork'
    default, which does inherit parent state, masking the problem)."""
    logging.basicConfig(level=level, format="%(message)s")


def _run_and_select(args: tuple[PriorConfig, SelectionFunction, int]) -> dict:
    global _worker_completed
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

    _worker_completed += 1
    logger.info("worker pid=%d: completed sim seed=%d (%d done by this worker)",
                os.getpid(), seed, _worker_completed)

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
    seed0: int = -1,
    selection_fn: SelectionFunction | None = None,
    use_tqdm: bool = False,
) -> None:
    """Write shard_00000.parquet, shard_00001.parquet, ... to out_dir.

    Each row is one (x, theta) training pair: theta (7,), features (N, 11)
    ragged, survey_meta (3,), n_objects. Downstream training reads these
    shards with a streaming loader rather than concatenating them in memory.

    seed0: base seed -- simulation k in [0, n_simulations) uses seed
        seed0+k. Pass any non-negative integer for a fully reproducible run
        (two calls with the same prior_config/n_simulations/seed0 produce
        bit-identical output). seed0<0 (the default, -1) instead derives
        seed0 from the machine's current timestamp (nanosecond resolution,
        so back-to-back runs still get distinct seeds), so repeated runs of
        the same command generate DIFFERENT datasets -- the resolved seed0
        is logged so a particular "random" run can still be reproduced
        later by passing it back explicitly.

    Progress: each worker process logs its own completions as they happen
    (see _run_and_select) -- genuine per-worker visibility, not just an
    aggregate count, since with simulations this long-running (hours to
    days each) knowing THIS worker is stuck/progressing matters. The main
    process additionally logs an aggregate summary roughly every 1% of each
    shard (or drives a tqdm bar instead if use_tqdm=True and tqdm is
    installed -- tqdm's carriage-return-based bar renders nicely in an
    interactive terminal but is noisy in a redirected/Cloud-Logging-captured
    log file, which is why it's opt-in rather than the default for this
    project's actual cluster-batch use case).
    """
    if seed0 < 0:
        seed0 = time.time_ns()
        logger.info("seed0 not specified (<0): using timestamp-derived seed0=%d "
                    "(non-reproducible run -- pass this value back via --seed0 to reproduce it)", seed0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_fn = selection_fn or SimpleSelectionFunction()

    seeds = list(range(seed0, seed0 + n_simulations))
    n_shards = (n_simulations + shard_size - 1) // shard_size

    tqdm_cls = None
    if use_tqdm:
        try:
            from tqdm import tqdm as tqdm_cls
        except ImportError:
            logger.warning("use_tqdm=True but tqdm isn't installed -- falling back to 1%%-interval logging")

    root_level = logging.getLogger().getEffectiveLevel()
    with cf.ProcessPoolExecutor(
        max_workers=n_workers, initializer=_init_worker_logging, initargs=(root_level,)
    ) as ex:
        for shard_idx in range(n_shards):
            shard_seeds = seeds[shard_idx * shard_size : (shard_idx + 1) * shard_size]
            args = [(prior_config, selection_fn, s) for s in shard_seeds]
            n_shard = len(args)
            log_every = max(1, n_shard // 100)  # ~every 1%

            futures = [ex.submit(_run_and_select, a) for a in args]
            completed = cf.as_completed(futures)
            if tqdm_cls is not None:
                completed = tqdm_cls(completed, total=n_shard, desc=f"shard {shard_idx + 1}/{n_shards}")

            records = []
            for i, future in enumerate(completed, start=1):
                records.append(future.result())
                if tqdm_cls is None and (i % log_every == 0 or i == n_shard):
                    logger.info("shard %d/%d: %d/%d sims done (%.0f%%)",
                                shard_idx + 1, n_shards, i, n_shard, 100 * i / n_shard)

            df = pd.DataFrame.from_records(records)
            df.to_parquet(out_dir / f"shard_{shard_idx:05d}.parquet")
            logger.info("wrote shard %d/%d (%d sims) -> %s", shard_idx + 1, n_shards, len(records), out_dir)
