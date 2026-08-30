"""GPU-batched counterpart to orchestrate.generate_dataset. Fans out many
(simulate -> select) runs the same way, but dispatches whole batches to a
single GPU kernel launch (gpu_nbody.py / gpu_secular.py) instead of one
worker process per simulation -- the GPU throughput these backends measure
only materializes when many simulations share one launch (see
gpu_nbody.py/gpu_secular.py's module docstrings for the validation-session
benchmarks this relies on).

REQUIRES A REAL NVIDIA GPU and the optional `gpu` dependency group (`uv
sync --extra gpu`) -- this module is never imported by the default CPU
pipeline (planetx.cli only imports it inside the `simgen run-gpu` command).

Two gpu_mode values:
  "full_nbody" -- gpu_nbody.run_ensemble_gpu_nbody propagates massive
      bodies AND all test particles via real N-body on GPU. Physically
      equivalent to disk_backend="rebound", just GPU-accelerated -- no
      approximation, no accuracy caveats beyond REBOUND's own. Measured
      ~235 days for 10000 sims at this project's production 4.5e9 yr /
      dt=0.5 / n_test_particles=500 (A100, validation session).
  "hybrid_secular" -- gpu_nbody.run_ensemble_gpu_massive_only propagates
      ONLY the massive bodies (needed regardless, since HPX's own final
      state is the training label); the disk is propagated separately by
      gpu_secular.py's closed-form kernel. ~16.3 days for the same 10000
      sims -- but inherits secular.py's full accuracy caveat list
      (validated only to 2e7 yr, fails on resonant particles, no ejection
      modeling; see secular.py's module docstring, load-bearing, not a
      footnote). NOT a validated replacement for disk_backend="rebound";
      choose this only if you can accept those caveats for your dataset.

STATUS: validated end-to-end on a Colab A100 (2026-08-25) -- generate_dataset_gpu
was run for both gpu_mode values at dev scale, the resulting shard parquet files
were loaded via the real ShardedSimDataset and trained on via the real
model.train.train() with no changes needed to either. See gpu_nbody.py's module
docstring for the underlying kernel-vs-REBOUND validation numbers this relies on.
One real bug was found and fixed in the process (unrelated to this module itself):
model.train.ShardedSimDataset.__getitem__ raised on any row with exactly one
detected object (a numpy ragged-array shape-inference ambiguity) -- this affects
the CPU orchestrate.py path equally, just rarely triggered at production's larger
disk sizes; fixed in model/train.py.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import numpy as np
import pandas as pd

from planetx.config import PriorConfig
from planetx.simgen.priors import sample_nuisance, sample_theta, theta_to_vector
from planetx.simgen.selection import SelectionFunction, SimpleSelectionFunction

logger = logging.getLogger(__name__)


def _run_batch_full_nbody(prior_config: PriorConfig, theta_list: list[dict],
                           nuisance_list: list[dict], seeds: list[int]) -> list[dict]:
    from planetx.simgen.gpu_nbody import run_ensemble_gpu_nbody

    sim_cfg = prior_config.simulation
    return run_ensemble_gpu_nbody(
        theta_list=theta_list, nuisance_list=nuisance_list, seed_list=seeds,
        n_test_particles=sim_cfg.n_test_particles, dt_years=sim_cfg.dt_years,
        integration_years=sim_cfg.integration_years,
    )


def _run_batch_hybrid_secular(prior_config: PriorConfig, theta_list: list[dict],
                               nuisance_list: list[dict], seeds: list[int]) -> list[dict]:
    from planetx.simgen.gpu_nbody import run_ensemble_gpu_massive_only
    from planetx.simgen.gpu_secular import run_secular_ensemble_cuda
    from planetx.simgen.worker import _sample_primordial_disk

    sim_cfg = prior_config.simulation
    hpx_finals = run_ensemble_gpu_massive_only(
        theta_list=theta_list, seed_list=seeds,
        dt_years=sim_cfg.dt_years, integration_years=sim_cfg.integration_years,
    )

    disk_list = []
    for nuisance, seed in zip(nuisance_list, seeds):
        # Independent stream from the massive-body draw (run_ensemble_gpu_massive_only
        # seeds its own REBOUND-based sampling from `seed` directly) -- this path's
        # disk physics (closed-form secular) is unrelated to worker.run_one's REBOUND
        # RNG sequence, so there's no reason to try to replicate it bit-for-bit; just
        # needs a distinct, deterministic seed per simulation.
        rng = np.random.default_rng(seed + 500_000_000)
        disk_list.append(_sample_primordial_disk(nuisance, sim_cfg.n_test_particles, rng))

    tnos_lists = run_secular_ensemble_cuda(theta_list, disk_list, sim_cfg.integration_years)

    return [
        {"theta": dict(theta_list[i]), "hpx_final": hpx_finals[i], "tnos": tnos_lists[i]}
        for i in range(len(theta_list))
    ]


def generate_dataset_gpu(
    prior_config: PriorConfig,
    out_dir: str | Path,
    n_simulations: int,
    gpu_mode: str = "full_nbody",
    shard_size: int = 500,
    seed0: int = -1,
    selection_fn: SelectionFunction | None = None,
) -> None:
    """Write shard_00000.parquet, shard_00001.parquet, ... to out_dir --
    same schema as orchestrate.generate_dataset, so downstream
    (ShardedSimDataset, model.train) needs no changes. shard_size doubles
    as the GPU batch size here (one shard = one batch = one set of kernel
    launches), unlike the CPU path where it's purely a checkpoint/write
    granularity independent of worker-process concurrency.

    seed0: same convention as orchestrate.generate_dataset -- pass a
    non-negative integer for a fully reproducible run, or leave it at the
    default -1 to derive it from the machine's timestamp (a different
    dataset each invocation; the resolved value is logged).
    """
    if gpu_mode not in ("full_nbody", "hybrid_secular"):
        raise ValueError(f"Unknown gpu_mode: {gpu_mode!r} (expected 'full_nbody' or 'hybrid_secular')")

    if seed0 < 0:
        seed0 = time.time_ns()
        logger.info("seed0 not specified (<0): using timestamp-derived seed0=%d "
                    "(non-reproducible run -- pass this value back via --seed0 to reproduce it)", seed0)

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    selection_fn = selection_fn or SimpleSelectionFunction()
    run_batch = _run_batch_full_nbody if gpu_mode == "full_nbody" else _run_batch_hybrid_secular

    seeds_all = list(range(seed0, seed0 + n_simulations))
    n_shards = (n_simulations + shard_size - 1) // shard_size

    for shard_idx in range(n_shards):
        shard_seeds = seeds_all[shard_idx * shard_size: (shard_idx + 1) * shard_size]
        theta_list, nuisance_list = [], []
        for seed in shard_seeds:
            rng = np.random.default_rng(seed)
            theta_list.append(sample_theta(prior_config, rng))
            nuisance_list.append(sample_nuisance(prior_config, rng))

        batch_results = run_batch(prior_config, theta_list, nuisance_list, shard_seeds)

        records = []
        for seed, result in zip(shard_seeds, batch_results):
            rng = np.random.default_rng(seed + 1_000_000_000)  # separate stream from the physics sampling
            fset = selection_fn.apply(result["tnos"], rng)
            records.append({
                "seed": seed,
                "theta": theta_to_vector(result["hpx_final"]).tolist(),
                "features": fset.features.tolist(),
                "survey_meta": fset.survey_meta.tolist(),
                "n_objects": fset.n_objects,
            })

        df = pd.DataFrame.from_records(records)
        df.to_parquet(out_dir / f"shard_{shard_idx:05d}.parquet")
        logger.info("wrote shard %d/%d (%d sims, gpu_mode=%s) -> %s",
                    shard_idx + 1, n_shards, len(records), gpu_mode, out_dir)
