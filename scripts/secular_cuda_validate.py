"""Validates secular_cuda.py's real CUDA port against planetx.simgen.secular
itself (the actual production module, not a copy) -- REQUIRES A GPU
(numba.cuda.is_available() must be True); this project's own dev
environment has none, so this script cannot run locally (same constraint
as blockpersim_cuda_validate.py). Developed and run cell-by-cell in a
connected Colab notebook against an A100; see secular_cuda.py's module
docstring for the exact numbers and the two real bugs that run caught.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from secular_cuda import run_secular_ensemble_cuda  # noqa: E402
from planetx.simgen.secular import (  # noqa: E402
    EigenSystem, GM_SUN, massive_body_elements, propagate_disk,
)
from planetx.simgen.worker import _sample_primordial_disk  # noqa: E402

T_YEARS = 2.0e6  # well within secular.py's own validated range (up to 2e7 yr)
N_TEST = 500  # matches configs/prior.yaml's n_test_particles

nuisance = {"disk_inner_edge": 35.0, "disk_outer_edge": 80.0, "disk_e_scale": 0.08, "disk_i_scale": 5.0}
theta = {"mass": 5.0, "a": 400.0, "e": 0.3, "i": 20.0, "Omega": 90.0, "omega": 45.0, "M": 0.0}

print(f"=== Single sim: n_test={N_TEST}, t={T_YEARS:.0e} yr ===")
rng = np.random.default_rng(0)
disk = _sample_primordial_disk(nuisance, N_TEST, rng)

t0 = time.perf_counter()
eigsys = EigenSystem(massive_body_elements(theta))
cpu_results = propagate_disk(eigsys, disk, range(N_TEST), T_YEARS)
print(f"CPU (secular.py):  {len(cpu_results)}/{N_TEST} valid, {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
gpu_results = run_secular_ensemble_cuda([theta], [disk], T_YEARS, GM_SUN, massive_body_elements)[0]
print(f"GPU (secular_cuda): {len(gpu_results)}/{N_TEST} valid, {time.perf_counter()-t0:.2f}s")

max_err = 0.0
for c, g in zip(cpu_results, gpu_results):
    for key in ("e", "i", "Omega", "omega", "M"):
        max_err = max(max_err, abs(c[key] - g[key]))
print(f"max abs error across all fields: {max_err:.3e}")
print("Expect ~1e-10 to 1e-9: pure double-precision roundoff between the identical")
print("algorithm on CPU vs GPU, not a formula divergence.")

print(f"\n=== Ensemble: 32 independent simulations, n_test={N_TEST} ===")
theta_list = []
disk_list = []
for k in range(32):
    rk = np.random.default_rng(100 + k)
    th = {"mass": rk.uniform(1, 10), "a": rk.uniform(300, 500), "e": 0.3, "i": 20.0, "Omega": 90.0, "omega": 45.0, "M": 0.0}
    theta_list.append(th)
    disk_list.append(_sample_primordial_disk(nuisance, N_TEST, rk))

t0 = time.perf_counter()
gpu_ensemble = run_secular_ensemble_cuda(theta_list, disk_list, T_YEARS, GM_SUN, massive_body_elements)
dt_gpu = time.perf_counter() - t0
print(f"GPU: {len(theta_list)} sims in {dt_gpu:.2f}s ({dt_gpu/len(theta_list)*1000:.1f} ms/sim, "
      f"{len(theta_list)/dt_gpu:.2f} sims/s)")
print("Note: unlike blockpersim's N-body backend, this cost is INDEPENDENT of")
print("integration duration (T_YEARS) -- no time-stepping loop, a single closed-form")
print("evaluation per particle. Re-running at production's 4.5e9 yr target costs the same.")
