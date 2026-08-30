"""Validates blockpersim_cuda.py's real CUDA kernel against (1)
blockpersim_dev.py's already-REBOUND-validated NumPy reference and (2)
REBOUND directly -- same dev-scale settings as blockpersim_dev_validate.py
(DT=0.05, N_STEPS=40000, N_TEST=5), so the two validation passes are
directly comparable.

REQUIRES AN ACTUAL NVIDIA GPU (numba.cuda.is_available() must be True) --
this project's own dev environment has none (see blockpersim_dev.py's
module docstring), so this script cannot run locally. It was developed
and run cell-by-cell in a connected Colab notebook against a Tesla T4;
see blockpersim_cuda.py's module docstring for the exact numbers that run
produced. Run this script itself in that same kind of environment
(Colab, or any host with a CUDA GPU + numba's `cuda` extra installed) to
reproduce it end-to-end.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from blockpersim_dev import GM_SUN, G, run_ensemble, run_one_block  # noqa: E402
from blockpersim_cuda import run_ensemble_cuda  # noqa: E402
from blockpersim_dev_validate import build_state  # noqa: E402
from planetx.constants import EARTH_MASS_IN_MSUN, GIANT_PLANETS  # noqa: E402

GIANT_NAMES = ("jupiter", "saturn", "uranus", "neptune")

DT = 0.05
N_STEPS = 40000
N_TEST = 5

rng = np.random.default_rng(0)
theta_hpx = {"mass": 5.0, "a": 400.0, "e": 0.3, "i": 20.0, "Omega": 90.0, "omega": 45.0, "M": 0.0}
r_m0, v_m0, m_m, r_t0, v_t0, sim = build_state(theta_hpx, N_TEST, rng)

print(f"=== Single block: {N_STEPS} steps ({N_STEPS*DT:.0f} yr), {N_TEST} test particles ===")

t0 = time.perf_counter()
cpu_out = run_one_block(r_m0, v_m0, m_m, r_t0, v_t0, DT, N_STEPS)
print(f"NumPy reference (CPU):  {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
gpu_out = run_ensemble_cuda(
    [{"r_m0": r_m0, "v_m0": v_m0, "m_m": m_m, "r_t0": r_t0, "v_t0": v_t0}], DT, N_STEPS, G, GM_SUN
)[0]
print(f"CUDA kernel (GPU):      {time.perf_counter()-t0:.2f}s")

sim.integrator = "whfast"
sim.integrator.safe_mode = 0
sim.dt = DT
t0 = time.perf_counter()
sim.integrate(N_STEPS * DT)
print(f"REBOUND:                {time.perf_counter()-t0:.2f}s")

names = list(GIANT_NAMES) + ["hpx"]
sun = sim.particles[0]
r_sun = np.array([sun.x, sun.y, sun.z])

print("\n=== CUDA vs CPU (blockpersim_dev) reference, same algorithm ===")
max_err_gc = 0.0
for i, name in enumerate(names):
    err = np.linalg.norm(cpu_out["r_m"][i] - gpu_out["r_m"][i])
    max_err_gc = max(max_err_gc, err)
    print(f"  {name:8s}  |r_cpu - r_gpu| = {err:.3e} AU")
for i in range(N_TEST):
    err = np.linalg.norm(cpu_out["r_t"][i] - gpu_out["r_t"][i])
    max_err_gc = max(max_err_gc, err)
    print(f"  tno_{i}    |r_cpu - r_gpu| = {err:.3e} AU")
print(f"max CPU-vs-GPU error: {max_err_gc:.3e} AU")
print("Expect ~1e-10 AU: pure double-precision roundoff accumulated over")
print(f"{N_STEPS} steps of the IDENTICAL algorithm, not a formula divergence --")
print("both sides implement the exact same line-for-line Kepler drift/kick math.")

print("\n=== CUDA vs REBOUND (ground truth), heliocentric, no move_to_com ===")
max_err_gr = 0.0
for i, name in enumerate(names):
    p = sim.particles[1 + i]
    r_reb_helio = np.array([p.x, p.y, p.z]) - r_sun
    err = np.linalg.norm(r_reb_helio - gpu_out["r_m"][i])
    max_err_gr = max(max_err_gr, err)
    print(f"  {name:8s}  |r_gpu - r_rebound| = {err:.3e} AU  (|r|={np.linalg.norm(r_reb_helio):.2f} AU)")
for i in range(N_TEST):
    p = sim.particles[6 + i]
    r_reb_helio = np.array([p.x, p.y, p.z]) - r_sun
    err = np.linalg.norm(r_reb_helio - gpu_out["r_t"][i])
    max_err_gr = max(max_err_gr, err)
    print(f"  tno_{i}    |r_gpu - r_rebound| = {err:.3e} AU  (|r|={np.linalg.norm(r_reb_helio):.2f} AU)")
print(f"max GPU-vs-REBOUND error: {max_err_gr:.3e} AU")
print("Should land close to blockpersim_dev_validate.py's own CPU-vs-REBOUND")
print("error at this DT (documented there as 4.2e-4 AU) -- same O(dt^2)")
print("symplectic truncation error, not a GPU-specific bug.")

print("\n=== Ensemble driver: 3 independent simulations, CPU vs GPU ===")
sims = []
for k in range(3):
    r_m0_k, v_m0_k, m_m_k, r_t0_k, v_t0_k, _ = build_state(
        {"mass": 5.0 + k, "a": 400.0 + 50 * k, "e": 0.3, "i": 20.0, "Omega": 90.0, "omega": 45.0, "M": 0.0},
        N_TEST, np.random.default_rng(k + 1),
    )
    sims.append({"r_m0": r_m0_k, "v_m0": v_m0_k, "m_m": m_m_k, "r_t0": r_t0_k, "v_t0": v_t0_k})

t0 = time.perf_counter()
cpu_results = run_ensemble(sims, DT, N_STEPS)
print(f"CPU ran {len(cpu_results)} sims in {time.perf_counter()-t0:.2f}s")

t0 = time.perf_counter()
gpu_results = run_ensemble_cuda(sims, DT, N_STEPS, G, GM_SUN)
print(f"GPU ran {len(gpu_results)} sims ({len(gpu_results)} blocks) in {time.perf_counter()-t0:.2f}s")

max_err = 0.0
for k in range(3):
    err_m = np.linalg.norm(cpu_results[k]["r_m"] - gpu_results[k]["r_m"], axis=1).max()
    err_t = np.linalg.norm(cpu_results[k]["r_t"] - gpu_results[k]["r_t"], axis=1).max()
    max_err = max(max_err, err_m, err_t)
    print(f"  sim {k}: max |r_cpu-r_gpu| = {max(err_m, err_t):.3e} AU  "
          f"(hpx final |r| cpu={np.linalg.norm(cpu_results[k]['r_m'][4]):.2f}, "
          f"gpu={np.linalg.norm(gpu_results[k]['r_m'][4]):.2f} AU)")
print(f"max ensemble CPU-vs-GPU error: {max_err:.3e} AU")
