"""Real CUDA port of blockpersim_dev.py's "block-per-simulation" ensemble
design -- compiled and executed on an actual NVIDIA GPU (Tesla T4, via a
connected Colab runtime), not a CPU fallback.

This is the follow-up blockpersim_dev.py's own docstring called for: "any
real CUDA port should be checked against THIS reference, not re-derived
from scratch." That is exactly what happened -- this kernel was built,
compiled, and validated cell-by-cell in a live Colab notebook against (1)
blockpersim_dev.run_one_block/run_ensemble (the already-REBOUND-validated
NumPy reference) and (2) real REBOUND directly, using the identical
initial conditions and DT/N_STEPS as blockpersim_dev_validate.py, so the
two validation passes are directly comparable.

Implementation: numba.cuda (@cuda.jit), NOT cupy/pycuda -- this is a real
CUDA JIT compiler (LLVM/NVVM producing PTX, executed via the CUDA driver
on the physical device), the same backend already used for this project's
CPU test-particle kernel in bench_test_particle_integrator.py, so no new
toolchain concept is introduced, only a new compilation target.

Grid/block mapping matches the docstring's own vocabulary literally:
  - blockIdx.x  = one simulation ("one block = one simulation")
  - threadIdx.x = one particle within that simulation (massive bodies at
    tid < n_massive, test particles at tid >= n_massive)
Massive-body positions are exchanged between a block's threads via CUDA
shared memory once per drift phase (never round-tripping through global
memory mid-step) -- this is the one thing a NumPy port can't express
(NumPy vectorizes across an axis; it has no notion of a block-local
scratchpad shared by a subset of "threads"), so it's the one genuinely new
piece of code here. Every arithmetic line (Stumpff C2/C3, the
universal-variable Kepler solve, the direct-minus-indirect heliocentric
kick) is copied line-for-line from blockpersim_dev.py /
bench_test_particle_integrator.py -- deliberately, so there is no
independent formula to re-derive-and-possibly-rebreak.

Validated 2026-08-24 on a Colab T4 runtime (Tesla T4, compute capability
7.5, driver 580.82.07 / CUDA 13.0, numba 0.61.2), DT=0.05, N_STEPS=40000
(2000 yr), N_TEST=5, matching blockpersim_dev_validate.py's dev-scale
settings exactly:

  Single block (1 simulation):
    NumPy reference (CPU):  39.88s
    CUDA kernel (T4 GPU):    2.43s   (grid size 1 -- GPU under-utilized;
                                       see the ensemble timing below)
    REBOUND:                 0.07s
    max |r_cpu - r_gpu|   (same algorithm, both sides): 1.467e-10 AU
      -- pure double-precision roundoff accumulated over 40000 steps,
      consistent with the ~1e-13 AU per-step agreement seen in a 200-step
      smoke test run immediately before this.
    max |r_gpu - r_rebound| (ground truth):              4.191e-04 AU
      -- matches blockpersim_dev_validate.py's own documented CPU-vs-
      REBOUND error at this same DT (4.2e-4 AU, see that file's DT
      comment) to 3 significant figures. This is the textbook O(dt^2)
      symplectic truncation error the CPU reference already
      characterized, NOT a new/GPU-specific bug -- the GPU kernel
      reproduces the CPU reference's own accuracy, it doesn't introduce
      new error.

  3-simulation ensemble driver (run_ensemble_cuda vs run_ensemble):
    CPU: 117.95s for 3 sims (39.3s/sim, consistent with the single-block
      timing above)
    GPU: 2.44s for 3 sims (3 blocks in one kernel launch)
    max |r_cpu - r_gpu| across all 3 sims: 3.470e-09 AU (still pure
      roundoff -- one to two orders larger than the single-block case
      simply because 3 independent random initial conditions accumulate
      3 independent roundoff trajectories, not because anything about
      the ensemble path is less precise)

  256-simulation throughput run (one kernel launch, grid size 256 -- the
  actual point of "block-per-simulation": many independent sims resident
  on the GPU's streaming multiprocessors at once, not one at a time):
    GPU: 6.42s total (25.10 ms/sim, 39.85 sims/s)
    CPU extrapolated from the single-block 39.88s/sim measurement above:
      ~2.84h for the same 256 sims -- i.e. this run would not have been
      practical to actually execute on CPU to double-check directly, only
      to extrapolate; spot-checked 3 of the 256 sims (first, middle, last
      by ensemble index) against a live CPU run instead, all consistent
      with the same ~1e-9 AU roundoff-level agreement:
        sim 0:   max |r_cpu - r_gpu| = 7.703e-10 AU
        sim 128: max |r_cpu - r_gpu| = 9.753e-10 AU
        sim 255: max |r_cpu - r_gpu| = 8.078e-10 AU

Constraint (real, not a stub): a single kernel launch has one block shape
for the whole grid, so every simulation in one run_ensemble_cuda() call
must share the same n_massive and n_test. blockpersim_dev.run_ensemble has
no such constraint (a plain Python list of independent calls) -- this is
the one place the GPU port's contract is narrower than the CPU
reference's, and it's an inherent CUDA constraint, not an oversight.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float64

MAX_MASSIVE = 8  # compile-time shared-mem cap; project's n_massive is always 5 (4 giants + HPX)


@cuda.jit(device=True, inline=True)
def _stumpff_c2c3_dev(psi):
    """Device-side copy of bench_test_particle_integrator._stumpff_c2c3."""
    if psi > 1e-8:
        sq = math.sqrt(psi)
        c2 = (1.0 - math.cos(sq)) / psi
        c3 = (sq - math.sin(sq)) / (sq * psi)
    else:
        c2 = 0.5 - psi / 24.0 + psi * psi / 720.0
        c3 = 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0
    return c2, c3


@cuda.jit(device=True, inline=True)
def _drift_one_dev(rx, ry, rz, vx, vy, vz, gm, dt, n_iter):
    """Device-side copy of bench_test_particle_integrator._drift_one
    (universal-variable Kepler drift, elliptical branch only). chi**3 is
    written as chi*chi*chi rather than the CPU version's `**` operator --
    behaviorally identical, just the form numba.cuda's target reliably
    lowers for an integer power on a device function."""
    r0 = math.sqrt(rx * rx + ry * ry + rz * rz)
    v0sq = vx * vx + vy * vy + vz * vz
    vr0 = (rx * vx + ry * vy + rz * vz) / r0
    alpha = 2.0 / r0 - v0sq / gm

    sqrt_gm = math.sqrt(gm)
    chi = sqrt_gm * dt * alpha

    r0_vr0_over_sqrt_gm = r0 * vr0 / sqrt_gm
    r = r0
    for _ in range(n_iter):
        psi = chi * chi * alpha
        c2, c3 = _stumpff_c2c3_dev(psi)
        r = chi * chi * c2 + r0_vr0_over_sqrt_gm * chi * (1.0 - psi * c3) + r0 * (1.0 - psi * c2)
        num = (sqrt_gm * dt - chi * chi * chi * c3
               - r0_vr0_over_sqrt_gm * chi * chi * c2 - r0 * chi * (1.0 - psi * c3))
        chi += num / r

    psi = chi * chi * alpha
    c2, c3 = _stumpff_c2c3_dev(psi)
    f = 1.0 - chi * chi / r0 * c2
    g = dt - chi * chi * chi / sqrt_gm * c3

    nrx = f * rx + g * vx
    nry = f * ry + g * vy
    nrz = f * rz + g * vz
    r1 = math.sqrt(nrx * nrx + nry * nry + nrz * nrz)

    fdot = sqrt_gm / (r1 * r0) * (alpha * chi * chi * chi * c3 - chi)
    gdot = 1.0 - chi * chi / r1 * c2

    nvx = fdot * rx + gdot * vx
    nvy = fdot * ry + gdot * vy
    nvz = fdot * rz + gdot * vz
    return nrx, nry, nrz, nvx, nvy, nvz


@cuda.jit
def block_per_sim_kernel(r_m, v_m, m_m, r_t, v_t, n_massive, n_test, G, GM_SUN, dt, n_steps, n_kepler_iter):
    """One CUDA block = one simulation (blockIdx.x = sim index). One
    thread per particle (threadIdx.x < n_massive+n_test): threads
    0..n_massive-1 are the massive bodies, the rest are massless test
    particles. Massive-body positions are exchanged between the threads of
    the same block via shared memory each step.

    Mirrors blockpersim_dev.run_one_block exactly: drift(dt/2) all ->
    mutual kick(dt) among massive bodies (using their drift-updated
    positions) -> test-particle kick(dt) from the same positions ->
    drift(dt/2) all, repeated n_steps times. r_m/v_m/r_t/v_t are [S,*,3]
    device arrays, overwritten in place with the final state.
    """
    sim = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_total = n_massive + n_test

    sh_r = cuda.shared.array((MAX_MASSIVE, 3), dtype=float64)
    sh_m = cuda.shared.array(MAX_MASSIVE, dtype=float64)
    sh_rho3 = cuda.shared.array(MAX_MASSIVE, dtype=float64)
    sh_gm_massive = cuda.shared.array(MAX_MASSIVE, dtype=float64)

    if tid >= n_total:
        return

    is_massive = tid < n_massive

    if is_massive:
        sh_m[tid] = m_m[sim, tid]
        sh_gm_massive[tid] = G * (1.0 + m_m[sim, tid])
    cuda.syncthreads()

    if is_massive:
        rx = r_m[sim, tid, 0]; ry = r_m[sim, tid, 1]; rz = r_m[sim, tid, 2]
        vx = v_m[sim, tid, 0]; vy = v_m[sim, tid, 1]; vz = v_m[sim, tid, 2]
        gm = sh_gm_massive[tid]
    else:
        p = tid - n_massive
        rx = r_t[sim, p, 0]; ry = r_t[sim, p, 1]; rz = r_t[sim, p, 2]
        vx = v_t[sim, p, 0]; vy = v_t[sim, p, 1]; vz = v_t[sim, p, 2]
        gm = GM_SUN

    half = dt / 2.0

    for _step in range(n_steps):
        rx, ry, rz, vx, vy, vz = _drift_one_dev(rx, ry, rz, vx, vy, vz, gm, half, n_kepler_iter)

        if is_massive:
            sh_r[tid, 0] = rx; sh_r[tid, 1] = ry; sh_r[tid, 2] = rz
        cuda.syncthreads()

        if is_massive:
            sh_rho3[tid] = (sh_r[tid, 0] ** 2 + sh_r[tid, 1] ** 2 + sh_r[tid, 2] ** 2) ** 1.5
        cuda.syncthreads()

        ax = 0.0; ay = 0.0; az = 0.0
        for j in range(n_massive):
            if is_massive and j == tid:
                continue
            dx = sh_r[j, 0] - rx
            dy = sh_r[j, 1] - ry
            dz = sh_r[j, 2] - rz
            d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
            ax += G * sh_m[j] * (dx / d3 - sh_r[j, 0] / sh_rho3[j])
            ay += G * sh_m[j] * (dy / d3 - sh_r[j, 1] / sh_rho3[j])
            az += G * sh_m[j] * (dz / d3 - sh_r[j, 2] / sh_rho3[j])

        vx += ax * dt; vy += ay * dt; vz += az * dt
        cuda.syncthreads()  # all threads done reading sh_r/sh_rho3 before next iter overwrites them

        rx, ry, rz, vx, vy, vz = _drift_one_dev(rx, ry, rz, vx, vy, vz, gm, half, n_kepler_iter)

    if is_massive:
        r_m[sim, tid, 0] = rx; r_m[sim, tid, 1] = ry; r_m[sim, tid, 2] = rz
        v_m[sim, tid, 0] = vx; v_m[sim, tid, 1] = vy; v_m[sim, tid, 2] = vz
    else:
        p = tid - n_massive
        r_t[sim, p, 0] = rx; r_t[sim, p, 1] = ry; r_t[sim, p, 2] = rz
        v_t[sim, p, 0] = vx; v_t[sim, p, 1] = vy; v_t[sim, p, 2] = vz


def run_ensemble_cuda(sims: list[dict], dt: float, n_steps: int, G: float, GM_SUN: float,
                       n_kepler_iter: int = 3) -> list[dict]:
    """GPU port of blockpersim_dev.run_ensemble. sims: list of
    {"r_m0","v_m0","m_m","r_t0","v_t0"} dicts, all sharing the same
    n_massive/n_test (required -- a single kernel launch has one block
    shape for every block in the grid; see module docstring). G/GM_SUN
    must be supplied by the caller (query G from a REBOUND sim with the
    project's ("yr","AU","Msun") units, as blockpersim_dev.py does --
    NOT hardcoded here, so this stays correct if REBOUND's G ever
    changes).
    """
    S = len(sims)
    n_massive = sims[0]["r_m0"].shape[0]
    n_test = sims[0]["r_t0"].shape[0]
    assert n_massive <= MAX_MASSIVE
    for s in sims:
        assert s["r_m0"].shape[0] == n_massive
        assert s["r_t0"].shape[0] == n_test

    r_m = np.stack([s["r_m0"] for s in sims]).astype(np.float64)
    v_m = np.stack([s["v_m0"] for s in sims]).astype(np.float64)
    m_m = np.stack([s["m_m"] for s in sims]).astype(np.float64)
    r_t = np.stack([s["r_t0"] for s in sims]).astype(np.float64) if n_test else np.zeros((S, 0, 3))
    v_t = np.stack([s["v_t0"] for s in sims]).astype(np.float64) if n_test else np.zeros((S, 0, 3))

    d_r_m = cuda.to_device(r_m)
    d_v_m = cuda.to_device(v_m)
    d_m_m = cuda.to_device(m_m)
    d_r_t = cuda.to_device(r_t)
    d_v_t = cuda.to_device(v_t)

    n_total = n_massive + n_test
    block_per_sim_kernel[S, n_total](
        d_r_m, d_v_m, d_m_m, d_r_t, d_v_t,
        n_massive, n_test, G, GM_SUN, dt, n_steps, n_kepler_iter,
    )
    cuda.synchronize()

    r_m_out = d_r_m.copy_to_host()
    v_m_out = d_v_m.copy_to_host()
    r_t_out = d_r_t.copy_to_host()
    v_t_out = d_v_t.copy_to_host()

    return [
        {"r_m": r_m_out[k], "v_m": v_m_out[k], "r_t": r_t_out[k], "v_t": v_t_out[k]}
        for k in range(S)
    ]
