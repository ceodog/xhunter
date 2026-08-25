"""Real CUDA (numba.cuda) port of the massive-body + test-particle N-body
dynamics worker.run_one drives via REBOUND when disk_backend="rebound" --
i.e. a GPU-accelerated drop-in for the "pure N-body" path, not an
approximation. Ported line-for-line from scripts/blockpersim_cuda.py
(validated there against both blockpersim_dev.py's NumPy reference and
REBOUND directly -- see that script's module docstring for the full
validation record and the T4/A100 benchmark numbers this port inherits),
plus the new state_to_elements conversion (gpu_common.py) needed to turn
this kernel's Cartesian output into the same (a,e,i,Omega,omega,M) schema
worker.run_one already returns.

REQUIRES A REAL NVIDIA GPU (numba.cuda.is_available()) and the optional
`gpu` dependency group (`uv sync --extra gpu`) -- not part of the base
install, same reasoning as bench_test_particle_integrator.py's numba note:
numba's CUDA support can trail this project's numpy pin. This module is
NOT imported anywhere in the default (CPU) pipeline; only
orchestrate_gpu.py imports it, and only when a GPU backend is explicitly
requested.

Validation status (2026-08-25, Colab A100): the kernel physics itself
(block_per_sim_kernel below) is validated to ~1e-10 AU against both the
NumPy reference and REBOUND (see blockpersim_cuda.py). The NEW piece added
here -- run_ensemble_gpu_nbody's REBOUND-based initial-condition sampling
(reusing worker._add_giant_planets/_add_hpx/_add_primordial_disk verbatim,
so this matches worker.run_one's own sampling exactly) plus
state_to_elements' Cartesian-to-elements conversion and the a<=0/e>=1
survival filter -- has NOT yet been validated against a live REBOUND run
end-to-end. Do this before trusting output from this module for a real
dataset: run a REBOUND-backed worker.run_one and this module's
run_ensemble_gpu_nbody on the SAME sampled theta/nuisance/seed and compare
hpx_final/tnos directly, the same discipline every other GPU port in this
project's history was held to before being trusted.
"""

from __future__ import annotations

import math

import numpy as np
import rebound
from numba import cuda, float64

from planetx.constants import THETA_KEYS
from planetx.simgen.gpu_common import state_to_elements
from planetx.simgen.secular import GM_SUN
from planetx.simgen.worker import _add_giant_planets, _add_hpx, _add_primordial_disk

MAX_MASSIVE = 8  # compile-time shared-mem cap; project's n_massive is always 5 (4 giants + HPX)


def _measure_G() -> float:
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    return sim.G


G = _measure_G()


@cuda.jit(device=True, inline=True)
def _stumpff_c2c3_dev(psi):
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
    particles. Mirrors worker.run_one's disk_backend="rebound" physics
    exactly: democratic-heliocentric drift(dt/2) -> mutual kick(dt) among
    massive bodies -> test-particle kick(dt) -> drift(dt/2), repeated
    n_steps times. r_m/v_m/r_t/v_t are [S,*,3] device arrays, overwritten
    in place with the final state. n_test=0 is valid (massive-bodies-only
    integration, for the gpu_hybrid_secular backend)."""
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

    sqrt_gm = math.sqrt(gm)
    half = dt / 2.0

    for _step in range(n_steps):
        rx, ry, rz, vx, vy, vz = _drift_one_dev(rx, ry, rz, vx, vy, vz, gm, half, n_kepler_iter)

        if is_massive:
            sh_r[tid, 0] = rx; sh_r[tid, 1] = ry; sh_r[tid, 2] = rz
            sh_rho3[tid] = (rx * rx + ry * ry + rz * rz) ** 1.5
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
        cuda.syncthreads()

        rx, ry, rz, vx, vy, vz = _drift_one_dev(rx, ry, rz, vx, vy, vz, gm, half, n_kepler_iter)

    if is_massive:
        r_m[sim, tid, 0] = rx; r_m[sim, tid, 1] = ry; r_m[sim, tid, 2] = rz
        v_m[sim, tid, 0] = vx; v_m[sim, tid, 1] = vy; v_m[sim, tid, 2] = vz
    else:
        p = tid - n_massive
        r_t[sim, p, 0] = rx; r_t[sim, p, 1] = ry; r_t[sim, p, 2] = rz
        v_t[sim, p, 0] = vx; v_t[sim, p, 1] = vy; v_t[sim, p, 2] = vz


def run_ensemble_cuda(sims: list[dict], dt: float, n_steps: int, n_kepler_iter: int = 3) -> list[dict]:
    """Low-level driver: sims = list of {"r_m0","v_m0","m_m","r_t0","v_t0"}
    (heliocentric Cartesian, Msun/AU/yr). Returns final Cartesian state per
    sim -- callers wanting orbital elements should use
    run_ensemble_gpu_nbody instead."""
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


def _build_initial_state(theta: dict, nuisance: dict, n_test_particles: int, rng: np.random.Generator,
                          include_disk: bool):
    """REBOUND-based initial-condition sampling, reusing worker.py's own
    particle-adding helpers verbatim so this matches worker.run_one's
    sampling exactly (not a re-derived/independent RNG draw sequence)."""
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    _add_giant_planets(sim, rng)
    _add_hpx(sim, theta)
    sim.N_active = len(sim.particles)

    if include_disk and n_test_particles:
        _add_primordial_disk(sim, nuisance, n_test_particles, rng)

    r_m0 = np.array([[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z] for i in range(1, 6)])
    v_m0 = np.array([[sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz] for i in range(1, 6)])
    m_m = np.array([sim.particles[i].m for i in range(1, 6)])
    if include_disk and n_test_particles:
        r_t0 = np.array([[sim.particles[i].x, sim.particles[i].y, sim.particles[i].z] for i in range(6, 6 + n_test_particles)])
        v_t0 = np.array([[sim.particles[i].vx, sim.particles[i].vy, sim.particles[i].vz] for i in range(6, 6 + n_test_particles)])
    else:
        r_t0, v_t0 = np.empty((0, 3)), np.empty((0, 3))

    return r_m0, v_m0, m_m, r_t0, v_t0


def run_ensemble_gpu_nbody(
    theta_list: list[dict], nuisance_list: list[dict], seed_list: list[int],
    n_test_particles: int, dt_years: float, integration_years: float,
    n_kepler_iter: int = 3,
) -> list[dict]:
    """High-level driver matching worker.run_one's return schema exactly
    (theta, hpx_final, tnos) for disk_backend="rebound"-equivalent physics,
    batched across the whole ensemble in one GPU kernel launch. Every
    simulation must share n_test_particles (a single kernel launch has one
    block shape for the whole grid -- see block_per_sim_kernel's docstring).
    """
    S = len(theta_list)
    r_m0_all, v_m0_all, m_m_all, r_t0_all, v_t0_all = [], [], [], [], []
    for k in range(S):
        rng = np.random.default_rng(seed_list[k])
        r_m0, v_m0, m_m, r_t0, v_t0 = _build_initial_state(
            theta_list[k], nuisance_list[k], n_test_particles, rng, include_disk=True
        )
        r_m0_all.append(r_m0); v_m0_all.append(v_m0); m_m_all.append(m_m)
        r_t0_all.append(r_t0); v_t0_all.append(v_t0)

    sims = [
        {"r_m0": r_m0_all[k], "v_m0": v_m0_all[k], "m_m": m_m_all[k], "r_t0": r_t0_all[k], "v_t0": v_t0_all[k]}
        for k in range(S)
    ]
    n_steps = int(round(integration_years / dt_years))
    results = run_ensemble_cuda(sims, dt_years, n_steps, n_kepler_iter)

    out = []
    for k, res in enumerate(results):
        gm_m_k = G * (1.0 + m_m_all[k])
        hpx_state = state_to_elements(res["r_m"][4], res["v_m"][4], gm_m_k[4])
        hpx_final = {
            "mass": theta_list[k]["mass"],
            "a": float(hpx_state["a"]), "e": float(hpx_state["e"]), "i": float(np.degrees(hpx_state["i"])),
            "Omega": float(np.degrees(hpx_state["Omega"]) % 360.0),
            "omega": float(np.degrees(hpx_state["omega"]) % 360.0),
            "M": float(np.degrees(hpx_state["M"]) % 360.0),
        }

        tnos = []
        if n_test_particles:
            t_state = state_to_elements(res["r_t"], res["v_t"], GM_SUN)
            for p in range(n_test_particles):
                a_p, e_p = float(t_state["a"][p]), float(t_state["e"][p])
                if not np.isfinite(a_p) or not np.isfinite(e_p) or a_p <= 0.0 or e_p >= 1.0:
                    continue  # unbound/ejected, same filter worker.run_one applies to REBOUND's output
                tnos.append({
                    "a": a_p, "e": e_p, "i": float(np.degrees(t_state["i"][p])),
                    "Omega": float(np.degrees(t_state["Omega"][p]) % 360.0),
                    "omega": float(np.degrees(t_state["omega"][p]) % 360.0),
                    "M": float(np.degrees(t_state["M"][p]) % 360.0),
                })

        out.append({
            "theta": {kk: theta_list[k][kk] for kk in THETA_KEYS},
            "hpx_final": hpx_final,
            "tnos": tnos,
        })
    return out


def run_ensemble_gpu_massive_only(
    theta_list: list[dict], seed_list: list[int], dt_years: float, integration_years: float,
    n_kepler_iter: int = 3,
) -> list[dict]:
    """Same as run_ensemble_gpu_nbody but n_test_particles=0 -- massive
    bodies (Sun's reflex via giants+HPX) only, for the GPU-hybrid backend
    where the disk is propagated separately by gpu_secular.py. ~10x faster
    per-sim than the full run (5 threads/block instead of n_test+5 -- see
    the validation session's benchmark: ~16.3 days for 10000 sims at this
    project's production 4.5e9 yr / dt=0.5, vs ~235 days for the full
    disk-included GPU N-body run)."""
    S = len(theta_list)
    r_m0_all, v_m0_all, m_m_all = [], [], []
    for k in range(S):
        rng = np.random.default_rng(seed_list[k])
        r_m0, v_m0, m_m, _, _ = _build_initial_state(theta_list[k], {}, 0, rng, include_disk=False)
        r_m0_all.append(r_m0); v_m0_all.append(v_m0); m_m_all.append(m_m)

    sims = [
        {"r_m0": r_m0_all[k], "v_m0": v_m0_all[k], "m_m": m_m_all[k],
         "r_t0": np.empty((0, 3)), "v_t0": np.empty((0, 3))}
        for k in range(S)
    ]
    n_steps = int(round(integration_years / dt_years))
    results = run_ensemble_cuda(sims, dt_years, n_steps, n_kepler_iter)

    out = []
    for k, res in enumerate(results):
        gm_m_k = G * (1.0 + m_m_all[k])
        hpx_state = state_to_elements(res["r_m"][4], res["v_m"][4], gm_m_k[4])
        hpx_final = {
            "mass": theta_list[k]["mass"],
            "a": float(hpx_state["a"]), "e": float(hpx_state["e"]), "i": float(np.degrees(hpx_state["i"])),
            "Omega": float(np.degrees(hpx_state["Omega"]) % 360.0),
            "omega": float(np.degrees(hpx_state["omega"]) % 360.0),
            "M": float(np.degrees(hpx_state["M"]) % 360.0),
        }
        out.append(hpx_final)
    return out
