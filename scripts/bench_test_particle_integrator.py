"""Stage 1 prototype/benchmark for the fast vectorized test-particle propagator
(see /Users/xtan/.claude/plans/robust-puzzling-flamingo.md).

Architecture: REBOUND (validated, GR-capable) drives the Sun + 4 giants + HPX
among themselves exactly as worker.py already does; a Numba-JIT'd kernel
propagates the (much larger) massless test-particle array through the same
timesteps using a Drift-Kick-Drift split in heliocentric coordinates:

  - Drift: each test particle's own two-body Kepler motion around the Sun
    (GM_sun completely dominates -- giants+HPX total ~1.3e-3 Msun vs Sun's
    1.0), solved via the universal-variable formulation (Stumpff C2/C3 +
    Gauss f,g functions) -- the same family of method REBOUND's own WHFast
    uses internally (confirmed via the Stage-1 research against REBOUND's
    GitHub source), not the classical-elements Newton solve in
    planetx.simgen.selection._solve_kepler (that solver is a poor fit for
    billions of repeated calls -- see the plan's Context section).
  - Kick: direct + indirect heliocentric perturbation from the giants + HPX,
    using REBOUND-supplied perturber positions at the *midpoint* of each
    step (needed for the DKD split), extracted from REBOUND driven forward
    half a step at a time. This project's disk is always bound (e < 0.9 in
    prior.yaml), so alpha = 1/a > 0 always -- only the elliptical branch of
    the universal Kepler solver is implemented (no parabolic/hyperbolic
    fallback), which is a deliberate scope simplification for this specific
    disk, not a general-purpose Kepler solver.

This script does NOT touch the production pipeline (worker.py/orchestrate.py
are untouched). It only benchmarks cost/step and validates against REBOUND
on short/medium timescales, per the plan's verification section.

Setup: numba is NOT declared as a project extra (e.g. `[project.optional-
dependencies]`), because that would share this project's single lockfile/
environment -- and numba's supported numpy range currently trails behind
this project's numpy==2.2.6 (confirmed: `uv sync` with numba added as an
extra resolves numba==0.67.0, whose llvmlite==0.49.0 dependency has no
prebuilt wheel for this platform and fails to build from source; forcing an
older numba==0.60.0 that does have wheels here would downgrade the *whole
project's* numpy to 2.0.2). Run this script in its own throwaway venv
instead, kept fully separate from the main project's .venv:

    uv venv .venv-bench --python 3.10
    uv pip install --python .venv-bench/bin/python3.10 \\
        "numba==0.60.0" "rebound>=4.4,<5.1.0" pyyaml
    uv pip install --python .venv-bench/bin/python3.10 --no-deps -e .
    .venv-bench/bin/python3.10 scripts/bench_test_particle_integrator.py
"""

from __future__ import annotations

import time

import numba
import numpy as np
import rebound

from planetx.constants import EARTH_MASS_IN_MSUN, GIANT_PLANETS

def _measure_G():
    """REBOUND's ('yr','AU','Msun') units do NOT set G=1 -- verified
    empirically (sim.G == 39.476926421373, the Gaussian gravitational
    constant, close to but not exactly 4*pi**2 == 39.4784...). Query it from
    REBOUND directly rather than hardcoding either value, so the Numba
    kernel's GM is guaranteed to match whatever this REBOUND build actually
    uses.
    """
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    return sim.G


G = _measure_G()
GM_SUN = G * 1.0  # Sun's mass is exactly 1.0 Msun everywhere in this project


# ---------------------------------------------------------------------------
# Numba kernel: universal-variable Kepler drift + heliocentric kick
# ---------------------------------------------------------------------------


@numba.njit(cache=True, fastmath=True)
def _stumpff_c2c3(psi):
    """C2, C3 Stumpff functions, elliptical branch only (psi >= 0 always for
    this project's bound disk -- see module docstring)."""
    if psi > 1e-8:
        sq = np.sqrt(psi)
        c2 = (1.0 - np.cos(sq)) / psi
        c3 = (sq - np.sin(sq)) / (sq * psi)
    else:
        # series limit as psi -> 0 (avoids 0/0); good to ~1e-8 relative error
        c2 = 0.5 - psi / 24.0 + psi * psi / 720.0
        c3 = 1.0 / 6.0 - psi / 120.0 + psi * psi / 5040.0
    return c2, c3


@numba.njit(cache=True, fastmath=True)
def _drift_one(rx, ry, rz, vx, vy, vz, gm, dt, n_iter):
    """Universal-variable Kepler drift of one (r, v) forward by dt under GM.
    Elliptical branch only. Returns (rx,ry,rz,vx,vy,vz) after the drift."""
    r0 = np.sqrt(rx * rx + ry * ry + rz * rz)
    v0sq = vx * vx + vy * vy + vz * vz
    vr0 = (rx * vx + ry * vy + rz * vz) / r0
    alpha = 2.0 / r0 - v0sq / gm  # = 1/a, > 0 for this project's bound disk

    sqrt_gm = np.sqrt(gm)
    chi = sqrt_gm * dt * alpha  # standard elliptical initial guess

    r0_vr0_over_sqrt_gm = r0 * vr0 / sqrt_gm  # standard universal-Kepler coefficient (r0*vr0/sqrt(GM), not vr0/sqrt(GM) alone -- see plan doc "Root cause of the drift" for the derivation/citation this was checked against)
    r = r0  # will hold the "r" (radial distance) implied by the current chi
    for _ in range(n_iter):
        psi = chi * chi * alpha
        c2, c3 = _stumpff_c2c3(psi)
        r = chi * chi * c2 + r0_vr0_over_sqrt_gm * chi * (1.0 - psi * c3) + r0 * (1.0 - psi * c2)
        num = sqrt_gm * dt - chi**3 * c3 - r0_vr0_over_sqrt_gm * chi * chi * c2 - r0 * chi * (1.0 - psi * c3)
        chi += num / r

    psi = chi * chi * alpha
    c2, c3 = _stumpff_c2c3(psi)
    f = 1.0 - chi * chi / r0 * c2
    g = dt - chi**3 / sqrt_gm * c3

    nrx = f * rx + g * vx
    nry = f * ry + g * vy
    nrz = f * rz + g * vz
    r1 = np.sqrt(nrx * nrx + nry * nry + nrz * nrz)

    fdot = sqrt_gm / (r1 * r0) * (alpha * chi**3 * c3 - chi)
    gdot = 1.0 - chi * chi / r1 * c2

    nvx = fdot * rx + gdot * vx
    nvy = fdot * ry + gdot * vy
    nvz = fdot * rz + gdot * vz
    return nrx, nry, nrz, nvx, nvy, nvz


@numba.njit(cache=True, fastmath=True, parallel=True)
def propagate(r0, v0, rho_mid, m_pert, gm_sun, dt, n_kepler_iter):
    """Propagate N test particles through len(rho_mid) DKD steps.

    r0, v0: [N,3] initial heliocentric state.
    rho_mid: [n_steps, n_pert, 3] perturber heliocentric positions at the
        midpoint of each step (supplied by REBOUND, computed in Python).
    m_pert: [n_pert] perturber G*mass values (see G/GM_SUN note above).
    Returns final (r, v) as two [N,3] arrays.

    parallel=True + prange over particles: each particle is independent
    (see module docstring -- no backreaction, no particle-particle
    interaction), so this loop is embarrassingly parallel across threads.
    fastmath=True: acceptable here since nothing in this kernel relies on
    strict IEEE edge-case behavior (no NaN/Inf-sensitive branching).
    """
    n = r0.shape[0]
    n_steps = rho_mid.shape[0]
    n_pert = m_pert.shape[0]
    r = r0.copy()
    v = v0.copy()
    half = dt / 2.0

    for s in range(n_steps):
        for i in numba.prange(n):
            r[i, 0], r[i, 1], r[i, 2], v[i, 0], v[i, 1], v[i, 2] = _drift_one(
                r[i, 0], r[i, 1], r[i, 2], v[i, 0], v[i, 1], v[i, 2], gm_sun, half, n_kepler_iter
            )

        # indirect term (-G*m_j*rho_j/|rho_j|^3) depends only on the
        # perturber, not the test particle -- hoist it out of the particle
        # loop instead of recomputing it N times per step.
        rho3 = np.empty(n_pert)
        for p in range(n_pert):
            rho3[p] = (
                rho_mid[s, p, 0] ** 2 + rho_mid[s, p, 1] ** 2 + rho_mid[s, p, 2] ** 2
            ) ** 1.5

        # kick using perturber positions at this step's midpoint
        for i in numba.prange(n):
            ax, ay, az = 0.0, 0.0, 0.0
            for p in range(n_pert):
                dx = rho_mid[s, p, 0] - r[i, 0]
                dy = rho_mid[s, p, 1] - r[i, 1]
                dz = rho_mid[s, p, 2] - r[i, 2]
                d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
                ax += m_pert[p] * (dx / d3 - rho_mid[s, p, 0] / rho3[p])
                ay += m_pert[p] * (dy / d3 - rho_mid[s, p, 1] / rho3[p])
                az += m_pert[p] * (dz / d3 - rho_mid[s, p, 2] / rho3[p])
            v[i, 0] += dt * ax
            v[i, 1] += dt * ay
            v[i, 2] += dt * az

        for i in numba.prange(n):
            r[i, 0], r[i, 1], r[i, 2], v[i, 0], v[i, 1], v[i, 2] = _drift_one(
                r[i, 0], r[i, 1], r[i, 2], v[i, 0], v[i, 1], v[i, 2], gm_sun, half, n_kepler_iter
            )

    return r, v


# ---------------------------------------------------------------------------
# REBOUND helpers: build the massive-body sim, drive it, extract perturber
# heliocentric positions at each step's midpoint.
# ---------------------------------------------------------------------------


def native_dt_for(dt_disk, dt_massive_floor=0.5):
    """REBOUND's WHFast must never be asked to integrate() to a target time
    that isn't an exact multiple of sim.dt -- verified empirically (see
    plan doc "Root cause of the multi-rate drift"): doing so forces an
    irregular partial step and produces a measurably wrong trajectory even
    with safe_mode=1, independent of synchronization. Since
    drive_and_extract_mid() requests both t+dt_disk/2 (the kick midpoint)
    and t+dt_disk, sim.dt must evenly divide dt_disk/2. Use the finer of
    (a) what Jupiter's own period requires (dt_massive_floor, default 0.593
    yr's safe margin -> 0.5 yr) and (b) dt_disk/2, so the massive bodies are
    never under-resolved and the disk-step midpoint always lands exactly on
    a native step boundary.
    """
    return min(dt_massive_floor, dt_disk / 2.0)


def build_massive_sim(hpx_mass_earth, hpx_a, hpx_e, hpx_i_deg, seed=0, dt_massive=0.5):
    """dt_massive is REBOUND's own native sim.dt -- see native_dt_for()'s
    docstring for why this must be chosen relative to dt_disk, not just set
    to 0.5 unconditionally. NOTE: earlier versions of this script never set
    sim.dt explicitly, so REBOUND silently used its own default (0.001,
    500x finer than needed) -- harmless for correctness but meant the
    standalone script's own "REBOUND massive-body drive" timing print was
    not representative of production worker.py (which does set dt_years
    correctly). Fixed here; doesn't change any previously-reported
    production numbers, which all came from calling worker.run_one directly.
    """
    rng = np.random.default_rng(seed)
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    for name, el in GIANT_PLANETS.items():
        sim.add(
            m=el["m"], a=el["a"], e=el["e"], inc=np.radians(el["inc"]),
            Omega=np.radians(el["Omega"]), omega=np.radians(el["omega"]),
            M=rng.uniform(0, 2 * np.pi), name=name,
        )
    sim.add(
        m=hpx_mass_earth * EARTH_MASS_IN_MSUN, a=hpx_a, e=hpx_e, inc=np.radians(hpx_i_deg),
        Omega=0.0, omega=0.0, M=0.0, name="hpx",
    )
    sim.N_active = len(sim.particles)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.integrator.safe_mode = 0
    sim.dt = dt_massive
    return sim


def drive_and_extract_mid(sim, dt_disk, n_steps, track_index=None):
    """Advance sim in DKD half-steps *at the disk's own cadence* (dt_disk,
    which may be much coarser than sim.dt), recording perturber heliocentric
    positions (relative to the Sun particle) at each disk step's midpoint.
    REBOUND internally still takes as many sim.dt-sized sub-steps as needed
    to reach each target time, so the massive bodies remain correctly
    resolved at their own (finer) required timestep regardless of dt_disk --
    this function only controls how often their position is *sampled* for
    handoff to the disk kernel. Returns rho_mid [n_steps, n_pert, 3] and
    masses [n_pert].

    If track_index is given (e.g. Neptune's particle index), also returns a
    [n_steps] array of that particle's mean longitude (Omega+omega+M, deg)
    at the *end* of each disk step -- used by validate_multirate_disk.py to
    build the resonant-angle time series at whatever cadence the disk kernel
    is actually stepping at, from the same driving loop (no separate,
    potentially time-misaligned re-simulation needed).
    """
    n_pert = len(sim.particles) - 1  # everything except the Sun
    # multiply by G here (not just Sun's mass, per the GM_SUN fix above) --
    # the kick acceleration needs G*m_j, not just m_j.
    m_pert = G * np.array([sim.particles[k].m for k in range(1, n_pert + 1)])
    rho_mid = np.empty((n_steps, n_pert, 3))
    lam_track = np.empty(n_steps) if track_index is not None else None
    t = sim.t
    for s in range(n_steps):
        sim.integrate(t + dt_disk / 2)
        sun = sim.particles[0]
        for p in range(n_pert):
            part = sim.particles[p + 1]
            rho_mid[s, p] = (part.x - sun.x, part.y - sun.y, part.z - sun.z)
        t += dt_disk
        sim.integrate(t)
        if track_index is not None:
            o = sim.particles[track_index].orbit(primary=sim.particles[0])
            lam_track[s] = (np.degrees(o.Omega) + np.degrees(o.omega) + np.degrees(o.M)) % 360.0
    if track_index is not None:
        return rho_mid, m_pert, lam_track
    return rho_mid, m_pert


def heliocentric_state(sim, index):
    sun = sim.particles[0]
    p = sim.particles[index]
    r = np.array([p.x - sun.x, p.y - sun.y, p.z - sun.z])
    v = np.array([p.vx - sun.vx, p.vy - sun.vy, p.vz - sun.vz])
    return r, v


def elements_from_state(gm_sun, r, v):
    """Reuse REBOUND's own state->elements conversion (via a throwaway
    two-body sim) so both the REBOUND and Numba results are converted to
    osculating elements identically -- only the propagation itself is under
    test, not a second, separately-written elements routine."""
    tmp = rebound.Simulation()
    tmp.units = ("yr", "AU", "Msun")
    tmp.add(m=gm_sun)  # gm_sun here is the Sun's MASS (1.0 Msun); this throwaway
    # sim sets its own correct G internally via tmp.units, independent of GM_SUN above
    tmp.add(m=0.0, x=r[0], y=r[1], z=r[2], vx=v[0], vy=v[1], vz=v[2])
    o = tmp.particles[1].orbit(primary=tmp.particles[0])
    return o.a, o.e, np.degrees(o.inc), np.degrees(o.Omega) % 360, np.degrees(o.omega) % 360, np.degrees(o.M) % 360


# ---------------------------------------------------------------------------
# Benchmark + validation
# ---------------------------------------------------------------------------


def make_disk(n, seed=1):
    rng = np.random.default_rng(seed)
    r0 = np.empty((n, 3))
    v0 = np.empty((n, 3))
    gm_sun = 1.0
    for i in range(n):
        a = rng.uniform(30, 100)
        e = float(np.clip(rng.rayleigh(0.08), 0, 0.9))
        inc = np.radians(float(np.clip(rng.rayleigh(5), 0, 60)))
        Omega = rng.uniform(0, 2 * np.pi)
        omega = rng.uniform(0, 2 * np.pi)
        M = rng.uniform(0, 2 * np.pi)
        tmp = rebound.Simulation()
        tmp.units = ("yr", "AU", "Msun")
        tmp.add(m=gm_sun)
        tmp.add(m=0.0, a=a, e=e, inc=inc, Omega=Omega, omega=omega, M=M)
        p = tmp.particles[1]
        r0[i] = (p.x, p.y, p.z)
        v0[i] = (p.vx, p.vy, p.vz)
    return r0, v0


def run_benchmark(n_test_values, n_steps, dt=0.5, hpx=(5.0, 400.0, 0.3, 20.0)):
    print(f"# n_steps={n_steps} dt={dt} ({n_steps*dt:.1f} yr integration)")
    sim = build_massive_sim(*hpx, dt_massive=native_dt_for(dt))
    t0 = time.perf_counter()
    rho_mid, m_pert = drive_and_extract_mid(sim, dt, n_steps)
    t_rebound_drive = time.perf_counter() - t0
    print(f"REBOUND massive-body drive (half+full steps): {t_rebound_drive:.3f}s "
          f"({t_rebound_drive/n_steps*1e6:.2f} us/step, independent of n_test_particles)")

    for n_test in n_test_values:
        r0, v0 = make_disk(n_test)
        # warm up JIT (compile time shouldn't count against the measured rate)
        propagate(r0[:min(5, n_test)].copy(), v0[:min(5, n_test)].copy(), rho_mid[:2], m_pert, GM_SUN, dt, 3)

        t0 = time.perf_counter()
        rf, vf = propagate(r0.copy(), v0.copy(), rho_mid, m_pert, GM_SUN, dt, 3)
        t_kernel = time.perf_counter() - t0
        per_particle_us = (t_kernel / n_steps / max(n_test, 1)) * 1e6
        print(f"n_test={n_test:6d}  numba kernel: {t_kernel:.3f}s total, "
              f"{t_kernel/n_steps*1e6:8.2f} us/step, {per_particle_us:.4f} us/particle/step")


def run_validation(n_test, n_steps_short, n_steps_medium, dt=0.5, hpx=(5.0, 400.0, 0.3, 20.0)):
    print(f"\n# Validation: n_test={n_test}")

    # --- short-timescale exact trajectory match ---
    sim_r = build_massive_sim(*hpx)
    r0, v0 = make_disk(n_test, seed=2)
    for i in range(n_test):
        sim_r.add(m=0.0, x=r0[i, 0] + sim_r.particles[0].x, y=r0[i, 1] + sim_r.particles[0].y,
                  z=r0[i, 2] + sim_r.particles[0].z,
                  vx=v0[i, 0] + sim_r.particles[0].vx, vy=v0[i, 1] + sim_r.particles[0].vy,
                  vz=v0[i, 2] + sim_r.particles[0].vz)
    sim_r.integrate(n_steps_short * dt)
    rebound_elems = np.array([
        [*sim_r.particles[6 + i].orbit(primary=sim_r.particles[0])._asdict().values()][:6]
        if False else elements_from_state(1.0, *heliocentric_state(sim_r, 6 + i))
        for i in range(n_test)
    ])

    sim_m = build_massive_sim(*hpx, dt_massive=native_dt_for(dt))
    rho_mid, m_pert = drive_and_extract_mid(sim_m, dt, n_steps_short)
    rf, vf = propagate(r0.copy(), v0.copy(), rho_mid, m_pert, GM_SUN, dt, 3)
    numba_elems = np.array([elements_from_state(1.0, rf[i], vf[i]) for i in range(n_test)])

    diff = np.abs(rebound_elems - numba_elems)
    diff[:, 2:] = np.minimum(diff[:, 2:], 360 - diff[:, 2:])  # wrap angles
    labels = ["a(AU)", "e", "i(deg)", "Omega(deg)", "omega(deg)", "M(deg)"]
    print(f"short-timescale ({n_steps_short*dt:.0f} yr) max |diff| over {n_test} particles:")
    for k, lab in enumerate(labels):
        print(f"  {lab:12s} max={diff[:,k].max():.3e}  mean={diff[:,k].mean():.3e}")

    # --- medium-timescale statistical match ---
    sim_r2 = build_massive_sim(*hpx)
    r0b, v0b = make_disk(n_test, seed=3)
    for i in range(n_test):
        sim_r2.add(m=0.0, x=r0b[i, 0] + sim_r2.particles[0].x, y=r0b[i, 1] + sim_r2.particles[0].y,
                   z=r0b[i, 2] + sim_r2.particles[0].z,
                   vx=v0b[i, 0] + sim_r2.particles[0].vx, vy=v0b[i, 1] + sim_r2.particles[0].vy,
                   vz=v0b[i, 2] + sim_r2.particles[0].vz)
    sim_r2.integrate(n_steps_medium * dt)
    rebound_elems2 = np.array([elements_from_state(1.0, *heliocentric_state(sim_r2, 6 + i)) for i in range(n_test)])

    sim_m2 = build_massive_sim(*hpx, dt_massive=native_dt_for(dt))
    rho_mid2, m_pert2 = drive_and_extract_mid(sim_m2, dt, n_steps_medium)
    rf2, vf2 = propagate(r0b.copy(), v0b.copy(), rho_mid2, m_pert2, GM_SUN, dt, 3)
    numba_elems2 = np.array([elements_from_state(1.0, rf2[i], vf2[i]) for i in range(n_test)])

    print(f"\nmedium-timescale ({n_steps_medium*dt:.0f} yr) distributional comparison over {n_test} particles:")
    for k, lab in enumerate(labels):
        rmean, rstd = rebound_elems2[:, k].mean(), rebound_elems2[:, k].std()
        nmean, nstd = numba_elems2[:, k].mean(), numba_elems2[:, k].std()
        print(f"  {lab:12s} REBOUND mean={rmean:9.4f} std={rstd:9.4f}   "
              f"numba mean={nmean:9.4f} std={nstd:9.4f}")


if __name__ == "__main__":
    print("=== Benchmark ===")
    run_benchmark(n_test_values=[2000, 20000], n_steps=2000, dt=0.5)

    print("\n=== Validation ===")
    run_validation(n_test=50, n_steps_short=200, n_steps_medium=20000, dt=0.5)
