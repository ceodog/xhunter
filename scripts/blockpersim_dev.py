"""Dev-mode NumPy reference implementation of the "block-per-simulation" CUDA
ensemble design discussed for this project's GPU feasibility investigation.

NOT a GPU kernel. No CUDA toolkit or NVIDIA GPU exists in this environment
to compile/run one (checked directly: no nvcc, an Intel integrated GPU
only). Writing real CUDA here would be uncompiled, unvalidated code -- given
this project has found real, non-obvious bugs in every previous physics
port attempted (the Numba kernel's missing r0 factor and GM_sun unit bug;
the secular-theory alphabar coupling bug), untestable CUDA would almost
certainly contain one too, with no way to catch it.

What this IS: an executable specification of the exact per-block algorithm
a future CUDA kernel would need to implement -- one Python function per
"block" (one simulation), vectorized with NumPy across that simulation's
particles the way one CUDA block would vectorize across its threads. This
is testable and validatable against REBOUND right now, and any real CUDA
port should be checked against THIS reference, not re-derived from scratch.

Physics: democratic heliocentric coordinates -- confirmed via direct
inspection of REBOUND's own C source (5.0.1 tag) this session to be
REBOUND's actual implementation choice for this coordinate system, and the
natural fit for "particles drift independently around a fixed central
mass," as originally scoped in the CUDA proposal. Reuses the
ALREADY-VALIDATED universal-variable Kepler drift (_drift_one, from
bench_test_particle_integrator.py, itself checked against REBOUND to
1e-16-level energy conservation this session) for every particle's drift
step, massive and massless alike -- heliocentric two-body drift math
doesn't depend on the drifting particle's own mass, only on which GM it
drifts under.

NEW physics built here (never implemented in this project before --
worker.py's hybrid designs always deferred the massive bodies to real
REBOUND): the massive bodies' own mutual N-body dynamics. Two things
generalize beyond the existing test-particle-only kernel:
  1. A massive body's own Kepler drift uses GM = G*(M_sun + m_i), not
     G*M_sun alone -- a massless test particle (m_i=0) reduces to the same
     GM_SUN the existing kernel already assumed, so this is a strict
     generalization, not a divergent formula.
  2. Each massive body's kick includes mutual attraction from every OTHER
     massive body (direct term) plus the same heliocentric indirect-term
     correction already used for test particles (-G*m_j*r_j/|r_j|^3 --
     the Sun's own reflex motion under body j pulls every OTHER body's
     heliocentric frame too, massive or massless alike).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import GM_SUN, G, _drift_one  # noqa: E402
from planetx.constants import EARTH_MASS_IN_MSUN, GIANT_PLANETS  # noqa: E402

GIANT_NAMES = ("jupiter", "saturn", "uranus", "neptune")


def _mutual_kick(r: np.ndarray, m: np.ndarray, dt: float) -> np.ndarray:
    """r: [N,3] heliocentric positions, m: [N] masses (Msun). Returns
    velocity DELTA [N,3] for the full-dt kick each massive body receives
    from every OTHER massive body, in heliocentric coordinates (direct
    attraction minus each perturber's own indirect/Sun-reflex term)."""
    n = r.shape[0]
    dv = np.zeros_like(r)
    rho3 = np.sum(r * r, axis=1) ** 1.5  # |r_j|^3 for each body, reused as the indirect term's denominator
    for i in range(n):
        ax = ay = az = 0.0
        for j in range(n):
            if i == j:
                continue
            dx, dy, dz = r[j, 0] - r[i, 0], r[j, 1] - r[i, 1], r[j, 2] - r[i, 2]
            d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
            ax += G * m[j] * (dx / d3 - r[j, 0] / rho3[j])
            ay += G * m[j] * (dy / d3 - r[j, 1] / rho3[j])
            az += G * m[j] * (dz / d3 - r[j, 2] / rho3[j])
        dv[i] = (ax * dt, ay * dt, az * dt)
    return dv


def _test_particle_kick(r_t: np.ndarray, r_m: np.ndarray, m_m: np.ndarray, dt: float) -> np.ndarray:
    """Same direct-minus-indirect heliocentric kick, from n_massive bodies
    onto n_test massless particles (no massive<-test reaction, no
    test<->test interaction -- see worker.py's N_active convention)."""
    n_t = r_t.shape[0]
    n_m = r_m.shape[0]
    rho3 = np.sum(r_m * r_m, axis=1) ** 1.5
    dv = np.zeros_like(r_t)
    for p in range(n_m):
        dx = r_m[p, 0] - r_t[:, 0]
        dy = r_m[p, 1] - r_t[:, 1]
        dz = r_m[p, 2] - r_t[:, 2]
        d3 = (dx * dx + dy * dy + dz * dz) ** 1.5
        dv[:, 0] += G * m_m[p] * (dx / d3 - r_m[p, 0] / rho3[p])
        dv[:, 1] += G * m_m[p] * (dy / d3 - r_m[p, 1] / rho3[p])
        dv[:, 2] += G * m_m[p] * (dz / d3 - r_m[p, 2] / rho3[p])
    return dv * dt


def _drift_all(r: np.ndarray, v: np.ndarray, gm: np.ndarray, dt: float, n_iter: int) -> tuple[np.ndarray, np.ndarray]:
    """Drift every row of r,v by dt under its OWN gm (per-row -- massive
    bodies use G*(Msun+m_i), test particles all share GM_SUN)."""
    out_r = np.empty_like(r)
    out_v = np.empty_like(v)
    for i in range(r.shape[0]):
        out_r[i, 0], out_r[i, 1], out_r[i, 2], out_v[i, 0], out_v[i, 1], out_v[i, 2] = _drift_one(
            r[i, 0], r[i, 1], r[i, 2], v[i, 0], v[i, 1], v[i, 2], gm[i], dt, n_iter
        )
    return out_r, out_v


def run_one_block(
    r_m0: np.ndarray, v_m0: np.ndarray, m_m: np.ndarray,
    r_t0: np.ndarray, v_t0: np.ndarray,
    dt: float, n_steps: int, n_kepler_iter: int = 3,
) -> dict:
    """One "block" = one simulation. r_m0/v_m0 [n_massive,3], m_m [n_massive]
    (Msun); r_t0/v_t0 [n_test,3] (test particles, if any). All heliocentric.
    Democratic-heliocentric DKD, mirroring REBOUND's real step structure
    (confirmed via source inspection): drift(dt/2) all -> mutual kick(dt)
    among massive bodies -> test-particle kick(dt) from the now-drifted
    massive positions -> drift(dt/2) all.
    """
    gm_massive = G * (1.0 + m_m)  # GM_sun + m_i per massive body
    r_m, v_m = r_m0.copy(), v_m0.copy()
    r_t, v_t = r_t0.copy(), v_t0.copy()
    half = dt / 2.0
    gm_test = np.full(r_t.shape[0], GM_SUN)

    for _ in range(n_steps):
        r_m, v_m = _drift_all(r_m, v_m, gm_massive, half, n_kepler_iter)
        if r_t.shape[0]:
            r_t, v_t = _drift_all(r_t, v_t, gm_test, half, n_kepler_iter)

        v_m = v_m + _mutual_kick(r_m, m_m, dt)
        if r_t.shape[0]:
            v_t = v_t + _test_particle_kick(r_t, r_m, m_m, dt)

        r_m, v_m = _drift_all(r_m, v_m, gm_massive, half, n_kepler_iter)
        if r_t.shape[0]:
            r_t, v_t = _drift_all(r_t, v_t, gm_test, half, n_kepler_iter)

    return {"r_m": r_m, "v_m": v_m, "r_t": r_t, "v_t": v_t}


def run_ensemble(sims: list[dict], dt: float, n_steps: int) -> list[dict]:
    """sims: list of {"r_m0","v_m0","m_m","r_t0","v_t0"} dicts -- one per
    independent simulation ("one grid of blocks"). Sequential Python loop
    here since correctness, not throughput, is dev mode's job -- each
    simulation is already independent by construction (same guarantee real
    ensemble-parallel CUDA execution would rely on), so proving that N of
    them can run "at once" isn't a Python-level correctness question."""
    return [
        run_one_block(s["r_m0"], s["v_m0"], s["m_m"], s["r_t0"], s["v_t0"], dt, n_steps)
        for s in sims
    ]
