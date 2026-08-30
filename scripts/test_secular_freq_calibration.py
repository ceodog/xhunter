"""Lever 1 from the accuracy-improvement discussion: empirical eigenfrequency
calibration. Hypothesis (grounded in measurements already made this
session): the population-level phase-error degradation over time is
dominated by eigenfrequency error in the theoretical Laplace-Lagrange
matrix, not a structural flaw in the forced+free formula itself -- Step 1's
massive-body validation found systematic 12-15% precession-rate errors for
Jupiter/Saturn/Uranus and real multi-mode phase mismatch for Neptune, and
phase error compounds as Delta-frequency * time (already established via
this project's own chaos-sensitivity test), which would be invisible at the
2e6-2e7 yr windows tested so far but catastrophic at the 4.5e9 yr production
target.

Approach: replace the theoretical eigenvalues/eigenvectors of the massive-
body secular matrix with values FIT directly to a short (cheap) N-body
reference run, using variable projection (Golub & Pereyra 1973): for
candidate frequencies omega_l, the per-(body,mode) complex amplitudes are a
LINEAR least-squares problem (closed form via lstsq); only the 5 frequencies
themselves are optimized nonlinearly (scipy.optimize.least_squares), with
the theoretical eigenvalues as the initial guess. This is a simplified,
single-fit-per-mode version of Laskar (1990)'s frequency analysis technique
("NAFF"), not the full iterative algorithm -- adequate for testing whether
frequency accuracy is really the bottleneck before investing in more.

Two-stage test:
  1. Fit calibrated (omega, C) from a SHORT (3e6 yr) N-body run -- cheap.
  2. Test GENERALIZATION at the same 2e7 yr population-level benchmark
     already used in test_secular_statistical.py / test_secular_allparticles.py
     (non-resonant disk particles), comparing calibrated-frequency secular
     predictions against the theoretical-frequency baseline's already-known
     numbers (e correlation r=0.41, i correlation r=0.45 at 2e7 yr).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound
from scipy import stats
from scipy.optimize import least_squares

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import GM_SUN  # noqa: E402
from test_secular_multibody import (  # noqa: E402
    build_AB_matrices, eigen_solution, hk_from_elements, elements_from_hk,
    pq_from_elements, elements_from_pq, particle_coupling,
)
from test_secular_step34 import resonance_width_coefficient, resonance_half_width  # noqa: E402
from planetx.constants import GIANT_PLANETS, EARTH_MASS_IN_MSUN  # noqa: E402


names = ["jupiter", "saturn", "uranus", "neptune"]
a_arr = np.array([GIANT_PLANETS[n]["a"] for n in names] + [400.0])
e_arr = np.array([GIANT_PLANETS[n]["e"] for n in names] + [0.3])
inc_arr = np.array([GIANT_PLANETS[n]["inc"] for n in names] + [20.0])
Omega_arr = np.array([GIANT_PLANETS[n]["Omega"] for n in names] + [0.0])
omega_arr = np.array([GIANT_PLANETS[n]["omega"] for n in names] + [0.0])
m_arr = np.array([GIANT_PLANETS[n]["m"] for n in names] + [5.0 * EARTH_MASS_IN_MSUN])
names5 = names + ["hpx"]

# ---------------------------------------------------------------------------
# Stage 1: short N-body calibration run + variable-projection frequency fit
# ---------------------------------------------------------------------------

T_CALIB = 6.0e7  # covers ~4 cycles of HPX's own ~15 Myr mode (theoretical
# g_HPX ~ 0.024 deg/Myr -> period ~15e6 yr); 3e6 yr (the original guess,
# matching Step 1's validation baseline) covered only ~20% of one cycle and
# caused the fit to run away to a spurious high frequency for that mode --
# a fundamental least-squares frequency-resolution limit, not a bug.
N_SNAP_CALIB = 2000


def run_massive_body_calibration(T_total, n_snap):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    for i, name in enumerate(names5):
        sim.add(m=m_arr[i], a=a_arr[i], e=e_arr[i], inc=np.radians(inc_arr[i]),
                Omega=np.radians(Omega_arr[i]), omega=np.radians(omega_arr[i]), M=0.0, name=name)
    sim.N_active = len(sim.particles)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.integrator.safe_mode = 0
    sim.dt = min(0.5, a_arr[0] ** 1.5 / 20.0)

    snap_dt = T_total / n_snap
    times = np.arange(1, n_snap + 1) * snap_dt
    h = np.empty((5, n_snap)); k = np.empty((5, n_snap))
    p = np.empty((5, n_snap)); q = np.empty((5, n_snap))
    for s in range(n_snap):
        sim.integrate((s + 1) * snap_dt)
        for b in range(5):
            o = sim.particles[1 + b].orbit(primary=sim.particles[0])
            pomega = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360
            h[b, s], k[b, s] = hk_from_elements(o.e, pomega)
            p[b, s], q[b, s] = pq_from_elements(np.degrees(o.inc), np.degrees(o.Omega) % 360)
    return times, h + 1j * k, p + 1j * q


def fit_calibrated_modes(times, z_meas, omega0):
    """z_meas: [5 bodies, n_snap] complex. omega0: [5] initial frequency
    guess (theoretical eigenvalues). Returns (omega_fit [5], C_fit [5,5]
    complex, body x mode) via variable projection: for fixed omega, C is a
    linear least-squares fit (closed form); only omega is optimized
    nonlinearly."""
    n_modes = len(omega0)

    def build_design(omega):
        # [n_snap, n_modes] basis matrix, shared across all 5 bodies
        return np.exp(-1j * np.outer(times, omega))

    def amplitudes_for(omega):
        D = build_design(omega)  # [n_snap, n_modes]
        C = np.empty((5, n_modes), dtype=complex)
        for b in range(5):
            C[b], *_ = np.linalg.lstsq(D, z_meas[b], rcond=None)
        return C

    def residuals(omega):
        C = amplitudes_for(omega)
        D = build_design(omega)
        resid = []
        for b in range(5):
            pred = D @ C[b]
            r = z_meas[b] - pred
            resid.extend([r.real, r.imag])
        return np.concatenate(resid)

    result = least_squares(residuals, omega0, method="lm", max_nfev=2000)
    omega_fit = result.x
    C_fit = amplitudes_for(omega_fit)
    return omega_fit, C_fit, result


print(f"Running {T_CALIB:.0e} yr massive-body calibration N-body run ({N_SNAP_CALIB} snapshots)...")
t0 = time.perf_counter()
times_calib, z_e_meas, z_i_meas = run_massive_body_calibration(T_CALIB, N_SNAP_CALIB)
print(f"  done in {time.perf_counter()-t0:.2f}s")

A5, B5 = build_AB_matrices(a_arr, m_arr)
g_theory, Ve_theory = eigen_solution(A5)
f_theory, Vi_theory = eigen_solution(B5)

print("\nFitting calibrated eccentricity/apsidal modes (variable projection, init at theory)...")
g_fit, Ce_fit, res_e = fit_calibrated_modes(times_calib, z_e_meas, g_theory.real.copy())
print(f"  theoretical g (deg/Myr): {np.degrees(g_theory.real)*1e6}")
print(f"  calibrated  g (deg/Myr): {np.degrees(g_fit)*1e6}")
print(f"  residual norm: theory-init cost -> fit cost = {res_e.cost:.6e} (fit converged: {res_e.success})")

print("\nFitting calibrated inclination/node modes...")
f_fit, Ci_fit, res_i = fit_calibrated_modes(times_calib, z_i_meas, f_theory.real.copy())
print(f"  theoretical f (deg/Myr): {np.degrees(f_theory.real)*1e6}")
print(f"  calibrated  f (deg/Myr): {np.degrees(f_fit)*1e6}")
print(f"  fit converged: {res_i.success}")


# quick in-sample check: does the calibrated fit actually track the measured
# series better than the theoretical prediction, on the SAME calibration window?
def reconstruct(times, omega, C):
    D = np.exp(-1j * np.outer(times, omega))
    return (D @ C.T).T  # [5, n_snap]


# theoretical reconstruction: c = solve(V, z0), z(t) = V @ (c * exp(-i g t))
h0, k0 = hk_from_elements(e_arr, Omega_arr + omega_arr)
z0_e = h0 + 1j * k0
c0 = np.linalg.solve(Ve_theory, z0_e)
z_e_theory_pred = Ve_theory @ (c0[:, None] * np.exp(-1j * np.outer(g_theory, times_calib)))
z_e_fit_pred = reconstruct(times_calib, g_fit, Ce_fit)

rmse_theory = np.sqrt(np.mean(np.abs(z_e_meas - z_e_theory_pred) ** 2))
rmse_fit = np.sqrt(np.mean(np.abs(z_e_meas - z_e_fit_pred) ** 2))
print(f"\nIn-sample eccentricity-vector RMSE: theoretical={rmse_theory:.5f}  calibrated={rmse_fit:.5f}  "
      f"({'IMPROVED' if rmse_fit < rmse_theory else 'NOT improved'})")

np.savez(
    Path(__file__).parent / "_secular_calibration_cache.npz",
    g_theory=g_theory, f_theory=f_theory, g_fit=g_fit, f_fit=f_fit,
    Ce_fit=Ce_fit, Ci_fit=Ci_fit, a_arr=a_arr, m_arr=m_arr,
)
print("\nSaved calibrated modes to scripts/_secular_calibration_cache.npz for the generalization test.")
