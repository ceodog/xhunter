"""Tests the secular-perturbation-theory idea floated in the plan doc
(/Users/xtan/.claude/plans/robust-puzzling-flamingo.md, "A bigger, riskier
idea worth naming") as a possible massive speedup for the non-resonant
majority of the disk: classical (Laplace-Lagrange) secular theory averages
over the fast orbital motion and propagates only the slow precession of
(e, i, pomega, Omega) directly, skipping the orbital-period timescale
entirely.

Scope of this test, deliberately narrow: single massless test particle,
single massive perturber (Neptune alone -- the astronomically relevant one
for the 30-100 AU disk), at a semi-major axis safely away from any low-order
mean-motion resonance (so linear secular theory's validity assumption
actually holds -- it explicitly does NOT apply near resonance, per the
plan's own caveat). This tests only the best-established, most robust
prediction of the theory -- the free apsidal/nodal precession RATE -- not
the full forced+free amplitude solution (whose exact coefficients I don't
trust from memory after the Kepler-solver bug found earlier this session;
testing the rate alone is checkable without needing those).

Formulas (Murray & Dermott, "Solar System Dynamics", ch. 7): for a massless
test particle exterior to one perturber of mass m_p at a_p (alpha=a_p/a_test
< 1), linear secular theory gives uniform precession rates independent of
the test particle's own e, i:

  d(pomega)/dt = +n * (1/4) * (m_p/M_sun) * alpha * b_(3/2)^(1)(alpha)
  d(Omega)/dt  = -n * (1/4) * (m_p/M_sun) * alpha * b_(3/2)^(2)(alpha)

where n is the test particle's own mean motion and b_s^(j) is the Laplace
coefficient, computed here by direct numerical quadrature (not a memorized
series expansion) to avoid repeating the earlier session's mistake of
trusting a half-remembered formula.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import GM_SUN, build_massive_sim  # noqa: E402


def laplace_b(s, j, alpha, n_points=200000):
    """b_s^(j)(alpha) = (1/pi) * integral_0^{2pi} cos(j*psi) / (1 - 2*alpha*cos(psi) + alpha^2)^s dpsi
    Computed by direct numerical quadrature (composite Simpson's rule via
    numpy, not scipy, to avoid adding a dependency for a one-off diagnostic
    script) rather than a memorized series expansion -- avoids repeating the
    earlier session's mistake of trusting a half-remembered coefficient
    formula without checking it.
    """
    psi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    integrand = np.cos(j * psi) / (1 - 2 * alpha * np.cos(psi) + alpha**2) ** s
    # composite trapezoidal rule over a full period is exact for smooth
    # periodic functions to spectral accuracy (equivalent to the periodic
    # trapezoidal/midpoint rule's well-known super-convergence)
    val = np.trapz(np.append(integrand, integrand[0]), dx=2 * np.pi / n_points)
    return val / np.pi


def predicted_rates(a_test, a_p, m_p_msun, gm_sun=GM_SUN):
    n_test = np.sqrt(gm_sun / a_test**3)  # rad/yr
    alpha = a_p / a_test
    b1 = laplace_b(1.5, 1, alpha)
    b2 = laplace_b(1.5, 2, alpha)
    d_pomega = n_test * 0.25 * m_p_msun * alpha * b1
    d_Omega = -n_test * 0.25 * m_p_msun * alpha * b2
    return d_pomega, d_Omega, alpha, b1, b2


def measure_rebound_rates(a_test, e_test, inc_test, a_p, m_p_msun, e_p, inc_p, T_TOTAL, n_snap):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    sim.add(m=m_p_msun, a=a_p, e=e_p, inc=inc_p, Omega=0.0, omega=0.0, M=0.0, name="perturber")
    sim.N_active = 2
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.integrator.safe_mode = 0
    sim.dt = min(0.5, (a_p**1.5) / 20.0)
    sim.add(m=0.0, a=a_test, e=e_test, inc=inc_test, Omega=np.radians(20.0), omega=np.radians(40.0), M=0.0)

    snap_dt = T_TOTAL / n_snap
    pomega_hist = np.empty(n_snap)
    Omega_hist = np.empty(n_snap)
    t0 = time.perf_counter()
    for s in range(n_snap):
        sim.integrate((s + 1) * snap_dt)
        o = sim.particles[2].orbit(primary=sim.particles[0])
        Omega_hist[s] = np.degrees(o.Omega) % 360
        pomega_hist[s] = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360
    elapsed = time.perf_counter() - t0

    times = np.arange(1, n_snap + 1) * snap_dt
    pomega_unwrapped = np.unwrap(np.radians(pomega_hist))
    Omega_unwrapped = np.unwrap(np.radians(Omega_hist))
    # linear fit: rate = slope
    d_pomega_measured = np.polyfit(times, pomega_unwrapped, 1)[0]
    d_Omega_measured = np.polyfit(times, Omega_unwrapped, 1)[0]
    return d_pomega_measured, d_Omega_measured, elapsed


if __name__ == "__main__":
    # Neptune, from constants.GIANT_PLANETS
    a_p, m_p, e_p, inc_p = 30.06992276, 51.51389e-6, 0.00859048, np.radians(1.77004347)

    # a=45 AU: safely clear of the 3:2 (39.4) and 2:1 (47.8) Neptune MMRs
    a_test, e_test, inc_test = 45.0, 0.10, np.radians(5.0)

    d_pomega_pred, d_Omega_pred, alpha, b1, b2 = predicted_rates(a_test, a_p, m_p)
    print(f"alpha={alpha:.4f}  b_3/2^(1)={b1:.4f}  b_3/2^(2)={b2:.4f}")
    print(f"predicted: d(pomega)/dt = {np.degrees(d_pomega_pred)*1e6:.4f} deg/Myr  "
          f"(period {360/np.degrees(d_pomega_pred):.0f} yr)")
    print(f"predicted: d(Omega)/dt  = {np.degrees(d_Omega_pred)*1e6:.4f} deg/Myr  "
          f"(period {360/abs(np.degrees(d_Omega_pred)):.0f} yr)")

    # A full precession period here is ~5.7 Myr -- too long to fully cycle
    # in this session. Use a shorter window (still many orbital periods,
    # ~15000, so short-period jitter averages out over many snapshots) and
    # fit the rate via linear regression rather than requiring a full cycle.
    T_TOTAL = 1.0e6
    print(f"\nintegrating REBOUND for {T_TOTAL:.0f} yr "
          f"({T_TOTAL/abs(360/np.degrees(d_pomega_pred)):.2f} predicted precession periods, "
          f"~{T_TOTAL/(a_test**1.5):.0f} orbital periods)...")
    d_pomega_meas, d_Omega_meas, elapsed = measure_rebound_rates(
        a_test, e_test, inc_test, a_p, m_p, e_p, inc_p, T_TOTAL, n_snap=300
    )
    print(f"REBOUND N-body took {elapsed:.2f}s wall-clock")
    print(f"measured:  d(pomega)/dt = {np.degrees(d_pomega_meas)*1e6:.4f} deg/Myr")
    print(f"measured:  d(Omega)/dt  = {np.degrees(d_Omega_meas)*1e6:.4f} deg/Myr")

    pomega_err = abs(d_pomega_meas - d_pomega_pred) / abs(d_pomega_pred) * 100
    Omega_err = abs(d_Omega_meas - d_Omega_pred) / abs(d_Omega_pred) * 100
    print(f"\nrelative error: pomega rate {pomega_err:.2f}%,  Omega rate {Omega_err:.2f}%")

    print(f"\nsecular formula evaluation: effectively instantaneous (closed-form + one "
          f"numerical quadrature for the Laplace coefficient, microseconds)")
    print(f"N-body integration for the same physical time: {elapsed:.2f}s "
          f"(single perturber, single particle -- would scale linearly with particle count)")
