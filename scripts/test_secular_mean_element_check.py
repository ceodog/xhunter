"""Diagnostic #1 from the "how to find the secular method's error source"
discussion: does comparing N-body's OSCULATING elements at a single instant
(what every validation this session has done) against the secular theory's
MEAN-element prediction inflate the apparent error, versus averaging N-body's
elements over a window (removing short-period oscillation) before comparing?

Also folds in diagnostic #4 (does the simplest possible case -- single
perturber, no multi-mode coupling -- still show degradation at the longer
2e7 yr timescale where the full 5-body case broke, or was that specifically
a multi-body coupling artifact?) since it's nearly free to test alongside #1.

Setup: same single-perturber (Neptune only) test particle as the ORIGINAL
validated spot-check (test_secular_theory.py: a_test=45, e_test=0.10,
i_test=5deg -- clear of low-order Neptune MMRs), but using the full
forced+free AMPLITUDE solution (test_secular_multibody.py's machinery,
reused with a 1-element massive-body array -- Neptune alone has zero
self-coupling in the N-body secular matrix, so its forced term is
predicted to be a constant, not oscillating -- the general machinery
handles this correctly with no special-casing needed), not just the
rate-only check the original spot-check used.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from test_secular_multibody import (  # noqa: E402
    build_AB_matrices, eigen_solution, hk_from_elements, elements_from_hk,
    pq_from_elements, elements_from_pq, particle_forced_free_hk, particle_forced_free_pq,
)


a_neptune, m_neptune = 30.06992276, 51.51389e-6
e_neptune, inc_neptune = 0.00859048, 1.77004347
Omega_neptune, omega_neptune = 0.0, 0.0  # matches the original spot-check's Neptune setup

a_test, e_test, inc_test = 45.0, 0.10, 5.0
Omega_test, omega_test = 20.0, 40.0
P_test = a_test ** 1.5  # yr, ~301.9

A1, B1 = build_AB_matrices(np.array([a_neptune]), np.array([m_neptune]))
g1, Ve1 = eigen_solution(A1)
f1, Vi1 = eigen_solution(B1)
h0n, k0n = hk_from_elements(e_neptune, Omega_neptune + omega_neptune)
z0_massive = np.array([h0n + 1j * k0n])
p0n, q0n = pq_from_elements(inc_neptune, Omega_neptune)
w0_massive = np.array([p0n + 1j * q0n])

print(f"1-body massive eigensystem (should be trivial/zero, Neptune alone doesn't self-precess "
      f"under this linear theory): g={g1}, f={f1}")


def run_case(T_total, window_periods=30, n_window=150):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    sim.add(m=m_neptune, a=a_neptune, e=e_neptune, inc=np.radians(inc_neptune),
            Omega=0.0, omega=0.0, M=0.0, name="neptune")
    sim.N_active = 2
    sim.add(m=0.0, a=a_test, e=e_test, inc=np.radians(inc_test),
            Omega=np.radians(Omega_test), omega=np.radians(omega_test), M=0.0)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.integrator.safe_mode = 0
    sim.dt = min(0.5, (a_neptune ** 1.5) / 20.0)

    window = window_periods * P_test
    t_start = T_total - window
    assert t_start > 0
    t0 = time.perf_counter()
    sim.integrate(t_start)
    ts = np.linspace(t_start, T_total, n_window)
    h_series = np.empty(n_window); k_series = np.empty(n_window)
    p_series = np.empty(n_window); q_series = np.empty(n_window)
    for i, t in enumerate(ts):
        sim.integrate(t)
        o = sim.particles[2].orbit(primary=sim.particles[0])
        pomega = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360
        h_series[i], k_series[i] = hk_from_elements(o.e, pomega)
        p_series[i], q_series[i] = pq_from_elements(np.degrees(o.inc), np.degrees(o.Omega) % 360)
    elapsed = time.perf_counter() - t0

    # osculating: the single instant at exactly T_total (last sample)
    e_osc, pomega_osc = elements_from_hk(h_series[-1], k_series[-1])
    i_osc, Omega_osc = elements_from_pq(p_series[-1], q_series[-1])

    # windowed mean: average h,k / p,q over the window, THEN convert back
    e_mean, pomega_mean = elements_from_hk(h_series.mean(), k_series.mean())
    i_mean, Omega_mean = elements_from_pq(p_series.mean(), q_series.mean())

    # theoretical secular (mean-element, by construction) prediction at T_total
    h0p, k0p = hk_from_elements(e_test, Omega_test + omega_test)
    zt = particle_forced_free_hk(a_test, np.array([a_neptune]), np.array([m_neptune]),
                                  g1, Ve1, z0_massive, h0p, k0p, np.array([T_total]))
    e_pred, pomega_pred = elements_from_hk(zt.real, zt.imag)
    p0p, q0p = pq_from_elements(inc_test, Omega_test)
    wt = particle_forced_free_pq(a_test, np.array([a_neptune]), np.array([m_neptune]),
                                  f1, Vi1, w0_massive, p0p, q0p, np.array([T_total]))
    i_pred, Omega_pred = elements_from_pq(wt.real, wt.imag)

    print(f"\n=== T_total={T_total:.1e} yr ({T_total/P_test:.0f} orbital periods) -- N-body: {elapsed:.2f}s ===")
    print(f"predicted (secular, mean by construction): e={e_pred:.5f}  pomega={pomega_pred:.3f}  "
          f"i={i_pred:.4f}  Omega={Omega_pred:.3f}")
    print(f"osculating (single instant, t=T_total):     e={e_osc:.5f}  pomega={pomega_osc:.3f}  "
          f"i={i_osc:.4f}  Omega={Omega_osc:.3f}")
    print(f"windowed mean ({window_periods} periods, n={n_window}): e={e_mean:.5f}  pomega={pomega_mean:.3f}  "
          f"i={i_mean:.4f}  Omega={Omega_mean:.3f}")

    err_e_osc = abs(e_osc - e_pred)
    err_e_mean = abs(e_mean - e_pred)
    err_pomega_osc = min(abs(pomega_osc - pomega_pred), 360 - abs(pomega_osc - pomega_pred))
    err_pomega_mean = min(abs(pomega_mean - pomega_pred), 360 - abs(pomega_mean - pomega_pred))
    err_i_osc = abs(i_osc - i_pred)
    err_i_mean = abs(i_mean - i_pred)
    err_Omega_osc = min(abs(Omega_osc - Omega_pred), 360 - abs(Omega_osc - Omega_pred))
    err_Omega_mean = min(abs(Omega_mean - Omega_pred), 360 - abs(Omega_mean - Omega_pred))

    print(f"\n|error| vs prediction:                    osculating   windowed-mean   improvement")
    print(f"  e:                                        {err_e_osc:.5f}      {err_e_mean:.5f}       "
          f"{100*(1-err_e_mean/max(err_e_osc,1e-12)):+.1f}%")
    print(f"  pomega (deg):                              {err_pomega_osc:.3f}       {err_pomega_mean:.3f}        "
          f"{100*(1-err_pomega_mean/max(err_pomega_osc,1e-12)):+.1f}%")
    print(f"  i (deg):                                   {err_i_osc:.4f}      {err_i_mean:.4f}       "
          f"{100*(1-err_i_mean/max(err_i_osc,1e-12)):+.1f}%")
    print(f"  Omega (deg):                                {err_Omega_osc:.3f}       {err_Omega_mean:.3f}        "
          f"{100*(1-err_Omega_mean/max(err_Omega_osc,1e-12)):+.1f}%")


print("\n" + "=" * 70)
print("CASE 1: short timescale (1e6 yr) -- matches the original successful spot-check")
run_case(1.0e6)

print("\n" + "=" * 70)
print("CASE 2: long timescale (2e7 yr) -- matches where the full 5-body case degraded")
run_case(2.0e7)
