"""Diagnostic #3: does the single-perturber forced+free phase error found in
test_secular_mean_element_check.py scale as e^2 (the textbook-expected
signature of a genuine linear/first-order-in-e theory limitation), or does
it fail to shrink with e (pointing to an actual bug rather than a
fundamental theory limit)?

Reports the ABSOLUTE eccentricity-vector error |z_meas - z_pred| (z=h+ik),
not just the derived angle pomega -- pomega = atan2(h,k) is ill-conditioned
at small e (a fixed absolute h,k error produces a LARGER angular error as
e shrinks, since angular resolution ~ |error|/e), so scaling behavior must
be read off the well-conditioned complex-vector error, not the angle.
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
    particle_forced_free_hk,
)

a_neptune, m_neptune = 30.06992276, 51.51389e-6
e_neptune, inc_neptune = 0.00859048, 1.77004347
Omega_neptune, omega_neptune = 0.0, 0.0

a_test = 45.0
Omega_test, omega_test = 20.0, 40.0
P_test = a_test ** 1.5

A1, B1 = build_AB_matrices(np.array([a_neptune]), np.array([m_neptune]))
g1, Ve1 = eigen_solution(A1)
h0n, k0n = hk_from_elements(e_neptune, Omega_neptune + omega_neptune)
z0_massive = np.array([h0n + 1j * k0n])


def run_case(e_test, T_total):
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    sim.add(m=m_neptune, a=a_neptune, e=e_neptune, inc=np.radians(inc_neptune),
            Omega=0.0, omega=0.0, M=0.0, name="neptune")
    sim.N_active = 2
    sim.add(m=0.0, a=a_test, e=e_test, inc=np.radians(5.0),
            Omega=np.radians(Omega_test), omega=np.radians(omega_test), M=0.0)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.integrator.safe_mode = 0
    sim.dt = min(0.5, (a_neptune ** 1.5) / 20.0)

    t0 = time.perf_counter()
    sim.integrate(T_total)
    elapsed = time.perf_counter() - t0
    o = sim.particles[2].orbit(primary=sim.particles[0])
    pomega_meas = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360
    h_meas, k_meas = hk_from_elements(o.e, pomega_meas)
    z_meas = h_meas + 1j * k_meas

    h0p, k0p = hk_from_elements(e_test, Omega_test + omega_test)
    zt = particle_forced_free_hk(a_test, np.array([a_neptune]), np.array([m_neptune]),
                                  g1, Ve1, z0_massive, h0p, k0p, np.array([T_total]))
    z_pred = zt

    z_err = abs(z_meas - z_pred)
    return z_err, o.e, elapsed


print(f"{'e_test':>8s} {'T=1e6 yr |z_err|':>18s} {'|z_err|/e^2':>14s} "
      f"{'T=2e7 yr |z_err|':>18s} {'|z_err|/e^2':>14s}")

e_values = [0.10, 0.03, 0.01, 0.003]
results = {}
for e_test in e_values:
    err_short, _, t_short = run_case(e_test, 1.0e6)
    err_long, _, t_long = run_case(e_test, 2.0e7)
    results[e_test] = (err_short, err_long)
    print(f"{e_test:8.3f} {err_short:18.6f} {err_short/e_test**2:14.4f} "
          f"{err_long:18.6f} {err_long/e_test**2:14.4f}")

print("\nIf |z_err|/e^2 is roughly CONSTANT across rows -> consistent with a genuine "
      "O(e^2) linear-theory limitation (error shrinks quadratically with e, as expected).")
print("If |z_err| stays roughly the SAME (or shrinks much slower than e^2) as e_test "
      "shrinks -> points to a bug/offset independent of e, not a fundamental theory limit.")

print("\nRatio check (each step e_test/3, expect |z_err| ratio ~9x if O(e^2)):")
for i in range(len(e_values) - 1):
    e_hi, e_lo = e_values[i], e_values[i + 1]
    for label, idx in [("T=1e6", 0), ("T=2e7", 1)]:
        r = results[e_hi][idx] / max(results[e_lo][idx], 1e-15)
        print(f"  {label}: |z_err|({e_hi})/|z_err|({e_lo}) = {r:.2f}  (expect ~{(e_hi/e_lo)**2:.2f} if O(e^2))")
