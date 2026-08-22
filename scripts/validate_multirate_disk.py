"""Validates the multi-rate disk-integration idea from the plan
(/Users/xtan/.claude/plans/robust-puzzling-flamingo.md): can the disk kernel
in bench_test_particle_integrator.py use a coarser dt_disk than the massive
bodies' dt (0.5 yr, set by Jupiter's period) without corrupting mean-motion
resonance capture with Neptune -- specifically the 3:2 (Plutino) resonance
at a ~ 39.4 AU, the resonance actually relevant to this project's disk
(disk_inner_edge/disk_outer_edge span 30-100 AU in prior.yaml).

Method, in order:
  1. Scan a handful of test particles near a=39.4 AU using REBOUND alone
     (dt=0.5 yr, matching worker.py) -- ground truth. Classify each as
     librating or circulating via the resonant angle
     phi = 3*lambda_particle - 2*lambda_Neptune - pomega_particle.
  2. For any particle REBOUND shows librating, re-run the SAME initial
     condition through the hybrid kernel at dt_disk=0.5 (sanity check --
     must reproduce REBOUND closely) and at the proposed coarser
     dt_disk = P_Neptune/20 ~= 8.25 yr (the actual thing being tested).
  3. Compare libration center/amplitude between all three. If the coarser
     kernel still librates with a similar center/amplitude, the multi-rate
     idea is validated for this resonance; if it circulates or the
     amplitude is grossly different, it isn't safe as proposed.

This only tests the 3:2 -- other Neptune resonances (2:1 at ~47.8 AU, etc.)
would need the same check before trusting dt_disk=8.25 generally across the
whole 30-100 AU disk range.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import (  # noqa: E402
    GM_SUN, G, build_massive_sim, drive_and_extract_mid, elements_from_state,
    native_dt_for, propagate,
)


def mean_longitude(Omega, omega, M):
    return (Omega + omega + M) % 360.0


def pomega(Omega, omega):
    return (Omega + omega) % 360.0


def resonant_angle_32(p_Omega, p_omega, p_M, n_Omega, n_omega, n_M):
    """phi = 3*lambda_p - 2*lambda_N - pomega_p, for the 3:2 interior MMR
    with Neptune (the Plutino resonance). Returns degrees, wrapped to
    [-180, 180)."""
    lam_p = mean_longitude(p_Omega, p_omega, p_M)
    lam_n = mean_longitude(n_Omega, n_omega, n_M)
    pom_p = pomega(p_Omega, p_omega)
    phi = 3 * lam_p - 2 * lam_n - pom_p
    return (phi + 180.0) % 360.0 - 180.0


def libration_span(phi_series_deg):
    """Unwrap the angle series and return its total range in degrees.
    Circulating angles drift by >> 360 deg over a long run; librating
    angles stay confined to a bounded range."""
    unwrapped = np.degrees(np.unwrap(np.radians(phi_series_deg)))
    return unwrapped.max() - unwrapped.min(), unwrapped


A_RES_32 = 30.06992276 * (1.5) ** (2.0 / 3.0)  # ~39.41 AU, exact Neptune a from GIANT_PLANETS
P_NEPTUNE = 30.06992276**1.5  # ~164.9 yr
DT_MASSIVE = 0.5
# Rounded to 8.0 yr (from the raw P_Neptune/20 ~= 8.245 yr) so it's an exact
# multiple of 2*DT_MASSIVE=1.0 -- required by native_dt_for() below so
# REBOUND is never asked to integrate() to an off-grid time (see root-cause
# note in bench_test_particle_integrator.native_dt_for's docstring).
DT_DISK_COARSE = 8.0

print(f"3:2 resonance location: a = {A_RES_32:.3f} AU")
print(f"P_Neptune = {P_NEPTUNE:.2f} yr, proposed dt_disk = P_Neptune/20 = {DT_DISK_COARSE:.3f} yr "
      f"({DT_DISK_COARSE/DT_MASSIVE:.1f}x current dt_disk={DT_MASSIVE})\n")


# ---------------------------------------------------------------------------
# Step 1: REBOUND-native scan for a librating particle (ground truth)
# ---------------------------------------------------------------------------

HPX = (5.0, 400.0, 0.3, 20.0)
N_SNAP = 500
SNAP_DT = 400.0  # yr between snapshots
T_TOTAL = N_SNAP * SNAP_DT  # 200,000 yr -- several expected Plutino libration periods

print(f"=== Step 1: REBOUND-native scan near a={A_RES_32:.2f} AU, "
      f"{N_SNAP} snapshots over {T_TOTAL:.0f} yr ===")

rng = np.random.default_rng(11)
n_scan = 16
a0 = rng.uniform(A_RES_32 - 0.5, A_RES_32 + 0.5, n_scan)
e0 = np.full(n_scan, 0.20)
inc0 = np.radians(rng.uniform(2, 10, n_scan))
Omega0 = rng.uniform(0, 360, n_scan)
omega0 = rng.uniform(0, 360, n_scan)
M0 = rng.uniform(0, 360, n_scan)

sim = build_massive_sim(*HPX, dt_massive=DT_MASSIVE)
neptune_idx = 4  # sun=0, jupiter=1, saturn=2, uranus=3, neptune=4
for i in range(n_scan):
    sim.add(m=0.0, a=a0[i], e=e0[i], inc=inc0[i], Omega=np.radians(Omega0[i]),
            omega=np.radians(omega0[i]), M=np.radians(M0[i]))
first_tp = 6  # index of particle 0 in the scan: sun(0)+4 giants(1-4)+hpx(5) = 6 particles, so first test particle is index 6

phi_hist = np.empty((N_SNAP, n_scan))
for s in range(N_SNAP):
    sim.integrate((s + 1) * SNAP_DT)
    n_orb = sim.particles[neptune_idx].orbit(primary=sim.particles[0])
    for i in range(n_scan):
        p_orb = sim.particles[first_tp + i].orbit(primary=sim.particles[0])
        phi_hist[s, i] = resonant_angle_32(
            np.degrees(p_orb.Omega) % 360, np.degrees(p_orb.omega) % 360, np.degrees(p_orb.M) % 360,
            np.degrees(n_orb.Omega) % 360, np.degrees(n_orb.omega) % 360, np.degrees(n_orb.M) % 360,
        )

librating = []
for i in range(n_scan):
    span, unwrapped = libration_span(phi_hist[:, i])
    status = "LIBRATING" if span < 300 else "circulating"
    print(f"  particle {i:2d}  a0={a0[i]:.3f}  span={span:7.1f} deg  center={unwrapped.mean():7.1f}  -> {status}")
    if span < 300:
        librating.append(i)

if not librating:
    print("\nNo librating particle found in this scan -- widen the a0 range or "
          "increase n_scan/T_TOTAL and retry. Stopping here.")
    sys.exit(0)

spans = {i: libration_span(phi_hist[:, i])[0] for i in librating}
pick = min(spans, key=spans.get)  # tightest/cleanest libration case
print(f"\nUsing particle {pick} (a0={a0[pick]:.3f} AU, e0={e0[pick]:.3f}) as the test case for step 2.")
ref_span, ref_unwrapped = libration_span(phi_hist[:, pick])
print(f"REBOUND reference: libration span={ref_span:.1f} deg, center={ref_unwrapped.mean():.1f} deg")


# ---------------------------------------------------------------------------
# Step 2: run the SAME initial condition through the hybrid kernel at
# dt_disk=0.5 (sanity check vs. REBOUND) and dt_disk=P_Neptune/20 (the
# proposed multi-rate value), chunked at the same ~400 yr cadence.
# ---------------------------------------------------------------------------

print(f"\n=== Step 2: hybrid kernel at dt_disk={DT_MASSIVE} (sanity) "
      f"and dt_disk={DT_DISK_COARSE:.3f} (proposed) ===")

# same initial condition as the picked scan particle, as a standalone
# heliocentric (r, v) state for the kernel
tmp = rebound.Simulation()
tmp.units = ("yr", "AU", "Msun")
tmp.add(m=1.0)
tmp.add(m=0.0, a=a0[pick], e=e0[pick], inc=inc0[pick], Omega=np.radians(Omega0[pick]),
        omega=np.radians(omega0[pick]), M=np.radians(M0[pick]))
p = tmp.particles[1]
r_init = np.array([[p.x, p.y, p.z]])
v_init = np.array([[p.vx, p.vy, p.vz]])


def run_kernel_track(dt_disk, snap_dt=SNAP_DT, n_snap=N_SNAP):
    n_disk_steps_per_snap = max(1, round(snap_dt / dt_disk))
    n_disk_steps = n_disk_steps_per_snap * n_snap
    sim_k = build_massive_sim(*HPX, dt_massive=native_dt_for(dt_disk, DT_MASSIVE))
    rho_mid, m_pert, lam_neptune = drive_and_extract_mid(
        sim_k, dt_disk, n_disk_steps, track_index=neptune_idx
    )
    r, v = r_init.copy(), v_init.copy()
    phi = np.empty(n_snap)
    for k in range(n_snap):
        lo, hi = k * n_disk_steps_per_snap, (k + 1) * n_disk_steps_per_snap
        r, v = propagate(r, v, rho_mid[lo:hi], m_pert, GM_SUN, dt_disk, 3)
        a_e, e_e, i_e, Om_e, om_e, M_e = elements_from_state(1.0, r[0], v[0])
        # resonant_angle_32 expects raw (Omega,omega,M) for both bodies; feed
        # Neptune's Omega=lam_neptune, omega=0, M=0 so mean_longitude(...)
        # reduces to lam_neptune directly (only the combined mean longitude
        # is tracked for Neptune, not its Omega/omega split).
        phi[k] = resonant_angle_32(Om_e, om_e, M_e, lam_neptune[hi - 1], 0.0, 0.0)
    return phi


phi_fine = run_kernel_track(DT_MASSIVE)
phi_coarse = run_kernel_track(DT_DISK_COARSE)

fine_span, fine_unwrapped = libration_span(phi_fine)
coarse_span, coarse_unwrapped = libration_span(phi_coarse)

print(f"\nREBOUND reference   (dt=0.5, native):   span={ref_span:7.1f} deg  center={ref_unwrapped.mean():7.1f} deg")
print(f"Kernel dt_disk=0.5   (sanity check):     span={fine_span:7.1f} deg  center={fine_unwrapped.mean():7.1f} deg")
print(f"Kernel dt_disk={DT_DISK_COARSE:.2f}  (proposed):     span={coarse_span:7.1f} deg  center={coarse_unwrapped.mean():7.1f} deg")

print()
if fine_span < 300 and coarse_span < 300 and abs(coarse_span - ref_span) < 150:
    print("RESULT: coarse dt_disk still librates with a similar span/center -- "
          "multi-rate idea holds for this resonance/particle.")
elif coarse_span >= 300:
    print("RESULT: coarse dt_disk particle CIRCULATES where the reference "
          "librates -- the proposed dt_disk is NOT safe as-is for this resonance.")
else:
    print("RESULT: coarse dt_disk still librates but with a substantially "
          "different span than the reference -- treat as a partial pass, "
          "needs a tighter dt_disk or more investigation.")
