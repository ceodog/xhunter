"""#2 from the accuracy-improvement discussion: does the secular method's
phase error matter at the POPULATION level, even though it's real for
individual trajectories (Steps 1-2 in the plan doc)? Runs a batch of
non-resonant disk-representative particles through both REBOUND N-body and
the secular forced+free formula, and compares the resulting DISTRIBUTIONS
of final elements -- not individual trajectories -- since that's what the
Set Transformer actually trains on.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import GM_SUN  # noqa: E402
from test_secular_multibody import (  # noqa: E402
    build_AB_matrices, eigen_solution, hk_from_elements, particle_forced_free_hk, elements_from_hk,
    particle_forced_free_pq, pq_from_elements, elements_from_pq,
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

N_PARTICLES = 120
T_TOTAL = 2.0e7  # 10x the earlier check -- more secular periods covered,
# still ~225x short of the 4.5e9 yr production target (untestable in a
# session), but a meaningfully harder test of whether errors compound

rng = np.random.default_rng(7)

# sample from the actual disk priors, filtering out resonant particles
# (Step 3's width formula) -- secular theory doesn't claim to work there,
# and this is testing the population it's ACTUALLY meant to cover
a_neptune = GIANT_PLANETS["neptune"]["a"]
m_neptune = GIANT_PLANETS["neptune"]["m"]
width_coeffs = [resonance_width_coefficient(a_neptune, p, m_neptune) for p in range(1, 8)]

a_p = np.empty(0)
e_p = np.empty(0)
inc_p = np.empty(0)
Omega_p = np.empty(0)
omega_p = np.empty(0)
while len(a_p) < N_PARTICLES:
    n_try = N_PARTICLES * 2
    a_try = rng.uniform(30, 100, n_try)
    e_try = np.clip(rng.rayleigh(0.08, n_try), 0.001, 0.9)
    resonant = np.zeros(n_try, dtype=bool)
    for a_res, wc in width_coeffs:
        hw = resonance_half_width(wc, e_try)
        resonant |= np.abs(a_try - a_res) < hw
    keep = ~resonant
    a_p = np.concatenate([a_p, a_try[keep]])
    e_p = np.concatenate([e_p, e_try[keep]])
inc_p = np.radians(np.clip(rng.rayleigh(5.0, len(a_p)), 0.1, 60))
Omega_p = rng.uniform(0, 360, len(a_p))
omega_p = rng.uniform(0, 360, len(a_p))
a_p, e_p, inc_p, Omega_p, omega_p = (x[:N_PARTICLES] for x in (a_p, e_p, inc_p, Omega_p, omega_p))

print(f"Sampled {N_PARTICLES} non-resonant disk particles from configs/prior.yaml-like priors")
print(f"a range: [{a_p.min():.1f}, {a_p.max():.1f}] AU, mean e: {e_p.mean():.3f}")

# --- REBOUND N-body, all particles in one 5-body + N_PARTICLES sim ---
print(f"\nRunning REBOUND N-body for {T_TOTAL:.0e} yr, {N_PARTICLES} particles + 5 massive bodies...")
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")
sim.add(m=1.0, name="sun")
for i, name in enumerate(names5):
    sim.add(m=m_arr[i], a=a_arr[i], e=e_arr[i], inc=np.radians(inc_arr[i]),
            Omega=np.radians(Omega_arr[i]), omega=np.radians(omega_arr[i]), M=0.0, name=name)
sim.N_active = len(sim.particles)
for i in range(N_PARTICLES):
    sim.add(m=0.0, a=a_p[i], e=e_p[i], inc=inc_p[i], Omega=np.radians(Omega_p[i]),
            omega=np.radians(omega_p[i]), M=rng.uniform(0, 2 * np.pi))
sim.move_to_com()
sim.integrator = "whfast"
sim.integrator.safe_mode = 0
sim.dt = min(0.5, a_arr[0] ** 1.5 / 20.0)

t0 = time.perf_counter()
sim.integrate(T_TOTAL)
nbody_elapsed = time.perf_counter() - t0
print(f"N-body: {nbody_elapsed:.2f}s")

e_meas = np.empty(N_PARTICLES)
i_meas = np.empty(N_PARTICLES)
Omega_meas = np.empty(N_PARTICLES)
pomega_meas = np.empty(N_PARTICLES)  # Omega+omega -- the quantity the eccentricity-vector
# (h,k) secular formula predicts; Omega alone is the inclination/node problem's own
# variable (p,q), evaluated separately below
for k in range(N_PARTICLES):
    o = sim.particles[6 + k].orbit(primary=sim.particles[0])
    e_meas[k] = o.e
    i_meas[k] = np.degrees(o.inc)
    Omega_meas[k] = np.degrees(o.Omega) % 360
    pomega_meas[k] = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360

# --- secular forced+free, batched ---
print(f"\nEvaluating secular forced+free for the same {N_PARTICLES} particles...")
A5, B5 = build_AB_matrices(a_arr, m_arr)
g, Ve = eigen_solution(A5)
f, Vi = eigen_solution(B5)
z0e = hk_from_elements(e_arr, Omega_arr + omega_arr)
z0ec = z0e[0] + 1j * z0e[1]
w0i = pq_from_elements(inc_arr, Omega_arr)
w0ic = w0i[0] + 1j * w0i[1]

t0 = time.perf_counter()
e_pred = np.empty(N_PARTICLES)
pomega_pred = np.empty(N_PARTICLES)
i_pred = np.empty(N_PARTICLES)
Omega_pred = np.empty(N_PARTICLES)
for k in range(N_PARTICLES):
    h0k, k0k = e_p[k] * np.sin(np.radians(Omega_p[k] + omega_p[k])), e_p[k] * np.cos(np.radians(Omega_p[k] + omega_p[k]))
    zt = particle_forced_free_hk(a_p[k], a_arr, m_arr, g, Ve, z0ec, h0k, k0k, np.array([T_TOTAL]))
    e_pred[k], pomega_pred[k] = elements_from_hk(zt.real, zt.imag)

    p0k, q0k = pq_from_elements(np.degrees(inc_p[k]), Omega_p[k])
    wt = particle_forced_free_pq(a_p[k], a_arr, m_arr, f, Vi, w0ic, p0k, q0k, np.array([T_TOTAL]))
    i_pred[k], Omega_pred[k] = elements_from_pq(wt.real, wt.imag)
secular_elapsed = time.perf_counter() - t0
print(f"secular: {secular_elapsed:.3f}s ({secular_elapsed/N_PARTICLES*1e6:.1f} us/particle, both e and i)")

n_ejected = int(np.sum(e_meas >= 1.0))
print(f"\n{n_ejected}/{N_PARTICLES} particles went unbound (e>=1, dynamically ejected) "
      f"during the {T_TOTAL:.0e} yr N-body integration -- real dynamical instability "
      f"for some initial conditions, not a bug. Excluded from the distributional "
      f"comparison below (secular theory assumes bound, moderate-e orbits; an ejected "
      f"particle isn't part of the observable population anyway).")
bound = e_meas < 1.0
e_meas_b, e_pred_b = e_meas[bound], e_pred[bound]

print("\n=== Distributional comparison: eccentricity (bound particles only) ===")
print(f"N-body:  mean={e_meas_b.mean():.4f}  std={e_meas_b.std():.4f}  "
      f"min={e_meas_b.min():.4f}  max={e_meas_b.max():.4f}")
print(f"secular: mean={e_pred_b.mean():.4f}  std={e_pred_b.std():.4f}  "
      f"min={e_pred_b.min():.4f}  max={e_pred_b.max():.4f}")
ks_e = stats.ks_2samp(e_meas_b, e_pred_b)
print(f"KS test (e distributions, bound only): statistic={ks_e.statistic:.4f}, p-value={ks_e.pvalue:.4f}")

print("\n=== Per-particle correlation (does secular rank-order match N-body?) ===")
corr_e = np.corrcoef(e_meas_b, e_pred_b)[0, 1]
print(f"Pearson correlation, e_meas vs e_pred across the {bound.sum()} bound particles: {corr_e:.3f}")

pomega_meas_b, pomega_pred_b = pomega_meas[bound], pomega_pred[bound]
print("\n=== pomega (apsidal longitude) distribution -- the clustering-signal-relevant quantity, bound only ===")
print(f"N-body:  circular mean pomega = {np.degrees(np.angle(np.mean(np.exp(1j*np.radians(pomega_meas_b))))) % 360:.1f} deg, "
      f"circular std proxy = {1 - np.abs(np.mean(np.exp(1j*np.radians(pomega_meas_b)))):.4f}")
print(f"secular: circular mean pomega = {np.degrees(np.angle(np.mean(np.exp(1j*np.radians(pomega_pred_b))))) % 360:.1f} deg, "
      f"circular std proxy = {1 - np.abs(np.mean(np.exp(1j*np.radians(pomega_pred_b)))):.4f}")
ks_pom = stats.ks_2samp(pomega_meas_b, pomega_pred_b)
print(f"KS test (pomega distributions, bound only): statistic={ks_pom.statistic:.4f}, p-value={ks_pom.pvalue:.4f}")

i_meas_b, i_pred_b = i_meas[bound], i_pred[bound]
print("\n=== Distributional comparison: inclination (bound particles only) ===")
print(f"N-body:  mean={i_meas_b.mean():.4f}  std={i_meas_b.std():.4f}  "
      f"min={i_meas_b.min():.4f}  max={i_meas_b.max():.4f}")
print(f"secular: mean={i_pred_b.mean():.4f}  std={i_pred_b.std():.4f}  "
      f"min={i_pred_b.min():.4f}  max={i_pred_b.max():.4f}")
ks_i = stats.ks_2samp(i_meas_b, i_pred_b)
print(f"KS test (i distributions, bound only): statistic={ks_i.statistic:.4f}, p-value={ks_i.pvalue:.4f}")
corr_i = np.corrcoef(i_meas_b, i_pred_b)[0, 1]
print(f"Pearson correlation, i_meas vs i_pred across the {bound.sum()} bound particles: {corr_i:.3f}")

Omega_meas_b, Omega_pred_b = Omega_meas[bound], Omega_pred[bound]
print("\n=== Omega (node) distribution -- the other clustering-signal-relevant quantity, bound only ===")
print(f"N-body:  circular mean Omega = {np.degrees(np.angle(np.mean(np.exp(1j*np.radians(Omega_meas_b))))) % 360:.1f} deg, "
      f"circular std proxy = {1 - np.abs(np.mean(np.exp(1j*np.radians(Omega_meas_b)))):.4f}")
print(f"secular: circular mean Omega = {np.degrees(np.angle(np.mean(np.exp(1j*np.radians(Omega_pred_b))))) % 360:.1f} deg, "
      f"circular std proxy = {1 - np.abs(np.mean(np.exp(1j*np.radians(Omega_pred_b)))):.4f}")
ks_Om = stats.ks_2samp(Omega_meas_b, Omega_pred_b)
print(f"KS test (Omega distributions, bound only): statistic={ks_Om.statistic:.4f}, p-value={ks_Om.pvalue:.4f}")
