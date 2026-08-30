"""Generalization test for the frequency-calibration experiment
(test_secular_freq_calibration.py): does using the calibrated eigenmodes
(fit from a 6e7 yr N-body window, cached in _secular_calibration_cache.npz)
improve population-level disk-particle accuracy at 2e7 yr, relative to the
theoretical-eigenvalue baseline already measured this session (e correlation
r=0.41, i correlation r=0.45, N-body eccentricity std ~2.5x wider than
secular's)?

Same population setup as test_secular_statistical.py's extended validation
(N_PARTICLES=120, T_TOTAL=2e7, non-resonant only) for a direct, apples-to-
apples comparison against those already-known numbers.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent))
from test_secular_multibody import (  # noqa: E402
    particle_coupling, hk_from_elements, elements_from_hk, pq_from_elements, elements_from_pq,
)
from test_secular_step34 import resonance_width_coefficient, resonance_half_width  # noqa: E402
from planetx.constants import GIANT_PLANETS  # noqa: E402

cache = np.load(Path(__file__).parent / "_secular_calibration_cache.npz")
g_fit, f_fit = cache["g_fit"], cache["f_fit"]
Ce_fit, Ci_fit = cache["Ce_fit"], cache["Ci_fit"]
a_arr, m_arr = cache["a_arr"], cache["m_arr"]

names5 = ["jupiter", "saturn", "uranus", "neptune", "hpx"]

N_PARTICLES = 120
T_TOTAL = 2.0e7

rng = np.random.default_rng(7)  # same seed as the original extended validation

a_neptune = GIANT_PLANETS["neptune"]["a"]
m_neptune = GIANT_PLANETS["neptune"]["m"]
width_coeffs = [resonance_width_coefficient(a_neptune, p, m_neptune) for p in range(1, 8)]

a_p = np.empty(0); e_p = np.empty(0)
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

print(f"Sampled {N_PARTICLES} non-resonant disk particles (same seed/distributions as the "
      f"original extended validation)")

# --- REBOUND N-body reference ---
print(f"\nRunning REBOUND N-body for {T_TOTAL:.0e} yr, {N_PARTICLES} particles + 5 massive bodies...")
sim = rebound.Simulation()
sim.units = ("yr", "AU", "Msun")
sim.add(m=1.0, name="sun")
e_arr = np.array([GIANT_PLANETS[n]["e"] for n in names5[:-1]] + [0.3])
inc_arr = np.array([GIANT_PLANETS[n]["inc"] for n in names5[:-1]] + [20.0])
Omega_arr = np.array([GIANT_PLANETS[n]["Omega"] for n in names5[:-1]] + [0.0])
omega_arr = np.array([GIANT_PLANETS[n]["omega"] for n in names5[:-1]] + [0.0])
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
print(f"N-body: {time.perf_counter()-t0:.2f}s")

e_meas = np.empty(N_PARTICLES); i_meas = np.empty(N_PARTICLES)
Omega_meas = np.empty(N_PARTICLES); pomega_meas = np.empty(N_PARTICLES)
for k in range(N_PARTICLES):
    o = sim.particles[5 + k].orbit(primary=sim.particles[0])
    e_meas[k] = o.e
    i_meas[k] = np.degrees(o.inc)
    Omega_meas[k] = np.degrees(o.Omega) % 360
    pomega_meas[k] = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360


def forced_free_calibrated(free_rate, coupling_j, omega_l, C, z0_p, t):
    nu = coupling_j @ C  # [n_modes]
    denom = free_rate - omega_l
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    forced0 = -np.sum(nu / denom)
    z_free0 = z0_p - forced0
    forced_t = -np.sum((nu / denom) * np.exp(-1j * omega_l * t))
    free_t = z_free0 * np.exp(-1j * free_rate * t)
    return free_t + forced_t


print(f"\nEvaluating CALIBRATED secular forced+free for the same {N_PARTICLES} particles...")
t0 = time.perf_counter()
e_pred = np.empty(N_PARTICLES); pomega_pred = np.empty(N_PARTICLES)
i_pred = np.empty(N_PARTICLES); Omega_pred = np.empty(N_PARTICLES)
for k in range(N_PARTICLES):
    A_free, B_free, A_j, B_j = particle_coupling(a_p[k], a_arr, m_arr)
    h0k, k0k = hk_from_elements(e_p[k], Omega_p[k] + omega_p[k])
    zt = forced_free_calibrated(A_free, A_j, g_fit, Ce_fit, h0k + 1j * k0k, T_TOTAL)
    e_pred[k], pomega_pred[k] = elements_from_hk(zt.real, zt.imag)

    p0k, q0k = pq_from_elements(np.degrees(inc_p[k]), Omega_p[k])
    wt = forced_free_calibrated(B_free, B_j, f_fit, Ci_fit, p0k + 1j * q0k, T_TOTAL)
    i_pred[k], Omega_pred[k] = elements_from_pq(wt.real, wt.imag)
print(f"secular (calibrated): {time.perf_counter()-t0:.3f}s")

bound = e_meas < 1.0
n_ejected = int(np.sum(~bound))
print(f"\n{n_ejected}/{N_PARTICLES} particles ejected (e>=1) -- excluded below")

e_meas_b, e_pred_b = e_meas[bound], e_pred[bound]
i_meas_b, i_pred_b = i_meas[bound], i_pred[bound]
pomega_meas_b, pomega_pred_b = pomega_meas[bound], pomega_pred[bound]
Omega_meas_b, Omega_pred_b = Omega_meas[bound], Omega_pred[bound]

print("\n=== eccentricity (calibrated) ===")
print(f"N-body:  mean={e_meas_b.mean():.4f} std={e_meas_b.std():.4f}")
print(f"secular: mean={e_pred_b.mean():.4f} std={e_pred_b.std():.4f}")
print(f"KS p={stats.ks_2samp(e_meas_b, e_pred_b).pvalue:.4f}  "
      f"corr={np.corrcoef(e_meas_b, e_pred_b)[0,1]:.3f}  (baseline, theoretical freqs: r=0.41)")

print("\n=== inclination (calibrated) ===")
print(f"N-body:  mean={i_meas_b.mean():.4f} std={i_meas_b.std():.4f}")
print(f"secular: mean={i_pred_b.mean():.4f} std={i_pred_b.std():.4f}")
print(f"KS p={stats.ks_2samp(i_meas_b, i_pred_b).pvalue:.4f}  "
      f"corr={np.corrcoef(i_meas_b, i_pred_b)[0,1]:.3f}  (baseline, theoretical freqs: r=0.45)")

print("\n=== Omega (node) circular clustering (calibrated) -- the project-relevant statistic ===")
csp_meas = 1 - np.abs(np.mean(np.exp(1j * np.radians(Omega_meas_b))))
csp_pred = 1 - np.abs(np.mean(np.exp(1j * np.radians(Omega_pred_b))))
print(f"N-body circular-std-proxy:  {csp_meas:.4f}")
print(f"secular circular-std-proxy: {csp_pred:.4f}  (baseline, theoretical freqs: N-body=0.676 secular=0.816)")

print("\n=== pomega circular clustering (calibrated) ===")
csp_meas_p = 1 - np.abs(np.mean(np.exp(1j * np.radians(pomega_meas_b))))
csp_pred_p = 1 - np.abs(np.mean(np.exp(1j * np.radians(pomega_pred_b))))
print(f"N-body circular-std-proxy:  {csp_meas_p:.4f}")
print(f"secular circular-std-proxy: {csp_pred_p:.4f}")
