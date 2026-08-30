"""Benchmark the ALL-secular approach (no resonance triage) -- distinct from
test_secular_statistical.py, which deliberately excludes resonant particles
(that's the "hybrid" method's population). Here every particle sampled from
the actual disk priors is run through the secular forced+free formula
regardless of whether it falls inside a Neptune MMR width, matching what a
pure secular-only pipeline (no N-body fallback at all) would actually do.
Reports accuracy split by resonant/non-resonant subgroup (to see where error
concentrates) plus a direct per-particle speed measurement.
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

N_PARTICLES = 200
T_TOTAL = 2.0e7  # same as the hardest hybrid check already run -- comparable

rng = np.random.default_rng(11)

# Sample from the disk priors WITHOUT resonance filtering -- this is the point:
# a pure all-secular pipeline has no triage step at all.
a_p = rng.uniform(30, 100, N_PARTICLES)
e_p = np.clip(rng.rayleigh(0.08, N_PARTICLES), 0.001, 0.9)
inc_p = np.radians(np.clip(rng.rayleigh(5.0, N_PARTICLES), 0.1, 60))
Omega_p = rng.uniform(0, 360, N_PARTICLES)
omega_p = rng.uniform(0, 360, N_PARTICLES)

# tag which ones fall inside a low-order Neptune MMR width, for reporting only
a_neptune = GIANT_PLANETS["neptune"]["a"]
m_neptune = GIANT_PLANETS["neptune"]["m"]
width_coeffs = [resonance_width_coefficient(a_neptune, p, m_neptune) for p in range(1, 8)]
is_resonant = np.zeros(N_PARTICLES, dtype=bool)
for a_res, wc in width_coeffs:
    hw = resonance_half_width(wc, e_p)
    is_resonant |= np.abs(a_p - a_res) < hw

print(f"Sampled {N_PARTICLES} disk particles, NO resonance triage (all-secular test)")
print(f"a range: [{a_p.min():.1f}, {a_p.max():.1f}] AU, mean e: {e_p.mean():.3f}")
print(f"{is_resonant.sum()}/{N_PARTICLES} ({100*is_resonant.mean():.1f}%) fall inside a low-order "
      f"Neptune MMR width -- these get secular-propagated anyway under an all-secular policy")

# --- REBOUND N-body reference, all particles in one sim ---
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
print(f"N-body: {nbody_elapsed:.2f}s ({nbody_elapsed/N_PARTICLES*1e6:.1f} us/particle marginal, "
      f"though N-body cost is dominated by the massive-body integration, not per-particle)")

e_meas = np.empty(N_PARTICLES)
i_meas = np.empty(N_PARTICLES)
Omega_meas = np.empty(N_PARTICLES)
pomega_meas = np.empty(N_PARTICLES)
for k in range(N_PARTICLES):
    o = sim.particles[6 + k].orbit(primary=sim.particles[0])
    e_meas[k] = o.e
    i_meas[k] = np.degrees(o.inc)
    Omega_meas[k] = np.degrees(o.Omega) % 360
    pomega_meas[k] = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360

# --- secular forced+free, batched, ALL particles including resonant ---
print(f"\nEvaluating secular forced+free for all {N_PARTICLES} particles (no triage)...")
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
us_per_particle = secular_elapsed / N_PARTICLES * 1e6
print(f"secular: {secular_elapsed:.3f}s ({us_per_particle:.1f} us/particle, both e and i)")

n_ejected = int(np.sum(e_meas >= 1.0))
print(f"\n{n_ejected}/{N_PARTICLES} particles went unbound (e>=1) during the N-body integration "
      f"-- excluded from comparisons below")
bound = e_meas < 1.0


def report(label, mask):
    n = mask.sum()
    if n < 5:
        print(f"\n=== {label} (n={n}) -- too few for a meaningful stat ===")
        return
    em, ep = e_meas[mask], e_pred[mask]
    im, ip = i_meas[mask], i_pred[mask]
    pm, pp = pomega_meas[mask], pomega_pred[mask]
    Om, Op = Omega_meas[mask], Omega_pred[mask]
    ks_e = stats.ks_2samp(em, ep)
    ks_i = stats.ks_2samp(im, ip)
    corr_e = np.corrcoef(em, ep)[0, 1] if n > 1 else float("nan")
    corr_i = np.corrcoef(im, ip)[0, 1] if n > 1 else float("nan")
    print(f"\n=== {label} (n={n}) ===")
    print(f"  e:  N-body mean/std={em.mean():.4f}/{em.std():.4f}  secular mean/std={ep.mean():.4f}/{ep.std():.4f}  "
          f"KS p={ks_e.pvalue:.4f}  corr={corr_e:.3f}")
    print(f"  i:  N-body mean/std={im.mean():.4f}/{im.std():.4f}  secular mean/std={ip.mean():.4f}/{ip.std():.4f}  "
          f"KS p={ks_i.pvalue:.4f}  corr={corr_i:.3f}")


report("ALL particles (true all-secular policy, resonant+non-resonant mixed)", bound)
report("non-resonant subset only", bound & ~is_resonant)
report("resonant subset only (where linear theory is not supposed to apply)", bound & is_resonant)

# --- cost extrapolation to full production scale ---
n_prod = 500
years_prod = 4.5e9
secular_only_days = us_per_particle * 1e-6 * n_prod / 86400.0
print(f"\n=== Cost extrapolation: pure all-secular, n_test_particles={n_prod}, {years_prod:.1e} yr ===")
print(f"Measured secular cost: {us_per_particle:.1f} us/particle (independent of integration_years -- "
      f"closed-form evaluation, not a stepped integration)")
print(f"All-secular total: {secular_only_days*1e6:.3f} microseconds -> {secular_only_days:.6f} days/simulation")
print(f"(for comparison: current all-N-body production cost is 12.36 days/simulation at the same n)")
