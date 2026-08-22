"""Steps 3-4 of the Stage 3 plan (/Users/xtan/.claude/plans/robust-puzzling-flamingo.md):
resonance-width triage and the blended secular/N-body cost benchmark.

CAVEAT carried forward from Steps 1-2: the multi-body secular solution's
phase/timing accuracy is unresolved (amplitude-correct, phase-uncertain for
particles forced by multiple comparable perturbers). The cost numbers here
are a conditional "what this approach would cost if its accuracy is judged
adequate for population-level statistics," not a validated final answer.

Step 3 formula (Murray & Dermott Ch. 8, via Wallace/Quinn/Boley 2021
Appendix B, general q-th order width -- used in preference to the
first-order-specific form since one of that form's sub-terms was flagged
unverified by the earlier research):

    delta_a / a = +/- sqrt( (16/3) * |C_r| / n * e^|q| )
    C_r = (m_p/M_sun) * n * alpha * f_d(alpha)

f_d(alpha) for a first-order (q=1) exterior resonance (M&D Table 8.1):
    f_d(alpha) = -(p+1) * b_(1/2)^(p+1)(alpha) - (alpha/2) * D[b_(1/2)^(p+1)](alpha)

D[b_s^j](alpha) via the standard Laplace-coefficient derivative recursion
(Brouwer & Clemence 1961):
    D[b_s^j](alpha) = s * ( b_(s+1)^(j-1)(alpha) - 2*alpha*b_(s+1)^j(alpha) + b_(s+1)^(j+1)(alpha) )

Validated below against real empirical data already obtained this session
(scripts/validate_multirate_disk.py's 3:2 resonance scan at e0=0.20), not a
new N-body run -- reusing hard-won data rather than re-deriving it.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import GM_SUN  # noqa: E402
from test_secular_theory import laplace_b  # noqa: E402
from test_secular_multibody import (  # noqa: E402
    build_AB_matrices, eigen_solution, hk_from_elements, particle_forced_free_hk,
)


def laplace_b_deriv(s, j, alpha):
    """D[b_s^j](alpha), via the standard recursion (Brouwer & Clemence 1961)."""
    return s * (laplace_b(s + 1, j - 1, alpha) - 2 * alpha * laplace_b(s + 1, j, alpha)
                + laplace_b(s + 1, j + 1, alpha))


def resonance_width_coefficient(a_neptune, p, m_neptune, M_sun=1.0):
    """Everything about the (p+1):p exterior first-order MMR width that does
    NOT depend on the test particle's eccentricity (the expensive part --
    one Laplace-coefficient quadrature pair per resonance, not per
    particle). Width itself is width_coeff * sqrt(e); see
    resonance_half_width below.
    """
    a_res = a_neptune * ((p + 1) / p) ** (2.0 / 3.0)
    alpha = a_neptune / a_res  # < 1, perturber interior to particle
    n = np.sqrt(GM_SUN / a_res**3)
    j = p + 1
    fd = -j * laplace_b(0.5, j, alpha) - (alpha / 2.0) * laplace_b_deriv(0.5, j, alpha)
    Cr = (m_neptune / M_sun) * n * alpha * fd
    width_coeff = a_res * np.sqrt((16.0 / 3.0) * abs(Cr) / n)  # half_width = width_coeff * sqrt(e)
    return a_res, width_coeff


def resonance_half_width(width_coeff, e):
    return width_coeff * np.sqrt(e)


def resonance_width_exterior(a_neptune, p, e, m_neptune, M_sun=1.0):
    """Convenience wrapper (recomputes the coefficient each call -- fine for
    a handful of one-off checks, NOT for a per-particle loop; use
    resonance_width_coefficient once + resonance_half_width per-particle
    for that)."""
    a_res, width_coeff = resonance_width_coefficient(a_neptune, p, m_neptune, M_sun)
    return a_res, resonance_half_width(width_coeff, e)


if __name__ == "__main__":
    from planetx.constants import GIANT_PLANETS, EARTH_MASS_IN_MSUN

    a_neptune = GIANT_PLANETS["neptune"]["a"]
    m_neptune = GIANT_PLANETS["neptune"]["m"]

    print("=== Step 3a: validate width formula against already-obtained empirical data ===")
    print("(scripts/validate_multirate_disk.py's 3:2 scan at e0=0.20: librating a0 in "
          "[39.178, 39.525]; circulating outside that range)")
    a_res, half_width = resonance_width_exterior(a_neptune, p=2, e=0.20, m_neptune=m_neptune)
    print(f"formula: 3:2 resonance at a={a_res:.3f} AU, half-width={half_width:.3f} AU "
          f"-> zone [{a_res-half_width:.3f}, {a_res+half_width:.3f}]")
    print("empirical librating zone was approximately [39.178, 39.525] "
          "(asymmetric due to small-sample scan, not a clean boundary)")

    print("\n=== Step 3b: resonance map for the disk's actual a-range (30-100 AU) ===")
    print(f"{'p+1:p':>8s} {'a_res (AU)':>12s} {'half-width @ e=0.05':>22s} {'half-width @ e=0.15':>22s}")
    resonances = []
    for p in range(1, 8):
        a_res, hw_lo = resonance_width_exterior(a_neptune, p, e=0.05, m_neptune=m_neptune)
        _, hw_hi = resonance_width_exterior(a_neptune, p, e=0.15, m_neptune=m_neptune)
        resonances.append((a_res, hw_lo, hw_hi))
        print(f"{p+1}:{p:<6d} {a_res:12.3f} {hw_lo:22.4f} {hw_hi:22.4f}")

    print("\n=== Step 3c: resonant fraction for a realistic disk draw ===")
    from planetx.config import PriorConfig
    cfg = PriorConfig.from_yaml("configs/prior.yaml")
    rng = np.random.default_rng(42)
    n_sample = 20000
    a_sample = rng.uniform(cfg.nuisance["disk_inner_edge"].low, cfg.nuisance["disk_outer_edge"].high, n_sample)
    e_scale_sample = rng.uniform(cfg.nuisance["disk_e_scale"].low, cfg.nuisance["disk_e_scale"].high)
    e_sample = np.clip(rng.rayleigh(e_scale_sample, n_sample), 0, 0.9)

    # cache the (expensive) Laplace-coefficient part once per resonance,
    # then vectorize the cheap per-particle sqrt(e) scaling over all 20000
    # particles at once -- avoids recomputing the quadrature 140,000 times
    is_resonant = np.zeros(n_sample, dtype=bool)
    for p in range(1, 8):
        a_res, width_coeff = resonance_width_coefficient(a_neptune, p, m_neptune)
        hw = resonance_half_width(width_coeff, np.maximum(e_sample, 0.01))
        is_resonant |= np.abs(a_sample - a_res) < hw
    f_resonant = is_resonant.mean()
    print(f"disk_e_scale drawn for this sample: {e_scale_sample:.4f}")
    print(f"n_sample={n_sample}, resonant fraction (within any p=1..7 first-order Neptune MMR): "
          f"{f_resonant:.4f} ({is_resonant.sum()} particles)")

    print("\n=== Step 4: blended cost benchmark ===")
    # N-body cost: measured this session (Stage 0, worker.py with safe_mode=0)
    nbody_us_per_step = 0.235  # us/particle, marginal, safe_mode=0, from this session's benchmark
    dt_years = 0.5
    integration_years = 4.5e9
    n_steps = integration_years / dt_years
    nbody_cost_per_particle_days = (n_steps * nbody_us_per_step * 1e-6) / 86400

    # secular cost: measure directly -- time evaluating the forced+free
    # formula for many particles at a batch of output times (closed form,
    # cost should NOT scale with integration_years at all)
    a_arr = np.array([GIANT_PLANETS[n]["a"] for n in ["jupiter", "saturn", "uranus", "neptune"]] + [400.0])
    e_arr = np.array([GIANT_PLANETS[n]["e"] for n in ["jupiter", "saturn", "uranus", "neptune"]] + [0.3])
    Omega_arr = np.array([GIANT_PLANETS[n]["Omega"] for n in ["jupiter", "saturn", "uranus", "neptune"]] + [0.0])
    omega_arr = np.array([GIANT_PLANETS[n]["omega"] for n in ["jupiter", "saturn", "uranus", "neptune"]] + [0.0])
    m_arr = np.array([GIANT_PLANETS[n]["m"] for n in ["jupiter", "saturn", "uranus", "neptune"]]
                      + [5.0 * EARTH_MASS_IN_MSUN])
    A5, B5 = build_AB_matrices(a_arr, m_arr)
    g, Ve = eigen_solution(A5)
    z0 = hk_from_elements(e_arr, Omega_arr + omega_arr)
    z0c = z0[0] + 1j * z0[1]

    n_eval = 50  # purely for a per-particle timing average -- cost per call
    # is ~constant regardless of which particle, so 50 samples is plenty;
    # each call recomputes several Laplace-coefficient quadratures (the
    # same expensive-per-call pattern as Step 3's original mistake), so a
    # large n_eval here would be needlessly slow for no added precision
    a_eval = rng.uniform(30, 100, n_eval)
    e_eval = np.clip(rng.rayleigh(0.05, n_eval), 0, 0.9)
    h0_eval = e_eval * np.sin(rng.uniform(0, 2 * np.pi, n_eval))
    k0_eval = e_eval * np.cos(rng.uniform(0, 2 * np.pi, n_eval))

    t_final = np.array([integration_years])
    t0 = time.perf_counter()
    for i in range(n_eval):
        particle_forced_free_hk(a_eval[i], a_arr, m_arr, g, Ve, z0c, h0_eval[i], k0_eval[i], t_final)
    elapsed = time.perf_counter() - t0
    secular_us_per_particle = elapsed / n_eval * 1e6
    secular_cost_per_particle_days = (secular_us_per_particle * 1e-6) / 86400

    print(f"N-body:  {nbody_cost_per_particle_days*24:.4f} hours/particle for the full "
          f"{integration_years:.1e} yr integration ({n_steps:.2e} steps)")
    print(f"secular: {secular_us_per_particle:.2f} us/particle for the SAME span "
          f"(evaluated directly at t={integration_years:.1e} yr, no stepping) "
          f"= {secular_cost_per_particle_days*24*3600*1e6:.2f} us/particle-equivalent")

    n_test_particles = 20000
    nbody_full_days = n_test_particles * nbody_cost_per_particle_days
    secular_full_days = n_test_particles * secular_cost_per_particle_days
    blended_days = f_resonant * nbody_full_days + (1 - f_resonant) * secular_full_days

    print(f"\nAt n_test_particles={n_test_particles}, full integration_years={integration_years:.1e}:")
    print(f"  all-N-body (current):        {nbody_full_days:.1f} days/simulation")
    print(f"  all-secular (hypothetical):   {secular_full_days:.6f} days/simulation")
    print(f"  blended (f_resonant={f_resonant:.4f}): {blended_days:.2f} days/simulation")
    print(f"\n  speedup vs. current all-N-body: {nbody_full_days/blended_days:.1f}x")
