"""Stage 3 of the plan (/Users/xtan/.claude/plans/robust-puzzling-flamingo.md):
multi-body Laplace-Lagrange secular theory for Sun + 4 giants + HPX, and the
forced+free response of a massless test particle under all 5 simultaneously.

Formulas from Murray & Dermott "Solar System Dynamics" Ch. 7, verified by a
background research agent against two independent secondary re-derivations
(Fitzpatrick's online notes; Wallace, Quinn & Boley 2021 MNRAS Appendix A).
`celmech` (Hadden & Tamayo 2022) implements the same formalism and was
installed as a cross-check, but its `from_Simulation` path requires a
REBOUND API (`sim.N_real`) not present in this project's pinned rebound
5.0.1 -- used as a source-code reference instead (see plan doc), not a live
dependency. Primary validation is direct comparison against REBOUND N-body,
the same discipline that caught the Kepler-solver bug in Stage 1.

IMPORTANT correction versus the plan file's own shorthand: the off-diagonal
B_ij term uses the b^(1) Laplace coefficient, NOT b^(2) as the plan's
compressed paraphrase suggested -- reading the research agent's verbatim
quoted formula directly (below) rather than trusting my own summary.

    A_ii = +(n_i/4) * sum_{j!=i} eps_ij * alpha_ij * alphabar_ij * b1(alpha_ij)
    A_ij = -(n_i/4) * eps_ij * alpha_ij * alphabar_ij * b2(alpha_ij)   (i!=j)
    B_ii = -(n_i/4) * sum_{j!=i} eps_ij * alpha_ij * alphabar_ij * b1(alpha_ij)  [= -A_ii]
    B_ij = +(n_i/4) * eps_ij * alpha_ij * alphabar_ij * b1(alpha_ij)   (i!=j)  <- b1, not b2

    eps_ij = m_j / (M_sun + m_i)   (NOT m_j/M_sun alone)
    alpha_ij = min(a_i,a_j)/max(a_i,a_j)
    alphabar_ij = alpha_ij if a_i < a_j (body i interior to j) else 1
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import rebound

sys.path.insert(0, str(Path(__file__).parent))
from bench_test_particle_integrator import GM_SUN, build_massive_sim  # noqa: E402
from test_secular_theory import laplace_b  # noqa: E402


# ---------------------------------------------------------------------------
# Massive-body N-body secular matrix
# ---------------------------------------------------------------------------


def build_AB_matrices(a, m, M_sun=1.0):
    """a, m: arrays of semi-major axis (AU) and mass (Msun) for N massive
    bodies. Returns (A, B) matrices [N,N] in rad/yr (using GM_SUN's implicit
    G=1 unit convention -- n_i computed via GM_SUN below)."""
    N = len(a)
    n = np.sqrt(GM_SUN / np.asarray(a) ** 3)  # rad/yr, each body's own mean motion
    A = np.zeros((N, N))
    B = np.zeros((N, N))
    for i in range(N):
        diag_sum = 0.0
        for j in range(N):
            if i == j:
                continue
            eps_ij = m[j] / (M_sun + m[i])
            alpha_ij = min(a[i], a[j]) / max(a[i], a[j])
            alphabar_ij = alpha_ij if a[i] < a[j] else 1.0
            b1 = laplace_b(1.5, 1, alpha_ij)
            b2 = laplace_b(1.5, 2, alpha_ij)
            A[i, j] = -(n[i] / 4.0) * eps_ij * alpha_ij * alphabar_ij * b2
            B[i, j] = +(n[i] / 4.0) * eps_ij * alpha_ij * alphabar_ij * b1
            diag_sum += eps_ij * alpha_ij * alphabar_ij * b1
        A[i, i] = +(n[i] / 4.0) * diag_sum
        B[i, i] = -(n[i] / 4.0) * diag_sum
    return A, B


def eigen_solution(mat):
    """A (or B) -> (eigenvalues g_l, eigenvector matrix V, both possibly
    complex in general -- verified real for this physical problem below)."""
    g, V = np.linalg.eig(mat)
    return g, V


def hk_from_elements(e, pomega_deg):
    pomega = np.radians(pomega_deg)
    return e * np.sin(pomega), e * np.cos(pomega)


def elements_from_hk(h, k):
    e = np.sqrt(h**2 + k**2)
    pomega = np.degrees(np.arctan2(h, k)) % 360
    return e, pomega


def propagate_massive_hk(g, V, z0, t):
    """z0 = h(0)+i*k(0) for each body. Returns z(t) [N] at time t (or [N,len(t)] if t is array)."""
    c = np.linalg.solve(V, z0)
    t = np.atleast_1d(t)
    phase = np.exp(-1j * np.outer(g, t))  # [N_modes, N_t]
    z = V @ (c[:, None] * phase)  # [N_bodies, N_t]
    return z if len(t) > 1 else z[:, 0]


# ---------------------------------------------------------------------------
# Massless test-particle forced+free response
# ---------------------------------------------------------------------------


def particle_coupling(a_p, a_j_arr, m_j_arr, M_sun=1.0):
    """Test particle at a_p under massive bodies at a_j_arr (masses m_j_arr).
    Returns (A_free, B_free, A_j[], B_j[]) -- own free rate + per-body
    forcing coefficients.

    BUG FOUND AND FIXED (this session, via e-scaling diagnostic): alphabar_j
    must trigger on whether the TEST PARTICLE (a_p) is interior to the
    perturber (a_j_arr), not the other way around -- a previous version
    checked `a_j_arr < a_p` (perturber interior to particle), which is
    backwards and silently multiplied A_free/A_j by a spurious extra factor
    of alpha_j whenever the particle was exterior to a perturber (the
    common case for this disk). Confirmed via direct comparison against the
    independently validated single-perturber rate formula in
    test_secular_theory.py's predicted_rates -- the buggy version was off
    by exactly a factor of alpha (0.668 for a_test=45, a_neptune=30.07).
    """
    n_p = np.sqrt(GM_SUN / a_p**3)
    eps_j = m_j_arr / M_sun
    alpha_j = np.where(a_j_arr < a_p, a_j_arr / a_p, a_p / a_j_arr)
    alphabar_j = np.where(a_p < a_j_arr, alpha_j, 1.0)
    b1 = np.array([laplace_b(1.5, 1, a) for a in alpha_j])
    b2 = np.array([laplace_b(1.5, 2, a) for a in alpha_j])
    A_j = -(n_p / 4.0) * eps_j * alpha_j * alphabar_j * b2
    B_j = +(n_p / 4.0) * eps_j * alpha_j * alphabar_j * b1
    A_free = (n_p / 4.0) * np.sum(eps_j * alpha_j * alphabar_j * b1)
    B_free = -A_free
    return A_free, B_free, A_j, B_j


def _particle_forced_free_generic(free_rate, coupling_j, eigvals, eigvecs, z0_massive, z0_p, t):
    """Shared math for both the eccentricity (h,k; use A_free, A_j, g, V_e)
    and inclination (p,q; use B_free, B_j, f, V_i) forced+free solutions --
    identical closed-form structure per Murray & Dermott Sec. 7.5, just a
    different coupling coefficient / eigenmode system as input."""
    c = np.linalg.solve(eigvecs, z0_massive)  # mode amplitudes for each massive body
    nu = np.array([np.sum(coupling_j * c[l] * eigvecs[:, l]) for l in range(len(eigvals))])
    denom = free_rate - eigvals
    forced0 = -np.sum(nu / denom)
    z_free0 = z0_p - forced0
    t = np.atleast_1d(t)
    forced_t = -np.sum((nu / denom)[:, None] * np.exp(-1j * np.outer(eigvals, t)), axis=0)
    free_t = z_free0 * np.exp(-1j * free_rate * t)
    return free_t + forced_t


def particle_forced_free_hk(a_p, a_j_arr, m_j_arr, g, V_e, z0_massive, h0_p, k0_p, t):
    """Closed-form forced+free (h,k) [eccentricity/apsidal] for a massless
    test particle, given the already-solved massive-body eigenmode system
    (g, V_e) and its z0."""
    A_free, _, A_j, _ = particle_coupling(a_p, a_j_arr, m_j_arr)
    z0_p = h0_p + 1j * k0_p
    z_t = _particle_forced_free_generic(A_free, A_j, g, V_e, z0_massive, z0_p, t)
    return z_t if len(t) > 1 else z_t[0]


def particle_forced_free_pq(a_p, a_j_arr, m_j_arr, f, V_i, w0_massive, p0_p, q0_p, t):
    """Closed-form forced+free (p,q) [inclination/node] for a massless test
    particle -- same structure as particle_forced_free_hk, using the B
    matrix's eigenmode system (f, V_i) and the particle's B_j coupling
    instead of A_j."""
    _, B_free, _, B_j = particle_coupling(a_p, a_j_arr, m_j_arr)
    w0_p = p0_p + 1j * q0_p
    w_t = _particle_forced_free_generic(B_free, B_j, f, V_i, w0_massive, w0_p, t)
    return w_t if len(t) > 1 else w_t[0]


def pq_from_elements(inc_deg, Omega_deg):
    """p,q = I*sin(Omega), I*cos(Omega), I in RADIANS -- matches the actual
    linear-theory derivation used for B_free/B_j (I is the direct linear
    analog of e in the eccentricity problem, not sin(I); coincide for the
    modest inclinations in this disk anyway, but matching the formula as
    derived rather than introducing a second, unrelated approximation)."""
    inc = np.radians(inc_deg)
    Omega = np.radians(Omega_deg)
    return inc * np.sin(Omega), inc * np.cos(Omega)


def elements_from_pq(p, q):
    inc = np.degrees(np.hypot(p, q))
    Omega = np.degrees(np.arctan2(p, q)) % 360
    return inc, Omega


# ---------------------------------------------------------------------------
# Validation harness
# ---------------------------------------------------------------------------


def measure_rebound_precession(sim, body_indices, T_TOTAL, n_snap):
    """Measure each body's pomega/Omega precession rate via linear fit,
    same method as the validated single-perturber spot-check."""
    snap_dt = T_TOTAL / n_snap
    pomega_hist = np.zeros((n_snap, len(body_indices)))
    Omega_hist = np.zeros((n_snap, len(body_indices)))
    for s in range(n_snap):
        sim.integrate((s + 1) * snap_dt)
        for k, idx in enumerate(body_indices):
            o = sim.particles[idx].orbit(primary=sim.particles[0])
            Omega_hist[s, k] = np.degrees(o.Omega) % 360
            pomega_hist[s, k] = (np.degrees(o.Omega) + np.degrees(o.omega)) % 360
    times = np.arange(1, n_snap + 1) * snap_dt
    d_pomega = np.array([np.polyfit(times, np.unwrap(np.radians(pomega_hist[:, k])), 1)[0]
                          for k in range(len(body_indices))])
    d_Omega = np.array([np.polyfit(times, np.unwrap(np.radians(Omega_hist[:, k])), 1)[0]
                         for k in range(len(body_indices))])
    return d_pomega, d_Omega


if __name__ == "__main__":
    from planetx.constants import GIANT_PLANETS, EARTH_MASS_IN_MSUN

    names = list(GIANT_PLANETS.keys()) + ["hpx"]
    a_arr = np.array([GIANT_PLANETS[n]["a"] for n in names[:-1]] + [400.0])
    e_arr = np.array([GIANT_PLANETS[n]["e"] for n in names[:-1]] + [0.3])
    inc_arr = np.array([GIANT_PLANETS[n]["inc"] for n in names[:-1]] + [20.0])
    Omega_arr = np.array([GIANT_PLANETS[n]["Omega"] for n in names[:-1]] + [0.0])
    omega_arr = np.array([GIANT_PLANETS[n]["omega"] for n in names[:-1]] + [0.0])
    m_arr = np.array([GIANT_PLANETS[n]["m"] for n in names[:-1]] + [5.0 * EARTH_MASS_IN_MSUN])

    print("=== Sanity check 1: reduce to single-perturber (Neptune only) ===")
    idx_nep = names.index("neptune")
    A1, B1 = build_AB_matrices(a_arr[[idx_nep]], m_arr[[idx_nep]])
    print(f"1-body A (should be 0, no other perturber): {A1[0,0]:.6e}")

    print("\n=== Sanity check 2: Laplace relation B_ii = -A_ii (5-body) ===")
    A5, B5 = build_AB_matrices(a_arr, m_arr)
    print("max |B_ii + A_ii| over all i:", np.max(np.abs(np.diag(B5) + np.diag(A5))))

    print("\n=== Eigenvalues (5-body massive system) ===")
    g, Ve = eigen_solution(A5)
    f, Vi = eigen_solution(B5)
    print("eccentricity eigenvalues (deg/Myr):", np.degrees(g.real) * 1e6)
    print("max |Im(g)| (should be ~0 for a real physical system):", np.max(np.abs(g.imag)))
    print("inclination eigenvalues (deg/Myr):", np.degrees(f.real) * 1e6)
    print("max |Im(f)|:", np.max(np.abs(f.imag)))

    print("\n=== Validating against REBOUND N-body (5-body, no test particles) ===")
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    sim.add(m=1.0, name="sun")
    for i, name in enumerate(names):
        sim.add(m=m_arr[i], a=a_arr[i], e=e_arr[i], inc=np.radians(inc_arr[i]),
                Omega=np.radians(Omega_arr[i]), omega=np.radians(omega_arr[i]), M=0.0, name=name)
    sim.N_active = len(sim.particles)
    sim.move_to_com()
    sim.integrator = "whfast"
    sim.integrator.safe_mode = 0
    sim.dt = min(0.5, a_arr[0] ** 1.5 / 20.0)

    # HPX's predicted period is ~15 Myr (rate ~0.02 deg/Myr) -- a 2e5 yr
    # window only samples ~1.3% of a cycle, too short for a reliable rate
    # fit (dominated by short-period jitter, not the secular trend). Use a
    # longer baseline so every body's rate is measured over a meaningful
    # fraction of its own cycle.
    T_TOTAL = 3.0e6
    n_snap = 600
    t0 = time.perf_counter()
    d_pomega_meas, d_Omega_meas = measure_rebound_precession(sim, list(range(1, 6)), T_TOTAL, n_snap)
    print(f"REBOUND 5-body integration ({T_TOTAL:.0f} yr): {time.perf_counter()-t0:.2f}s")

    # Sample at the SAME cadence as the REBOUND measurement and unwrap +
    # linear-fit identically, rather than a naive 2-point difference (which
    # would alias if any eigenmode completes more than half a cycle within
    # T_TOTAL -- some modes can be much faster than the single-perturber
    # case already tested).
    z0 = hk_from_elements(e_arr, Omega_arr + omega_arr)
    z0c = z0[0] + 1j * z0[1]
    sample_times = np.arange(1, n_snap + 1) * (T_TOTAL / n_snap)
    zt = propagate_massive_hk(g, Ve, z0c, sample_times)  # [5, n_snap]
    pomega_pred_series = np.arctan2(zt.real, zt.imag)  # [5, n_snap]
    d_pomega_pred = np.array([
        np.polyfit(sample_times, np.unwrap(pomega_pred_series[i]), 1)[0]
        for i in range(len(names))
    ])

    print("\nbody         predicted d(pomega)/dt   measured d(pomega)/dt   rel. error")
    for i, name in enumerate(names):
        pred = np.degrees(d_pomega_pred[i]) * 1e6
        meas = np.degrees(d_pomega_meas[i]) * 1e6
        relerr = abs(pred - meas) / max(abs(meas), 1e-10) * 100
        print(f"{name:10s}  {pred:14.3f} deg/Myr   {meas:14.3f} deg/Myr   {relerr:6.1f}%")
