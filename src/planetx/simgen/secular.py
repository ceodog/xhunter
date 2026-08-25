"""Linear (Laplace-Lagrange) secular perturbation theory for the primordial
TNO disk -- a closed-form, non-N-body alternative to full REBOUND
propagation of massless test particles. Selectable via configs/prior.yaml
`simulation.disk_backend: rebound | secular | hybrid` (see worker.run_one):

  - "rebound" (default): every test particle integrated by REBOUND, exactly
    as before. Validated, expensive.
  - "secular": every test particle propagated by the closed-form formula in
    this module. ~8800x faster than "rebound" alone (though the massive
    bodies -- Sun + giants + HPX -- still integrate via REBOUND regardless
    of disk_backend, since HPX's own final state feeds the training label
    and was never validated via the secular route; that integration's cost
    is independent of n_test_particles, ~0.13 days/simulation at this
    project's production integration_years).
  - "hybrid": test particles within a low-order Neptune mean-motion
    resonance (where linear theory does not apply) are routed to real
    REBOUND N-body alongside the massive bodies; the rest use the closed
    form. ~14-16x faster than "rebound" at this project's production scale.

Formulas: Murray & Dermott, "Solar System Dynamics" Ch. 7-8, cross-checked
against two independent secondary sources and REBOUND N-body directly (see
/Users/xtan/.claude/plans/robust-puzzling-flamingo.md, "Stage 3" and
"all-secular benchmark", for the full derivation/validation record this
module is ported from -- originally scripts/test_secular_theory.py,
test_secular_multibody.py, test_secular_step34.py).

ACCURACY -- measured, not assumed, and load-bearing, not a footnote:
  - Validated only against REBOUND out to 2e7 yr. This project's production
    integration_years=4.5e9 is 225x longer than anything actually tested.
  - Population-level (not individual-trajectory) statistics, even at 2e7 yr:
    eccentricity correlation with true N-body ~0.4, N-body's eccentricity
    spread measured ~2.5x WIDER than secular's (nonlinear diffusion this
    LINEAR theory structurally cannot capture).
  - "secular" (no triage) measurably fails on resonant particles
    specifically: in a direct 200-particle benchmark, the resonant subset's
    inclination correlation with N-body was NEGATIVE (r=-0.52) and
    eccentricity was underestimated ~3x. "hybrid" avoids this by routing
    those particles to real N-body instead.
  - Cannot model dynamical ejection: closed-form solutions here are bounded/
    oscillatory by construction (e < 1 always, barring the numerical guard
    below), so particles a real N-body integration would eject (~5% of a
    disk within just 2e7 yr, measured) are silently retained rather than
    dropped as unbound.
  - Mean anomaly (M) is NOT part of the validated comparison at all --
    propagated below via plain unperturbed two-body Kepler drift, which
    rests on Kepler's third law alone (a much better-established piece of
    physics than the secular formula itself, and one that does not
    secularly accumulate error the way precession does).

Every shipped config defaults to `disk_backend: rebound`. "secular"/"hybrid"
are opt-in, exploratory backends, not a validated drop-in replacement --
see README.md.
"""

from __future__ import annotations

import numpy as np
import rebound

from planetx.constants import EARTH_MASS_IN_MSUN, GIANT_PLANETS

GIANT_NAMES = ("jupiter", "saturn", "uranus", "neptune")


def _gm_sun() -> float:
    """REBOUND's own G in (yr, AU, Msun) units (~39.4769) -- read directly
    rather than hardcoded; it is NOT 1 and NOT exactly 4*pi**2."""
    sim = rebound.Simulation()
    sim.units = ("yr", "AU", "Msun")
    return sim.G


GM_SUN = _gm_sun()


def laplace_b(s: float, j: int, alpha: float, n_points: int = 200_000) -> float:
    """b_s^(j)(alpha) via direct numerical quadrature (composite trapezoidal
    rule over a full period -- spectrally accurate for this smooth periodic
    integrand), not a memorized series expansion."""
    psi = np.linspace(0, 2 * np.pi, n_points, endpoint=False)
    integrand = np.cos(j * psi) / (1 - 2 * alpha * np.cos(psi) + alpha**2) ** s
    val = np.trapezoid(np.append(integrand, integrand[0]), dx=2 * np.pi / n_points)
    return val / np.pi


def laplace_b_deriv(s: float, j: int, alpha: float) -> float:
    """D[b_s^j](alpha), standard recursion (Brouwer & Clemence 1961)."""
    return s * (laplace_b(s + 1, j - 1, alpha) - 2 * alpha * laplace_b(s + 1, j, alpha)
                + laplace_b(s + 1, j + 1, alpha))


def massive_body_elements(theta: dict) -> dict[str, np.ndarray]:
    """The 5-body (4 giants + HPX) architecture that drives the secular
    matrix, built from the same constants.GIANT_PLANETS + theta inputs
    worker._add_giant_planets/_add_hpx use -- their INITIAL elements are
    what linear secular theory's matrix construction needs (mean anomaly is
    irrelevant here; it never enters the secular matrix)."""
    a = np.array([GIANT_PLANETS[n]["a"] for n in GIANT_NAMES] + [theta["a"]])
    e = np.array([GIANT_PLANETS[n]["e"] for n in GIANT_NAMES] + [theta["e"]])
    inc = np.array([GIANT_PLANETS[n]["inc"] for n in GIANT_NAMES] + [theta["i"]])
    Omega = np.array([GIANT_PLANETS[n]["Omega"] for n in GIANT_NAMES] + [theta["Omega"]])
    omega = np.array([GIANT_PLANETS[n]["omega"] for n in GIANT_NAMES] + [theta["omega"]])
    m = np.array([GIANT_PLANETS[n]["m"] for n in GIANT_NAMES] + [theta["mass"] * EARTH_MASS_IN_MSUN])
    return {"a": a, "e": e, "inc": inc, "Omega": Omega, "omega": omega, "m": m}


def build_AB_matrices(a: np.ndarray, m: np.ndarray, M_sun: float = 1.0):
    """N-body Laplace-Lagrange secular matrices (Murray & Dermott Eq. 7.8-
    7.12 generalization):
      A_ii = +(n_i/4) sum_{j!=i} eps_ij alpha_ij alphabar_ij b1(alpha_ij)
      A_ij = -(n_i/4) eps_ij alpha_ij alphabar_ij b2(alpha_ij)   (i!=j)
      B_ii = -A_ii
      B_ij = +(n_i/4) eps_ij alpha_ij alphabar_ij b1(alpha_ij)   (i!=j, b1 not b2)
      eps_ij = m_j / (M_sun + m_i)
    """
    N = len(a)
    n = np.sqrt(GM_SUN / np.asarray(a) ** 3)
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


def eigen_solution(mat: np.ndarray):
    return np.linalg.eig(mat)


def hk_from_elements(e, pomega_deg):
    pomega = np.radians(pomega_deg)
    return e * np.sin(pomega), e * np.cos(pomega)


def elements_from_hk(h, k):
    e = np.sqrt(h**2 + k**2)
    pomega = np.degrees(np.arctan2(h, k)) % 360
    return e, pomega


def pq_from_elements(inc_deg, Omega_deg):
    """p,q = I*sin(Omega), I*cos(Omega); I in RADIANS (the direct linear
    analog of e, not sin(I) -- coincide for this disk's modest inclinations
    anyway, but matching the formula as derived)."""
    inc = np.radians(inc_deg)
    Omega = np.radians(Omega_deg)
    return inc * np.sin(Omega), inc * np.cos(Omega)


def elements_from_pq(p, q):
    inc = np.degrees(np.hypot(p, q))
    Omega = np.degrees(np.arctan2(p, q)) % 360
    return inc, Omega


def particle_coupling(a_p: float, a_j_arr: np.ndarray, m_j_arr: np.ndarray, M_sun: float = 1.0):
    """Massless test particle at a_p under massive bodies at a_j_arr.
    Returns (A_free, B_free, A_j[], B_j[]) -- own free rate + per-body
    forcing coefficients.

    alphabar_ij = alpha_ij if body i (here, the test particle itself) is
    interior to body j (the perturber), else 1 -- the same rule
    build_AB_matrices uses for massive-body-to-massive-body coupling. A
    previous version of this condition checked whether the PERTURBER was
    interior to the particle (backwards), which silently multiplied every
    coupling term by a spurious extra factor of alpha_j whenever the test
    particle was exterior to a perturber -- the common case for this
    project's disk (particles at 30-100 AU, perturbed mainly by Neptune at
    ~30 AU). Confirmed via direct comparison against the independently
    validated single-perturber rate formula (test_secular_theory.py's
    predicted_rates): the buggy version was off by exactly a factor of
    alpha (0.668 for a representative a_test=45, a_neptune=30.07 case).
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


def _forced_free(free_rate, coupling_j, eigvals, eigvecs, z0_massive, z0_p, t: float):
    """Shared closed-form structure (Murray & Dermott Sec. 7.5) for both the
    eccentricity (h,k) and inclination (p,q) forced+free solutions -- single
    epoch t only (this project never needs a trajectory, only a final
    snapshot -- see worker.py's own "Single-epoch snapshots" design note)."""
    c = np.linalg.solve(eigvecs, z0_massive)
    nu = np.array([np.sum(coupling_j * c[l] * eigvecs[:, l]) for l in range(len(eigvals))])
    denom = free_rate - eigvals
    # Secular-resonance guard: free_rate landing on an eigenvalue makes the
    # forced amplitude diverge. Not detected/triaged (out of scope) --
    # clamped so this returns a large-but-finite value that the e>=1 check
    # in propagate_particle below can catch, instead of NaN/Inf.
    denom = np.where(np.abs(denom) < 1e-12, 1e-12, denom)
    forced0 = -np.sum(nu / denom)
    z_free0 = z0_p - forced0
    forced_t = -np.sum((nu / denom) * np.exp(-1j * eigvals * t))
    free_t = z_free0 * np.exp(-1j * free_rate * t)
    return free_t + forced_t


def particle_forced_free_hk(a_p, a_j_arr, m_j_arr, g, V_e, z0_massive, h0_p, k0_p, t: float):
    A_free, _, A_j, _ = particle_coupling(a_p, a_j_arr, m_j_arr)
    return _forced_free(A_free, A_j, g, V_e, z0_massive, h0_p + 1j * k0_p, t)


def particle_forced_free_pq(a_p, a_j_arr, m_j_arr, f, V_i, w0_massive, p0_p, q0_p, t: float):
    _, B_free, _, B_j = particle_coupling(a_p, a_j_arr, m_j_arr)
    return _forced_free(B_free, B_j, f, V_i, w0_massive, p0_p + 1j * q0_p, t)


# ---------------------------------------------------------------------------
# Resonance-width triage (used by disk_backend="hybrid" only)
# ---------------------------------------------------------------------------


def resonance_width_coefficient(a_neptune: float, p: int, m_neptune: float, M_sun: float = 1.0):
    """Everything about the (p+1):p exterior first-order Neptune MMR width
    that does NOT depend on a particle's eccentricity -- compute once per
    resonance order, not per particle (Murray & Dermott Ch. 8 general q-th
    order width, via Wallace/Quinn/Boley 2021 Appendix B)."""
    a_res = a_neptune * ((p + 1) / p) ** (2.0 / 3.0)
    alpha = a_neptune / a_res
    n = np.sqrt(GM_SUN / a_res**3)
    j = p + 1
    fd = -j * laplace_b(0.5, j, alpha) - (alpha / 2.0) * laplace_b_deriv(0.5, j, alpha)
    Cr = (m_neptune / M_sun) * n * alpha * fd
    width_coeff = a_res * np.sqrt((16.0 / 3.0) * abs(Cr) / n)
    return a_res, width_coeff


def resonance_half_width(width_coeff: float, e) -> float:
    return width_coeff * np.sqrt(np.maximum(e, 0.01))


def resonant_mask(a_p: np.ndarray, e_p: np.ndarray, p_max: int = 7) -> np.ndarray:
    """True where a disk particle falls within a low-order (first-order,
    p=1..p_max) Neptune mean-motion-resonance width -- linear secular theory
    is not valid there; disk_backend="hybrid" routes these to real N-body
    instead of the closed form."""
    a_neptune = GIANT_PLANETS["neptune"]["a"]
    m_neptune = GIANT_PLANETS["neptune"]["m"]
    mask = np.zeros(len(a_p), dtype=bool)
    for p in range(1, p_max + 1):
        a_res, wc = resonance_width_coefficient(a_neptune, p, m_neptune)
        hw = resonance_half_width(wc, e_p)
        mask |= np.abs(a_p - a_res) < hw
    return mask


# ---------------------------------------------------------------------------
# Production entry points
# ---------------------------------------------------------------------------


class EigenSystem:
    """Precomputed eigendecomposition of the 5-body (4 giants + HPX)
    secular matrices -- build once per simulation, reuse for every disk
    particle in it."""

    __slots__ = ("g", "Ve", "f", "Vi", "z0", "w0", "a_massive", "m_massive")

    def __init__(self, massive: dict[str, np.ndarray]):
        A, B = build_AB_matrices(massive["a"], massive["m"])
        self.g, self.Ve = eigen_solution(A)
        self.f, self.Vi = eigen_solution(B)
        h0, k0 = hk_from_elements(massive["e"], massive["Omega"] + massive["omega"])
        self.z0 = h0 + 1j * k0
        p0, q0 = pq_from_elements(massive["inc"], massive["Omega"])
        self.w0 = p0 + 1j * q0
        self.a_massive = massive["a"]
        self.m_massive = massive["m"]


def propagate_particle(
    eigsys: EigenSystem, a_p: float, e_p: float,
    inc_p_deg: float, Omega_p_deg: float, omega_p_deg: float, M_p_deg: float,
    t_years: float,
) -> dict | None:
    """Closed-form secular state of one massless disk particle at t_years.
    Returns None if the result is unphysical (e>=1, e.g. from the
    secular-resonance guard above) -- caller should treat this like an
    ejected/unbound REBOUND particle and drop it."""
    h0, k0 = hk_from_elements(e_p, Omega_p_deg + omega_p_deg)
    zt = particle_forced_free_hk(a_p, eigsys.a_massive, eigsys.m_massive, eigsys.g, eigsys.Ve,
                                  eigsys.z0, h0, k0, t_years)
    e_t, pomega_t = elements_from_hk(zt.real, zt.imag)

    p0, q0 = pq_from_elements(inc_p_deg, Omega_p_deg)
    wt = particle_forced_free_pq(a_p, eigsys.a_massive, eigsys.m_massive, eigsys.f, eigsys.Vi,
                                  eigsys.w0, p0, q0, t_years)
    inc_t, Omega_t = elements_from_pq(wt.real, wt.imag)

    if not np.isfinite(e_t) or e_t >= 1.0 or not np.isfinite(inc_t):
        return None

    omega_t = (pomega_t - Omega_t) % 360.0
    n_p = np.sqrt(GM_SUN / a_p**3)  # unperturbed two-body mean motion
    M_t = (M_p_deg + np.degrees(n_p * t_years)) % 360.0  # plain Kepler drift, see module docstring

    return {"a": a_p, "e": float(e_t), "i": float(inc_t), "Omega": float(Omega_t),
            "omega": float(omega_t), "M": float(M_t)}


def propagate_disk(eigsys: EigenSystem, disk: dict[str, np.ndarray], indices, t_years: float) -> list[dict]:
    """Secular-propagate the disk particles at `indices` (an int array/list
    into disk's per-particle arrays). `disk` uses worker._sample_primordial_disk's
    convention: a, e in AU/dimensionless; inc, Omega, omega, M in RADIANS.
    Returns a list of {a,e,i,Omega,omega,M} dicts (i/Omega/omega/M in
    degrees) -- same schema as worker.run_one's REBOUND-path tnos, silently
    dropping any particle propagate_particle flags as unphysical."""
    out = []
    for idx in indices:
        res = propagate_particle(
            eigsys, float(disk["a"][idx]), float(disk["e"][idx]),
            float(np.degrees(disk["inc"][idx])), float(np.degrees(disk["Omega"][idx])),
            float(np.degrees(disk["omega"][idx])), float(np.degrees(disk["M"][idx])),
            t_years,
        )
        if res is not None:
            out.append(res)
    return out
