"""Real CUDA (numba.cuda) port of planetx.simgen.secular's closed-form
Laplace-Lagrange disk-propagation backend. Ported line-for-line from
scripts/secular_cuda.py -- see that script's module docstring for the full
validation record (two real bugs found and fixed there: alphabar
interior/exterior swap, and eps_ij's missing +m[i] term in the matrix
build -- both caught by direct comparison against secular.py's own CPU
reference before this was trusted) and the A100 benchmark numbers this
port inherits (~4700x vs 1 CPU core at grid=256; cost independent of
integration duration since there's no time-stepping loop).

REQUIRES A REAL NVIDIA GPU and the optional `gpu` dependency group -- same
constraint as gpu_nbody.py, see that module's docstring. Not imported
anywhere in the default CPU pipeline; only orchestrate_gpu.py imports it.

Reuses secular.py's own tiny host-side helpers (massive_body_elements,
hk_from_elements, pq_from_elements) directly rather than re-deriving them
-- only the genuinely expensive pieces (the Laplace-coefficient-quadrature-
heavy matrix construction and per-particle propagation) are GPU kernels
here; the actual 5x5 eigendecomposition and the tiny per-sim
c_e=solve(Ve,z0)/c_i=solve(Vi,w0) hoist stay on the host (confirmed NOT a
bottleneck once the matrix construction itself moved to GPU -- see the
validation session's benchmark decomposition).

Inherits ALL of secular.py's documented accuracy caveats unchanged: this
is a faster version of the same opt-in/exploratory backend, not a
validated replacement for disk_backend="rebound". Re-read secular.py's
module docstring before using this for a real dataset.
"""

from __future__ import annotations

import math

import numpy as np
from numba import cuda, float64

from planetx.simgen.secular import GM_SUN, hk_from_elements, massive_body_elements, pq_from_elements

SEC_MAX_MASSIVE = 8  # compile-time shared-mem cap; project's n_massive is always 5 (4 giants + HPX)
N_LAPLACE_POINTS = 200_000  # matches secular.laplace_b's own n_points default exactly


@cuda.jit(device=True, inline=True)
def _laplace_b_dev(j, alpha):
    """b_{1.5}^{(j)}(alpha) via the same 200k-point periodic trapezoidal
    quadrature as secular.laplace_b -- periodic trapezoidal over a full
    period reduces exactly to dx * plain sum, confirmed by derivation and
    by direct GPU-vs-CPU comparison before trusting this loop form (see
    scripts/secular_cuda.py's validation session)."""
    dx = 2.0 * math.pi / N_LAPLACE_POINTS
    total = 0.0
    for k in range(N_LAPLACE_POINTS):
        psi = k * dx
        base = 1.0 - 2.0 * alpha * math.cos(psi) + alpha * alpha
        denom = base * math.sqrt(base)  # base**1.5, cheaper equivalent (s is always 1.5 here)
        total += math.cos(j * psi) / denom
    return total * dx / math.pi


@cuda.jit(device=True, inline=True)
def _cexp_dev(re, im):
    """exp(re + i*im) = exp(re) * (cos(im) + i*sin(im))."""
    m = math.exp(re)
    return m * math.cos(im), m * math.sin(im)


@cuda.jit
def build_AB_kernel(a_massive, m_massive, n_massive, GM_SUN, out_A, out_B):
    """GPU port of secular.build_AB_matrices. One block per simulation;
    threads 0..n_massive*(n_massive-1)-1 each own one ordered (i,j)
    off-diagonal pair (i!=j)."""
    sim = cuda.blockIdx.x
    tid = cuda.threadIdx.x
    n_pairs = n_massive * (n_massive - 1)

    sh_a = cuda.shared.array(SEC_MAX_MASSIVE, dtype=float64)
    sh_m = cuda.shared.array(SEC_MAX_MASSIVE, dtype=float64)
    sh_n = cuda.shared.array(SEC_MAX_MASSIVE, dtype=float64)
    sh_diag = cuda.shared.array(SEC_MAX_MASSIVE, dtype=float64)

    if tid < n_massive:
        sh_a[tid] = a_massive[sim, tid]
        sh_m[tid] = m_massive[sim, tid]
        sh_diag[tid] = 0.0
    cuda.syncthreads()
    if tid < n_massive:
        sh_n[tid] = math.sqrt(GM_SUN / (sh_a[tid] ** 3))
    cuda.syncthreads()

    if tid < n_pairs:
        i = tid // (n_massive - 1)
        j_raw = tid % (n_massive - 1)
        j = j_raw if j_raw < i else j_raw + 1  # skip j==i, matches the CPU loop's `if i==j: continue`

        ai = sh_a[i]; aj = sh_a[j]
        if ai < aj:
            alpha_ij = ai / aj
            alphabar_ij = alpha_ij
        else:
            alpha_ij = aj / ai
            alphabar_ij = 1.0
        # eps_ij = m[j] / (M_sun + m[i]) -- the +m[i] term is a real ~0.1%-level
        # correction here (body i is itself massive), unlike the per-particle
        # coupling kernel below where the particle is massless. See module docstring.
        eps_ij = sh_m[j] / (1.0 + sh_m[i])  # M_sun = 1.0
        b1 = _laplace_b_dev(1, alpha_ij)
        b2 = _laplace_b_dev(2, alpha_ij)
        coeff = eps_ij * alpha_ij * alphabar_ij
        out_A[sim, i, j] = -(sh_n[i] / 4.0) * coeff * b2
        out_B[sim, i, j] = (sh_n[i] / 4.0) * coeff * b1
        cuda.atomic.add(sh_diag, i, coeff * b1)

    cuda.syncthreads()
    if tid < n_massive:
        out_A[sim, tid, tid] = (sh_n[tid] / 4.0) * sh_diag[tid]
        out_B[sim, tid, tid] = -(sh_n[tid] / 4.0) * sh_diag[tid]


def build_AB_matrices_cuda(a_massive_batch: np.ndarray, m_massive_batch: np.ndarray, n_massive: int):
    """a_massive_batch, m_massive_batch: [S, n_massive]. Returns (A, B), each [S, n_massive, n_massive]."""
    S = a_massive_batch.shape[0]
    d_a = cuda.to_device(a_massive_batch)
    d_m = cuda.to_device(m_massive_batch)
    d_A = cuda.device_array((S, n_massive, n_massive))
    d_B = cuda.device_array((S, n_massive, n_massive))
    n_pairs = n_massive * (n_massive - 1)
    threads = max(n_pairs, n_massive)
    build_AB_kernel[S, threads](d_a, d_m, n_massive, GM_SUN, d_A, d_B)
    cuda.synchronize()
    return d_A.copy_to_host(), d_B.copy_to_host()


@cuda.jit
def secular_kernel(
    a_p_arr, e_p_arr, inc_p_arr, Omega_p_arr, omega_p_arr, M_p_arr,   # [S, n_test], degrees except a,e
    a_massive, m_massive,                                             # [S, n_massive]
    g_re, g_im, f_re, f_im,                                           # [S, n_massive] -- eigenvalues of A, B
    ce_re, ce_im, ci_re, ci_im,                                       # [S, n_massive] -- solve(Ve,z0), solve(Vi,w0)
    Ve_re, Ve_im, Vi_re, Vi_im,                                       # [S, n_massive, n_massive] -- eigenvectors
    n_massive, n_test, GM_SUN, t_years,
    out_e, out_inc, out_Omega, out_omega, out_M, out_valid,           # [S, n_test]
):
    """GPU port of secular.propagate_particle. One block per simulation,
    one thread per test particle -- no inter-particle interaction, no
    time-stepping loop (closed-form evaluation at t_years directly)."""
    sim = cuda.blockIdx.x
    p = cuda.threadIdx.x
    if p >= n_test:
        return

    sh_a = cuda.shared.array(SEC_MAX_MASSIVE, dtype=float64)
    sh_m = cuda.shared.array(SEC_MAX_MASSIVE, dtype=float64)
    if p < n_massive:
        sh_a[p] = a_massive[sim, p]
        sh_m[p] = m_massive[sim, p]
    cuda.syncthreads()

    a_p = a_p_arr[sim, p]
    e_p = e_p_arr[sim, p]
    inc_p_deg = inc_p_arr[sim, p]
    Omega_p_deg = Omega_p_arr[sim, p]
    omega_p_deg = omega_p_arr[sim, p]
    M_p_deg = M_p_arr[sim, p]

    n_p = math.sqrt(GM_SUN / (a_p * a_p * a_p))

    Aj = cuda.local.array(SEC_MAX_MASSIVE, dtype=float64)
    Bj = cuda.local.array(SEC_MAX_MASSIVE, dtype=float64)
    sum_b1 = 0.0
    for j in range(n_massive):
        a_j = sh_a[j]
        # alphabar_j = alpha_j if a_p < a_j (particle INTERIOR to perturber) else 1.0 --
        # verified line-by-line against secular.particle_coupling's own
        # np.where(a_p < a_j_arr, alpha_j, 1.0) (see module docstring: this exact
        # condition has a documented history of being reintroduced backwards).
        if a_j < a_p:
            alpha = a_j / a_p
            alphabar = 1.0
        else:
            alpha = a_p / a_j
            alphabar = alpha
        eps_j = sh_m[j]  # M_sun = 1.0; test particle is massless so M_sun+0=M_sun trivially
        b1 = _laplace_b_dev(1, alpha)
        b2 = _laplace_b_dev(2, alpha)
        coeff = eps_j * alpha * alphabar
        Aj[j] = -(n_p / 4.0) * coeff * b2
        Bj[j] = (n_p / 4.0) * coeff * b1
        sum_b1 += coeff * b1
    A_free = (n_p / 4.0) * sum_b1
    B_free = -A_free

    # === eccentricity (h,k) branch ===
    pomega0 = math.radians(Omega_p_deg + omega_p_deg)
    h0 = e_p * math.sin(pomega0)
    k0 = e_p * math.cos(pomega0)

    forced0_re = 0.0; forced0_im = 0.0
    forced_t_re = 0.0; forced_t_im = 0.0
    for l in range(n_massive):
        s_re = 0.0; s_im = 0.0
        for j in range(n_massive):
            s_re += Aj[j] * Ve_re[sim, j, l]
            s_im += Aj[j] * Ve_im[sim, j, l]
        cel_re = ce_re[sim, l]; cel_im = ce_im[sim, l]
        nu_re = cel_re * s_re - cel_im * s_im
        nu_im = cel_re * s_im + cel_im * s_re

        gl_re = g_re[sim, l]; gl_im = g_im[sim, l]
        denom_re = A_free - gl_re
        denom_im = -gl_im
        denom_abs = math.sqrt(denom_re * denom_re + denom_im * denom_im)
        if denom_abs < 1e-12:
            denom_re = 1e-12; denom_im = 0.0
        denom_abs2 = denom_re * denom_re + denom_im * denom_im

        ratio_re = (nu_re * denom_re + nu_im * denom_im) / denom_abs2
        ratio_im = (nu_im * denom_re - nu_re * denom_im) / denom_abs2

        forced0_re -= ratio_re
        forced0_im -= ratio_im

        exp_re, exp_im = _cexp_dev(t_years * gl_im, -t_years * gl_re)
        forced_t_re -= (ratio_re * exp_re - ratio_im * exp_im)
        forced_t_im -= (ratio_re * exp_im + ratio_im * exp_re)

    zfree0_re = h0 - forced0_re
    zfree0_im = k0 - forced0_im
    fexp_re, fexp_im = _cexp_dev(0.0, -A_free * t_years)
    freet_re = zfree0_re * fexp_re - zfree0_im * fexp_im
    freet_im = zfree0_re * fexp_im + zfree0_im * fexp_re

    zt_re = freet_re + forced_t_re
    zt_im = freet_im + forced_t_im
    e_t = math.sqrt(zt_re * zt_re + zt_im * zt_im)
    # elements_from_hk(h,k) uses arctan2(h,k) -- h is zt.real, k is zt.imag
    pomega_t = math.degrees(math.atan2(zt_re, zt_im)) % 360.0

    # === inclination (p,q) branch ===
    inc0_rad = math.radians(inc_p_deg)
    Omega0_rad = math.radians(Omega_p_deg)
    p0 = inc0_rad * math.sin(Omega0_rad)
    q0 = inc0_rad * math.cos(Omega0_rad)

    forced0i_re = 0.0; forced0i_im = 0.0
    forcedi_t_re = 0.0; forcedi_t_im = 0.0
    for l in range(n_massive):
        s_re = 0.0; s_im = 0.0
        for j in range(n_massive):
            s_re += Bj[j] * Vi_re[sim, j, l]
            s_im += Bj[j] * Vi_im[sim, j, l]
        cil_re = ci_re[sim, l]; cil_im = ci_im[sim, l]
        nu_re = cil_re * s_re - cil_im * s_im
        nu_im = cil_re * s_im + cil_im * s_re

        fl_re = f_re[sim, l]; fl_im = f_im[sim, l]
        denom_re = B_free - fl_re
        denom_im = -fl_im
        denom_abs = math.sqrt(denom_re * denom_re + denom_im * denom_im)
        if denom_abs < 1e-12:
            denom_re = 1e-12; denom_im = 0.0
        denom_abs2 = denom_re * denom_re + denom_im * denom_im

        ratio_re = (nu_re * denom_re + nu_im * denom_im) / denom_abs2
        ratio_im = (nu_im * denom_re - nu_re * denom_im) / denom_abs2

        forced0i_re -= ratio_re
        forced0i_im -= ratio_im

        exp_re, exp_im = _cexp_dev(t_years * fl_im, -t_years * fl_re)
        forcedi_t_re -= (ratio_re * exp_re - ratio_im * exp_im)
        forcedi_t_im -= (ratio_re * exp_im + ratio_im * exp_re)

    wfree0_re = p0 - forced0i_re
    wfree0_im = q0 - forced0i_im
    fiexp_re, fiexp_im = _cexp_dev(0.0, -B_free * t_years)
    freeti_re = wfree0_re * fiexp_re - wfree0_im * fiexp_im
    freeti_im = wfree0_re * fiexp_im + wfree0_im * fiexp_re

    wt_re = freeti_re + forcedi_t_re
    wt_im = freeti_im + forcedi_t_im
    inc_t = math.degrees(math.sqrt(wt_re * wt_re + wt_im * wt_im))
    # elements_from_pq(p,q) uses arctan2(p,q) -- p is wt.real, q is wt.imag
    Omega_t = math.degrees(math.atan2(wt_re, wt_im)) % 360.0

    valid = 1
    if not (e_t == e_t) or e_t >= 1.0 or not (inc_t == inc_t):  # NaN check via x!=x
        valid = 0

    omega_t = (pomega_t - Omega_t) % 360.0
    M_t = (M_p_deg + math.degrees(n_p * t_years)) % 360.0

    out_e[sim, p] = e_t
    out_inc[sim, p] = inc_t
    out_Omega[sim, p] = Omega_t
    out_omega[sim, p] = omega_t
    out_M[sim, p] = M_t
    out_valid[sim, p] = valid


def run_secular_ensemble_cuda(
    theta_list: list[dict], disk_list: list[dict], t_years: float, n_massive: int = 5,
) -> list[list[dict]]:
    """Full GPU-accelerated secular backend: build_AB_matrices on GPU
    (batched, one launch) -> tiny per-sim eig()/solve() on host (confirmed
    cheap, not a bottleneck) -> per-particle propagation on GPU (batched).

    theta_list[k]: HPX theta dict for sim k (see secular.massive_body_elements).
    disk_list[k]: disk dict {a,e,inc,Omega,omega,M} for sim k -- inc/Omega/
        omega/M in RADIANS, matching worker._sample_primordial_disk's
        convention. All disks must share the same n_test (one kernel
        launch = one block shape).

    Returns a list (one per sim) of lists of {a,e,i,Omega,omega,M} dicts
    (i/Omega/omega/M in degrees) -- same schema as secular.propagate_disk,
    silently dropping particles flagged unphysical (e>=1/non-finite).
    """
    S = len(theta_list)
    n_test = len(disk_list[0]["a"])

    a_massive = np.zeros((S, n_massive)); m_massive = np.zeros((S, n_massive))
    for k in range(S):
        massive = massive_body_elements(theta_list[k])
        a_massive[k] = massive["a"]; m_massive[k] = massive["m"]

    A_all, B_all = build_AB_matrices_cuda(a_massive, m_massive, n_massive)

    g_re = np.zeros((S, n_massive)); g_im = np.zeros((S, n_massive))
    f_re = np.zeros((S, n_massive)); f_im = np.zeros((S, n_massive))
    ce_re = np.zeros((S, n_massive)); ce_im = np.zeros((S, n_massive))
    ci_re = np.zeros((S, n_massive)); ci_im = np.zeros((S, n_massive))
    Ve_re = np.zeros((S, n_massive, n_massive)); Ve_im = np.zeros((S, n_massive, n_massive))
    Vi_re = np.zeros((S, n_massive, n_massive)); Vi_im = np.zeros((S, n_massive, n_massive))

    a_p = np.zeros((S, n_test)); e_p = np.zeros((S, n_test))
    inc_p = np.zeros((S, n_test)); Omega_p = np.zeros((S, n_test))
    omega_p = np.zeros((S, n_test)); M_p = np.zeros((S, n_test))

    for k in range(S):
        massive = massive_body_elements(theta_list[k])
        g, Ve = np.linalg.eig(A_all[k])
        f, Vi = np.linalg.eig(B_all[k])
        h0, k0 = hk_from_elements(massive["e"], massive["Omega"] + massive["omega"])
        z0 = h0 + 1j * k0
        p0, q0 = pq_from_elements(massive["inc"], massive["Omega"])
        w0 = p0 + 1j * q0
        c_e = np.linalg.solve(Ve, z0)
        c_i = np.linalg.solve(Vi, w0)

        g_re[k] = g.real; g_im[k] = g.imag
        f_re[k] = f.real; f_im[k] = f.imag
        ce_re[k] = c_e.real; ce_im[k] = c_e.imag
        ci_re[k] = c_i.real; ci_im[k] = c_i.imag
        Ve_re[k] = Ve.real; Ve_im[k] = Ve.imag
        Vi_re[k] = Vi.real; Vi_im[k] = Vi.imag

        disk = disk_list[k]
        a_p[k] = disk["a"]; e_p[k] = disk["e"]
        inc_p[k] = np.degrees(disk["inc"]); Omega_p[k] = np.degrees(disk["Omega"])
        omega_p[k] = np.degrees(disk["omega"]); M_p[k] = np.degrees(disk["M"])

    d = lambda x: cuda.to_device(x)
    d_a_p, d_e_p, d_inc_p, d_Omega_p, d_omega_p, d_M_p = d(a_p), d(e_p), d(inc_p), d(Omega_p), d(omega_p), d(M_p)
    d_a_m, d_m_m = d(a_massive), d(m_massive)
    d_g_re, d_g_im, d_f_re, d_f_im = d(g_re), d(g_im), d(f_re), d(f_im)
    d_ce_re, d_ce_im, d_ci_re, d_ci_im = d(ce_re), d(ce_im), d(ci_re), d(ci_im)
    d_Ve_re, d_Ve_im, d_Vi_re, d_Vi_im = d(Ve_re), d(Ve_im), d(Vi_re), d(Vi_im)

    out_e = cuda.device_array((S, n_test)); out_inc = cuda.device_array((S, n_test))
    out_Omega = cuda.device_array((S, n_test)); out_omega = cuda.device_array((S, n_test))
    out_M = cuda.device_array((S, n_test)); out_valid = cuda.device_array((S, n_test), dtype=np.int32)

    secular_kernel[S, n_test](
        d_a_p, d_e_p, d_inc_p, d_Omega_p, d_omega_p, d_M_p,
        d_a_m, d_m_m, d_g_re, d_g_im, d_f_re, d_f_im,
        d_ce_re, d_ce_im, d_ci_re, d_ci_im, d_Ve_re, d_Ve_im, d_Vi_re, d_Vi_im,
        n_massive, n_test, GM_SUN, t_years,
        out_e, out_inc, out_Omega, out_omega, out_M, out_valid,
    )
    cuda.synchronize()

    e_h = out_e.copy_to_host(); inc_h = out_inc.copy_to_host()
    Omega_h = out_Omega.copy_to_host(); omega_h = out_omega.copy_to_host()
    M_h = out_M.copy_to_host(); valid_h = out_valid.copy_to_host()

    results = []
    for k in range(S):
        rows = []
        for p in range(n_test):
            if valid_h[k, p]:
                rows.append({"a": float(a_p[k, p]), "e": float(e_h[k, p]), "i": float(inc_h[k, p]),
                             "Omega": float(Omega_h[k, p]), "omega": float(omega_h[k, p]), "M": float(M_h[k, p])})
        results.append(rows)
    return results
