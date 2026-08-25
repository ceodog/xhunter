"""Shared utilities for the GPU (CUDA) simulation backends -- gpu_nbody.py
and gpu_secular.py. The one piece of physics genuinely new to this trio
(not just a port of already-validated code) lives here: state_to_elements,
converting the GPU N-body kernel's Cartesian (r, v) output into the
classical osculating orbital elements (a, e, i, Omega, omega, M) that
worker.py's REBOUND path already produces via p.orbit(primary=...) and
that selection.SelectionFunction.apply/downstream featurelib expect.

state_to_elements is standard two-body osculating-element recovery
(Vallado/Curtis-style formulas, robust atan2 forms where available to
avoid arccos quadrant ambiguity) -- NOT re-derived physics, but also not
yet cross-checked against REBOUND's own convention the way every other
formula in this project has been before being trusted. Validate against
rebound.Particle.orbit() for a batch of random states before relying on
this for a real dataset (see gpu_nbody.py's module docstring for the
validation-session status).
"""

from __future__ import annotations

import numpy as np


def state_to_elements(r: np.ndarray, v: np.ndarray, gm) -> dict[str, np.ndarray]:
    """Heliocentric Cartesian (r, v) -> osculating elements (a, e, i, Omega,
    omega, M), vectorized over a leading batch dimension. r, v: [..., 3]
    AU / AU/yr. gm: scalar or [...] (this project's ('yr','AU','Msun')
    units -- G*(Msun+m) for a massive body, G*Msun for a massless one, same
    convention as blockpersim_dev.py/gpu_nbody.py's gm_massive/GM_SUN).

    Returns a, e, i, Omega, omega, M -- angles in RADIANS (caller converts
    to degrees mod 360 to match worker.run_one's hpx_final/tnos convention,
    same as worker.py itself does with REBOUND's o.inc/o.Omega/o.omega/o.M).
    i is always in [0, pi]; Omega, omega, M in [0, 2*pi) after the final
    mod-2pi below.
    """
    r = np.asarray(r, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    gm = np.asarray(gm, dtype=np.float64)

    r_mag = np.linalg.norm(r, axis=-1)
    v_mag2 = np.sum(v * v, axis=-1)

    h = np.cross(r, v)  # specific angular momentum
    h_mag = np.linalg.norm(h, axis=-1)

    # node vector n = z_hat x h = (-h_y, h_x, 0)
    n = np.stack([-h[..., 1], h[..., 0], np.zeros_like(h[..., 0])], axis=-1)
    n_mag = np.linalg.norm(n, axis=-1)

    # eccentricity vector e_vec = (v x h)/gm - r/|r|
    vxh = np.cross(v, h)
    e_vec = vxh / gm[..., None] - r / r_mag[..., None]
    e = np.linalg.norm(e_vec, axis=-1)

    energy = 0.5 * v_mag2 - gm / r_mag
    a = -gm / (2.0 * energy)

    i = np.arccos(np.clip(h[..., 2] / h_mag, -1.0, 1.0))

    Omega = np.arctan2(n[..., 1], n[..., 0])

    cos_omega = np.sum(n * e_vec, axis=-1) / (n_mag * e)
    omega = np.arccos(np.clip(cos_omega, -1.0, 1.0))
    omega = np.where(e_vec[..., 2] < 0.0, 2.0 * np.pi - omega, omega)

    cos_nu = np.sum(e_vec * r, axis=-1) / (e * r_mag)
    rv = np.sum(r * v, axis=-1)
    nu = np.arccos(np.clip(cos_nu, -1.0, 1.0))
    nu = np.where(rv < 0.0, 2.0 * np.pi - nu, nu)

    # true anomaly -> eccentric anomaly -> mean anomaly (robust half-angle form)
    E = 2.0 * np.arctan2(np.sqrt(np.maximum(1.0 - e, 0.0)) * np.sin(nu / 2.0),
                          np.sqrt(np.maximum(1.0 + e, 0.0)) * np.cos(nu / 2.0))
    M = E - e * np.sin(E)

    two_pi = 2.0 * np.pi
    return {
        "a": a, "e": e, "i": i,
        "Omega": Omega % two_pi, "omega": omega % two_pi, "M": M % two_pi,
    }
