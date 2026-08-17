"""Physical constants and fixed (non-inferred) Solar System parameters.

Units follow REBOUND's ('yr', 'AU', 'Msun') convention throughout this
project unless otherwise noted.

The giant planet orbital elements below are J2000.0 epoch values from JPL
SSD's "Keplerian Elements for Approximate Positions of the Major Planets"
(Standish 1992/2006; https://ssd.jpl.nasa.gov/planets/approx_pos.html),
converted from that table's (a, e, I, L, long.peri., long.node.) convention
to this project's (a, e, inc, Omega, omega, M) via Omega = long.node.,
omega = long.peri. - Omega, M = L - long.peri. (mod 360). Masses are the
well-known IAU planet/Sun mass ratios. For production runs, consider
replacing `GIANT_PLANETS` with a live JPL Horizons / DE440 query instead of
this fixed table, so the fixed bodies match the best available ephemeris at
whatever epoch you actually integrate from (see [[obsdata.orbitfit]] for the
analogous real-data lookup).
"""

from __future__ import annotations

EARTH_MASS_IN_MSUN = 3.0034896e-6
SUN_MASS_MSUN = 1.0

# name -> {m (Msun), a (AU), e, inc (deg), Omega (deg), omega (deg), M (deg)}
GIANT_PLANETS: dict[str, dict[str, float]] = {
    "jupiter": {
        "m": 954.79194e-6,
        "a": 5.20288700,
        "e": 0.04838624,
        "inc": 1.30439695,
        "Omega": 100.47390909,
        "omega": 274.25457074,
        "M": 19.66796068,
    },
    "saturn": {
        "m": 285.8860e-6,
        "a": 9.53667594,
        "e": 0.05386179,
        "inc": 2.48599187,
        "Omega": 113.66242448,
        "omega": 338.93645383,
        "M": 317.35536592,
    },
    "uranus": {
        "m": 43.66244e-6,
        "a": 19.18916464,
        "e": 0.04725744,
        "inc": 0.77263783,
        "Omega": 74.01692503,
        "omega": 96.93735127,
        "M": 142.28382821,
    },
    "neptune": {
        "m": 51.51389e-6,
        "a": 30.06992276,
        "e": 0.00859048,
        "inc": 1.77004347,
        "Omega": 131.78422574,
        "omega": 273.18053653,
        "M": 259.91520804,
    },
}

# NOT "how long from t=0 (today's giant-planet elements) to reach real
# calendar 'now'" -- GIANT_PLANETS seeds every simulation with TODAY's
# elements at t=0, so integrating forward by this many years lands on a
# future epoch, not the present. What this value actually sets is "how
# long to let a synthetic disk dynamically relax under a representative
# giant-planet architecture" -- Gyr-scale, to develop realistic
# resonant/secular structure -- using the true solar system age as a
# defensible order-of-magnitude choice for "long enough," not a literal
# replay of history from an accurate primordial giant-planet configuration
# (that would require modeling giant-planet migration itself, e.g. the
# Nice model, which this project doesn't attempt). What DOES still hold:
# theta (HPX) and x (the TNO population) are read out at the same final
# simulated instant, so they remain a valid, mutually consistent training
# pair regardless of what that instant means in calendar terms. Not read
# programmatically (YAML can't import this) -- configs/prior.yaml's
# `simulation.integration_years` is kept numerically equal to it by
# convention; if you change one, change the other.
SOLAR_SYSTEM_AGE_YEARS = 4.5e9

# Order of the theta (HPX) parameter vector everywhere in this project.
THETA_KEYS: tuple[str, ...] = ("mass", "a", "e", "i", "Omega", "omega", "M")

# Per-object feature order produced by featurelib and consumed by the model.
OBJECT_FEATURE_KEYS: tuple[str, ...] = (
    "a", "e", "i", "Omega", "omega",
    "sigma_a", "sigma_e", "sigma_i", "sigma_Omega", "sigma_omega",
    "H_mag",
)
