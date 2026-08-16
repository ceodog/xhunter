"""Physical constants and fixed (non-inferred) Solar System parameters.

Units follow REBOUND's ('yr', 'AU', 'Msun') convention throughout this
project unless otherwise noted.

The giant planet elements below are approximate J2000 mean orbital elements,
adequate for generating a scaffold training set. For production runs, replace
`GIANT_PLANETS` with a JPL Horizons / DE440 query so the fixed bodies match
the best available ephemeris (see [[obsdata.orbitfit]] for the analogous
real-data lookup).
"""

from __future__ import annotations

EARTH_MASS_IN_MSUN = 3.0034896e-6
SUN_MASS_MSUN = 1.0

# name -> {m (Msun), a (AU), e, inc (deg), Omega (deg), omega (deg), M (deg)}
GIANT_PLANETS: dict[str, dict[str, float]] = {
    "jupiter": {
        "m": 954.79194e-6,
        "a": 5.2038,
        "e": 0.0489,
        "inc": 1.303,
        "Omega": 100.464,
        "omega": 273.867,
        "M": 20.020,
    },
    "saturn": {
        "m": 285.8860e-6,
        "a": 9.5826,
        "e": 0.0565,
        "inc": 2.485,
        "Omega": 113.665,
        "omega": 339.392,
        "M": 317.020,
    },
    "uranus": {
        "m": 43.66244e-6,
        "a": 19.2184,
        "e": 0.0457,
        "inc": 0.773,
        "Omega": 74.006,
        "omega": 96.998,
        "M": 142.238,
    },
    "neptune": {
        "m": 51.51389e-6,
        "a": 30.1104,
        "e": 0.0113,
        "inc": 1.770,
        "Omega": 131.784,
        "omega": 273.187,
        "M": 256.228,
    },
}

SOLAR_SYSTEM_AGE_YEARS = 4.5e9

# Order of the theta (HPX) parameter vector everywhere in this project.
THETA_KEYS: tuple[str, ...] = ("mass", "a", "e", "i", "Omega", "omega", "M")

# Per-object feature order produced by featurelib and consumed by the model.
OBJECT_FEATURE_KEYS: tuple[str, ...] = (
    "a", "e", "i", "Omega", "omega",
    "sigma_a", "sigma_e", "sigma_i", "sigma_Omega", "sigma_omega",
    "H_mag",
)
