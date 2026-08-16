"""Survey selection forward model: turns REBOUND's raw final-epoch population
into what a real survey would actually have detected.

This module ships a simplified, illustrative selection function (magnitude
limit + sky-coverage fraction + tracking efficiency) so the end-to-end
pipeline runs out of the box. For a production system this MUST be replaced
by a real, characterized survey simulator -- the OSSOS Survey Simulator
(https://github.com/OSSOS/SurveySimulator) or Sorcha for Rubin/LSST -- both
of which use actual pointing history rather than a uniform sky fraction.
Skipping this step, or leaving it at the stub below, is the single most
likely way for this whole pipeline to produce an unusable posterior: the
network would learn the stub's artificial biases instead of a real survey's.
See README.md, "Selection function: why this is the load-bearing piece".

Any replacement need only implement the same `apply()` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from planetx.featurelib import FeatureSet, build_feature_set


class SelectionFunction(Protocol):
    def apply(self, tnos: list[dict], rng: np.random.Generator) -> FeatureSet: ...


def _solve_kepler(M_deg: np.ndarray, e: np.ndarray, tol: float = 1e-8, max_iter: int = 50) -> np.ndarray:
    """Newton's method for eccentric anomaly E from mean anomaly M and eccentricity e."""
    M = np.radians(M_deg)
    E = M.copy()
    for _ in range(max_iter):
        dE = (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
        E = E - dE
        if np.max(np.abs(dE)) < tol:
            break
    return E


def _heliocentric_distance(a: float, e: float, M_deg: float) -> float:
    E = _solve_kepler(np.array([M_deg]), np.array([e]))[0]
    return a * (1 - e * np.cos(E))


def _uncertainty_for(V: float) -> dict[str, float]:
    """Fainter detections get noisier, shorter-arc orbit fits (a rough proxy;
    real uncertainty should come from an actual orbit-fit covariance, as in
    planetx.obsdata.orbitfit).
    """
    scale = 1.0 + max(0.0, V - 21.0)
    return {
        "a": 5.0 * scale, "e": 0.02 * scale, "i": 0.5 * scale,
        "Omega": 0.5 * scale, "omega": 2.0 * scale,
    }


@dataclass
class SimpleSelectionFunction:
    """Illustrative stand-in for a real survey simulator. See module docstring."""

    sky_fraction: float = 0.03
    limiting_mag: float = 24.5
    tracking_efficiency: float = 0.8
    absolute_mag_range: tuple[float, float] = (4.0, 9.0)

    def apply(self, tnos: list[dict], rng: np.random.Generator) -> FeatureSet:
        detected = []
        for obj in tnos:
            if rng.uniform() > self.sky_fraction:
                continue

            H = rng.uniform(*self.absolute_mag_range)
            M_deg = obj.get("M", rng.uniform(0, 360))
            r = _heliocentric_distance(obj["a"], obj["e"], M_deg)
            delta = max(r - 1.0, 0.1)  # crude geocentric distance near opposition
            V = H + 5 * np.log10(max(r * delta, 1e-6))

            if V > self.limiting_mag:
                continue
            if rng.uniform() > self.tracking_efficiency:
                continue

            detected.append({
                "a": obj["a"], "e": obj["e"], "i": obj["i"],
                "Omega": obj["Omega"], "omega": obj["omega"],
                "H_mag": H,
                "sigma": _uncertainty_for(V),
            })

        meta = {
            "sky_fraction": self.sky_fraction,
            "limiting_mag": self.limiting_mag,
            "tracking_efficiency": self.tracking_efficiency,
        }
        return build_feature_set(detected, meta)
