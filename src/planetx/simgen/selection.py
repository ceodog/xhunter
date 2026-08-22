"""Survey selection forward model: turns REBOUND's raw final-epoch population
into what a real survey would actually have detected.

This module ships a simplified, illustrative selection function (smooth
magnitude-efficiency falloff + sky-coverage fraction + tracking efficiency)
so the end-to-end pipeline runs out of the box. Its constants are calibrated
against Rubin Observatory/LSST, via Sorcha's own published survey
characterization and the LSST TNO-yield forecast (see
SimpleSelectionFunction's docstring for sourcing; supersedes an earlier
OSSOS-based calibration), but its *mechanism* is still not a real survey
simulator: detection here is a flat, position- and epoch-independent
probability, not a check against actual pointing history on the sky at a
real observation date. For a production system this MUST still be replaced
by a real, characterized survey simulator -- Sorcha
(https://github.com/dirac-institute/sorcha) for Rubin/LSST, or the OSSOS
Survey Simulator (https://github.com/OSSOS/SurveySimulator) -- both of
which use actual pointing history rather than a uniform sky fraction.
Skipping this step, or leaving it at the stub below, is the single most
likely way for this whole pipeline to produce an unusable posterior: the
network would learn the stub's artificial biases instead of a real
survey's. See README.md, "Selection function: why this is the load-bearing
piece".

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


def _sample_H(rng: np.random.Generator, h_min: float, h_max: float, slope: float) -> float:
    """Draw an absolute magnitude from a single power-law luminosity function,
    dN/dH ~ 10^(slope*H), via inverse-CDF sampling -- the standard shape used
    for debiased KBO/TNO size distributions (many more small/faint objects
    than large/bright ones), rather than a flat draw across [h_min, h_max].

    slope (often called "alpha" in the literature) is NOT precisely
    constrained for the ETNO population specifically -- sample sizes are
    tiny (order 10) and it's degenerate with the (also poorly known)
    detection efficiency at extreme distances. Commonly-cited debiased KBO
    slope estimates (~0.7-0.9) are fit over a narrow magnitude range in
    their source studies; naively applied across this project's much wider
    (h_min, h_max) span, they blow up (e.g. slope=0.7 over an 11-magnitude
    range gives a ~5x10^7 density ratio between the faint and bright ends,
    collapsing essentially every draw to h_max -- verified empirically
    while tuning this). ~0.1 keeps the same qualitative shape (more faint
    objects than bright ones) without that extrapolation blowup; treat it
    as a tunable approximation, not an authoritative literature figure.
    """
    u = rng.uniform()
    lo, hi = 10 ** (slope * h_min), 10 ** (slope * h_max)
    return float(np.log10(u * (hi - lo) + lo) / slope)


@dataclass
class SimpleSelectionFunction:
    """Illustrative stand-in for a real survey simulator. See module docstring.

    sky_fraction, the magnitude-efficiency shape, and tracking_efficiency are
    calibrated against Rubin Observatory/LSST (via Sorcha, its official
    survey-simulator, and the published LSST TNO-yield forecast), superseding
    an earlier OSSOS calibration -- OSSOS is a small, purpose-built discovery
    survey (155 sq deg) with a cleanly citable characterized area; Rubin's
    ~18,000-20,000 sq deg Wide-Fast-Deep footprint is not a dedicated TNO
    survey the same way, so sky_fraction here rests on a different, still
    principled, sourcing method -- see below. As with the OSSOS calibration,
    the *defaults* are now realistic even though the *mechanism* (flat,
    position-independent sky fraction; no rate-of-motion cut) still isn't a
    real survey simulator.

    - detection_mag50 = 24.5: single-visit r-band 5-sigma depth (SMTN-002,
      "Calculating LSST limiting magnitudes and SNR", gives 24.52 at dark
      sky/zenith; Kurlander et al. 2025 use 24.0 as a "best-case" figure for
      their yield forecast). Deliberately the *single-visit* depth, not the
      ~27.5 mag 10-year coadd (Ivezic et al. 2019) -- TNOs move between
      visits, so discovery requires linking separate single-visit detections
      into tracklets, not a static-sky coadd; single-visit depth is the
      correct analog to OSSOS's own per-exposure characterization.
    - detection_width = 0.1, detection_peak = 1.0: Sorcha's own
      `fading_function_width`/`fading_function_peak_efficiency` defaults
      (github.com/dirac-institute/sorcha, `data/survey_setups/
      Rubin_full_footprint.ini`, `[FADINGFUNCTION]`), after Veres & Chesley
      (2017) -- a markedly sharper falloff than OSSOS's width=0.4, i.e.
      Rubin's per-visit efficiency curve turns off faster near the limit.
    - tracking_efficiency = 0.95: Sorcha's `SSP_detection_efficiency`
      default, matching LSST's own Solar System Processing design
      requirement (>=95% linking success given >=2 detections/night on
      >=3 nights within 15 days). NOT apples-to-apples with OSSOS's 0.998:
      OSSOS's number is "of what we detected at all, how many got a full
      orbit" (post-hoc); Rubin's 0.95 is "of what already met a cadence
      requirement, how many the automated pipeline links" -- a different
      pipeline stage. This module has no separate cadence/tracklet model to
      apply that requirement to, so 0.95 is used as the best available
      single-number analog, not a fully equivalent quantity.
    - sky_fraction = 0.129: NOT a direct geometric measurement the way
      OSSOS's 155/41253 was -- Rubin's footprint has no single citable
      "TNO-relevant area" (flagged explicitly during research; the WFD
      footprint is not built around the ecliptic the way a dedicated TNO
      survey is, and the Northern Ecliptic Spur minisurvey exists
      specifically to patch that gap). Naively using the full ~18,000-20,000
      sq deg footprint (155/41253-equivalent ~= 0.44-0.49) overpredicts
      badly: verified directly (this project's own selection-function code,
      a reference disk population, sky_fraction confirmed empirically linear
      in detection rate) that 0.44-0.49 implies ~150-167x more detections
      than OSSOS for the same population, versus the real, published ~44x
      more TNOs LSST is forecast to discover (37,000 vs. OSSOS's 838,
      Kurlander et al. 2025, "Predictions of the LSST Solar System Yield",
      arXiv:2506.02487). sky_fraction=0.129 is calibrated to reproduce that
      real 44x ratio with the other Rubin-sourced parameters above held
      fixed -- a legitimate way to resolve a genuinely underdetermined
      parameter (match a real aggregate outcome), but it assumes this
      project's synthetic disk's magnitude/orbital-element distribution is a
      reasonable stand-in for the real population Kurlander et al. modeled,
      which was not independently checked. Primary source for the Sorcha
      methodology: Merritt et al. 2025, "Sorcha: A Solar System Survey
      Simulator for LSST", AJ (arXiv:2506.02804).
    - detection_mag50 / detection_width / detection_peak use a smooth
      logistic falloff, `peak / (1 + exp((V - mag50) / width))`, matching
      Sorcha's own functional form exactly (not just its shape, unlike the
      earlier OSSOS calibration, which could not confirm OSSOS's own more
      complex double quadratic-times-logistic parameter ordering).
      `limiting_mag` is kept as the field name (aliasing detection_mag50)
      so the 3-number `survey_meta` schema (`sky_fraction, limiting_mag,
      tracking_efficiency`) consumed by `featurelib.build_feature_set` and
      `PosteriorNet`'s `survey_meta_dim=3` doesn't have to change.

    What's still NOT modeled, and can't be without a larger redesign: real
    detection efficiency also depends on an object's on-sky rate of motion
    (too-fast movers streak/shear out of a field, too-slow movers aren't
    distinguishable from background sources over the survey's cadence) --
    OSSOS's own characterization files include an explicit `rate_cut`. This
    project's simulator has no notion of on-sky angular rate at all (worker.py
    only outputs heliocentric osculating elements, never an observer-relative
    sky-plane projection), and a rate cut requires knowing a specific
    observation cadence/baseline in real calendar time. More fundamentally,
    true geometric survey-simulator matching (checking whether an object's
    actual RA/Dec falls inside a real pointing at a real observation date)
    is blocked by this project's own epoch design: every simulation's final
    state is deliberately NOT anchored to a real calendar date (see
    README.md, "The primordial disk and dynamical relaxation" -- t=0 seeds
    from J2000 elements, then integrates ~4.5 Gyr into an arbitrary future
    epoch with no calendar meaning), so there's no real date to look up
    OSSOS's actual pointings against. Wiring in a real geometric survey
    simulator would require resolving that epoch question first, not just
    swapping this function out.

    absolute_mag_range default (1.0, 12.0) and luminosity_function_slope are
    calibrated against real JPL SBDB data for 8 securely-classified ETNOs
    (a > 150 AU, q > 30 AU): Sedna (H=1.50), 2012 VP113 (4.05), 2000 CR105
    (6.14), 2015 TG387 (5.57), 2004 VN112 (6.46), 2007 TG422 (6.47),
    2010 GB174 (6.74), 2013 FT28 (7.20) -- observed range 1.5-7.2. The
    range here is deliberately wider than that: 1.0 so the brightest known
    class (Sedna-like) is reachable at all (a flat (4.0, 9.0) range used
    previously made this structurally impossible), and 12.0 to represent
    the true underlying population, most of which is too faint to have
    ever been detected -- using the detected sample's own range as the
    prior would be circular, since it's already brightness-selected by
    construction (see README.md/chat history for the full reasoning).
    """

    sky_fraction: float = 0.129
    limiting_mag: float = 24.5  # detection_mag50: the logistic's 50%-efficiency magnitude
    tracking_efficiency: float = 0.95
    detection_width: float = 0.1
    detection_peak: float = 1.0
    absolute_mag_range: tuple[float, float] = (1.0, 12.0)
    luminosity_function_slope: float = 0.1

    def _detection_efficiency(self, V: float) -> float:
        """Smooth logistic falloff, not a hard cutoff -- see class docstring."""
        return self.detection_peak / (1.0 + np.exp((V - self.limiting_mag) / self.detection_width))

    def apply(self, tnos: list[dict], rng: np.random.Generator) -> FeatureSet:
        detected = []
        for obj in tnos:
            if rng.uniform() > self.sky_fraction:
                continue

            H = _sample_H(rng, *self.absolute_mag_range, self.luminosity_function_slope)
            M_deg = obj.get("M", rng.uniform(0, 360))
            r = _heliocentric_distance(obj["a"], obj["e"], M_deg)
            delta = max(r - 1.0, 0.1)  # crude geocentric distance near opposition
            V = H + 5 * np.log10(max(r * delta, 1e-6))

            if rng.uniform() > self._detection_efficiency(V):
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
