"""Orbit determination and uncertainty extraction for real objects.

Primary path: JPL SBDB already provides fitted elements + per-element sigma
for well-observed objects (planetx.obsdata.fetch.query_sbdb). The fallback
path (refit_from_astrometry) is a stub for objects lacking a usable SBDB
solution -- wiring it up requires an actual orbit-determination package
(OpenOrb, find_orb) fed with archival astrometry, optionally extended via
precovery, which is beyond this scaffold's scope.
"""

from __future__ import annotations

from planetx.obsdata.fetch import RawOrbit, query_sbdb


def get_elements_with_covariance(designation: str) -> RawOrbit:
    orbit = query_sbdb(designation)
    if not orbit.sigma or any(s <= 0 for s in orbit.sigma.values()):
        return refit_from_astrometry(designation)
    return orbit


def refit_from_astrometry(designation: str, include_precovery: bool = True) -> RawOrbit:
    """Fallback orbit fit for objects with no usable SBDB uncertainty.

    TODO: pull archival astrometry (optionally extended via a precovery
    search in archival sky-survey images) and fit with OpenOrb or find_orb.
    Left unimplemented in this scaffold.
    """
    raise NotImplementedError(
        f"{designation!r} has no usable SBDB covariance; wire up an "
        "OpenOrb/find_orb refit from archival astrometry here"
    )
