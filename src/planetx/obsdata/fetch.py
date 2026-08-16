"""Real-catalog connectors.

JPLSBDBConnector / query_sbdb talk to the actual JPL Small-Body Database API
(https://ssd-api.jpl.nasa.gov/doc/sbdb.html) and are functional as written --
verified against a live query for (523794) 2015 RR245, whose `orbit.elements`
response carries per-element `value` and `sigma` fields keyed by `name` in
("e","a","q","i","om","w","ma",...). This module uses that per-element sigma
as a diagonal uncertainty, matching what planetx.featurelib expects; it does
not parse the full covariance matrix (`cov=mat`), which would be the natural
next step if off-diagonal terms turn out to matter.

The OSSOS / DES / Rubin connectors are stubs: they define the interface a
survey-catalog connector must implement, but the actual data pulls (from
each survey's public release / Sorcha output) are left as TODOs, since the
download endpoints and formats are survey-specific and evolve independently
of this project. CRITICALLY, only connectors backed by a published,
characterized selection function should be used for real inference -- see
planetx.simgen.selection's module docstring. JPL SBDB itself is a
compilation with no selection function, so JPLSBDBConnector deliberately
refuses to fabricate one (see its selection_function_metadata below).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import requests

SBDB_URL = "https://ssd-api.jpl.nasa.gov/sbdb.api"

# JPL SBDB element name -> planetx element name
_ELEMENT_MAP = {"a": "a", "e": "e", "i": "i", "om": "Omega", "w": "omega", "ma": "M"}


@dataclass
class RawOrbit:
    designation: str
    elements: dict[str, float]  # a, e, i, Omega, omega, M
    sigma: dict[str, float]  # matching keys, same units
    H_mag: float | None


def query_sbdb(designation: str, timeout: float = 15.0) -> RawOrbit:
    """Fetch osculating elements + per-element uncertainty for one object."""
    resp = requests.get(
        SBDB_URL, params={"sstr": designation, "full-prec": "true"}, timeout=timeout
    )
    resp.raise_for_status()
    data = resp.json()
    if "orbit" not in data:
        raise ValueError(f"SBDB has no orbit for {designation!r}: {data.get('message')}")

    elements: dict[str, float] = {}
    sigma: dict[str, float] = {}
    for el in data["orbit"]["elements"]:
        key = _ELEMENT_MAP.get(el["name"])
        if key is None:
            continue
        elements[key] = float(el["value"])
        sigma_val = el.get("sigma")
        sigma[key] = float(sigma_val) if sigma_val not in (None, "") else 0.0

    H_mag = None
    for p in data.get("phys_par", []):
        if p.get("name") == "H" and p.get("value") not in (None, ""):
            H_mag = float(p["value"])
            break

    return RawOrbit(designation=designation, elements=elements, sigma=sigma, H_mag=H_mag)


class CatalogConnector(Protocol):
    """Interface for a real-catalog connector with a KNOWN selection function."""

    def fetch_all(self) -> list[str]: ...
    def selection_function_metadata(self) -> dict: ...


@dataclass
class JPLSBDBConnector:
    """Pull elements/uncertainty for a fixed, user-supplied designation list.

    Use this to resolve orbits for a designation list assembled from a
    *known* survey (e.g. an OSSOS discovery list) -- pass that survey's own
    selection_function_metadata() alongside it, not this connector's (which
    intentionally raises, since SBDB carries no selection function itself).
    """

    designations: list[str]

    def fetch_all(self) -> list[str]:
        return list(self.designations)

    def selection_function_metadata(self) -> dict:
        raise NotImplementedError(
            "JPL SBDB is a compilation, not a characterized survey -- supply "
            "selection_function_metadata from the actual discovery survey "
            "(e.g. OSSOSConnector) instead of this connector"
        )


@dataclass
class OSSOSConnector:
    """Stub: OSSOS public release + its published Survey Simulator characterization.

    TODO: implement against the OSSOS data release
    (https://www.ossos-survey.org/data/) and the OSSOS Survey Simulator
    (https://github.com/OSSOS/SurveySimulator) for selection_function_metadata.
    """

    def fetch_all(self) -> list[str]:
        raise NotImplementedError("wire up the OSSOS data release download here")

    def selection_function_metadata(self) -> dict:
        raise NotImplementedError("wire up the OSSOS Survey Simulator characterization here")


@dataclass
class DESConnector:
    """Stub: Dark Energy Survey TNO discovery list + its published selection function."""

    def fetch_all(self) -> list[str]:
        raise NotImplementedError("wire up the DES TNO catalog download here")

    def selection_function_metadata(self) -> dict:
        raise NotImplementedError("wire up DES's published detection efficiency here")


@dataclass
class RubinConnector:
    """Stub: Rubin/LSST Solar System Object catalog + Sorcha-based selection function."""

    def fetch_all(self) -> list[str]:
        raise NotImplementedError("wire up the Rubin SSO catalog query here")

    def selection_function_metadata(self) -> dict:
        raise NotImplementedError("wire up a Sorcha-based characterization here")
