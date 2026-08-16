"""Turn a real catalog pull into network-ready x -- using the SAME
featurelib.build_feature_set() the simulation pipeline calls in
planetx.simgen.selection, so training and inference never diverge in
feature engineering. See README.md, "Feature parity".
"""

from __future__ import annotations

from planetx.featurelib import FeatureSet, build_feature_set
from planetx.obsdata.fetch import CatalogConnector, RawOrbit
from planetx.obsdata.orbitfit import get_elements_with_covariance


def _to_object_dict(orbit: RawOrbit) -> dict:
    return {
        "a": orbit.elements["a"],
        "e": orbit.elements["e"],
        "i": orbit.elements["i"],
        "Omega": orbit.elements["Omega"],
        "omega": orbit.elements["omega"],
        "H_mag": orbit.H_mag if orbit.H_mag is not None else 6.0,
        "sigma": orbit.sigma,
    }


def build_x_from_designations(designations: list[str], survey_meta: dict) -> FeatureSet:
    """designations: real object identifiers, ideally drawn from a survey
    with a known selection function.
    survey_meta: that survey's own selection_function_metadata() -- never
    invented here, since a mismatched selection function silently breaks
    the validity of the trained network's posterior on real data.
    """
    orbits = [get_elements_with_covariance(d) for d in designations]
    objects = [_to_object_dict(o) for o in orbits]
    return build_feature_set(objects, survey_meta)


def build_x_from_connector(connector: CatalogConnector) -> FeatureSet:
    """Preferred entry point once a real connector (OSSOS/DES/Rubin) is wired
    up: designations and selection_function_metadata come from the same
    characterized survey, which is the whole point.
    """
    designations = connector.fetch_all()
    survey_meta = connector.selection_function_metadata()
    return build_x_from_designations(designations, survey_meta)
