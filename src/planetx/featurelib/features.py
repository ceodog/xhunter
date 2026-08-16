"""Build the network-ready feature set x from a population of objects.

x is a *set*, not a sequence: object order carries no meaning, and the
population size varies (simulated survey yield, or however many real TNOs a
given catalog contains). Consumers must treat it as permutation-invariant
(see planetx.model.encoder.SetTransformerEncoder, which deliberately omits
positional encoding for this reason).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from planetx.constants import OBJECT_FEATURE_KEYS

# Default uncertainty assumed for elements lacking an explicit covariance
# (e.g. freshly "discovered" objects in a simulated survey). Rough proxy for
# a short-arc orbit fit; real objects should carry a fitted covariance
# instead (see planetx.obsdata.orbitfit).
DEFAULT_SIGMA = {"a": 5.0, "e": 0.02, "i": 0.5, "Omega": 0.5, "omega": 2.0}


@dataclass
class FeatureSet:
    """Network-ready representation of one population snapshot.

    features:     [N, F] float32 array, columns ordered per OBJECT_FEATURE_KEYS
    survey_meta:  [M] float32 array describing the selection function that
                  produced this population (sky coverage fraction, limiting
                  magnitude, mean tracking efficiency, ...)
    n_objects:    N, kept explicitly since callers may pad/batch this set
    """

    features: np.ndarray
    survey_meta: np.ndarray
    n_objects: int

    def as_padded(self, max_n: int) -> tuple[np.ndarray, np.ndarray]:
        """Pad to [max_n, F] and return (padded_features, key_padding_mask).

        mask[i] = True means "padding, ignore" -- matches the convention
        expected by torch.nn.MultiheadAttention's key_padding_mask.
        """
        n, f = self.features.shape
        if n > max_n:
            raise ValueError(f"n_objects={n} exceeds max_n={max_n}")
        padded = np.zeros((max_n, f), dtype=np.float32)
        padded[:n] = self.features
        mask = np.ones(max_n, dtype=bool)
        mask[:n] = False
        return padded, mask


def _object_row(obj: dict) -> list[float]:
    sigma = obj.get("sigma", {})
    return [
        obj["a"], obj["e"], obj["i"], obj["Omega"], obj["omega"],
        sigma.get("a", DEFAULT_SIGMA["a"]),
        sigma.get("e", DEFAULT_SIGMA["e"]),
        sigma.get("i", DEFAULT_SIGMA["i"]),
        sigma.get("Omega", DEFAULT_SIGMA["Omega"]),
        sigma.get("omega", DEFAULT_SIGMA["omega"]),
        obj.get("H_mag", 6.0),
    ]


def build_feature_set(objects: list[dict], survey_meta: dict) -> FeatureSet:
    """objects: list of {"a","e","i","Omega","omega", optional "sigma": {...}, "H_mag"}
    survey_meta: {"sky_fraction", "limiting_mag", "tracking_efficiency"}

    Elements in degrees for i/Omega/omega, AU for a; this matches the
    convention used throughout planetx.simgen and planetx.obsdata so no
    conversion is needed at the boundary.
    """
    if len(objects) == 0:
        features = np.zeros((0, len(OBJECT_FEATURE_KEYS)), dtype=np.float32)
    else:
        rows = [_object_row(o) for o in objects]
        features = np.asarray(rows, dtype=np.float32)

    meta = np.array(
        [
            survey_meta.get("sky_fraction", 1.0),
            survey_meta.get("limiting_mag", 24.0),
            survey_meta.get("tracking_efficiency", 1.0),
        ],
        dtype=np.float32,
    )
    return FeatureSet(features=features, survey_meta=meta, n_objects=len(objects))
