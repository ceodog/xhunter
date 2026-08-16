import numpy as np

from planetx.constants import OBJECT_FEATURE_KEYS
from planetx.featurelib import build_feature_set


def _obj(a=100.0, e=0.3, i=20.0, Omega=50.0, omega=10.0):
    return {"a": a, "e": e, "i": i, "Omega": Omega, "omega": omega, "H_mag": 6.5}


def test_build_feature_set_shapes():
    objects = [_obj(), _obj(a=200.0)]
    meta = {"sky_fraction": 0.1, "limiting_mag": 24.0, "tracking_efficiency": 0.9}
    fset = build_feature_set(objects, meta)
    assert fset.features.shape == (2, len(OBJECT_FEATURE_KEYS))
    assert fset.survey_meta.shape == (3,)
    assert fset.n_objects == 2


def test_build_feature_set_empty():
    fset = build_feature_set([], {})
    assert fset.features.shape == (0, len(OBJECT_FEATURE_KEYS))
    assert fset.n_objects == 0


def test_default_sigma_used_when_missing():
    fset = build_feature_set([_obj()], {})
    a_col = OBJECT_FEATURE_KEYS.index("a")
    sigma_a_col = OBJECT_FEATURE_KEYS.index("sigma_a")
    assert fset.features[0, a_col] == 100.0
    assert fset.features[0, sigma_a_col] > 0


def test_as_padded():
    objects = [_obj(), _obj(a=150.0)]
    fset = build_feature_set(objects, {})
    padded, mask = fset.as_padded(max_n=5)
    assert padded.shape == (5, len(OBJECT_FEATURE_KEYS))
    assert mask.tolist() == [False, False, True, True, True]
    assert np.allclose(padded[2:], 0.0)
