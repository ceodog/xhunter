"""Shared feature engineering, used identically by the simulation pipeline
(planetx.simgen.selection) and the real-data pipeline (planetx.obsdata.build_x).

This is the single module responsible for turning "a population of objects
with orbital elements + uncertainties" into the tensor-ready x consumed by
the model. Having exactly one implementation on both sides is what prevents
train/inference skew -- see README.md, "Feature parity".
"""

from planetx.featurelib.features import FeatureSet, build_feature_set

__all__ = ["FeatureSet", "build_feature_set"]
