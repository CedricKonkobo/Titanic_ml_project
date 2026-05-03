"""Smoke imports — no data or trained model required."""

from src.features.build_features import FeatureEngineer


def test_feature_engineer_instantiation():
    fe = FeatureEngineer()
    assert fe.preprocessor is None
