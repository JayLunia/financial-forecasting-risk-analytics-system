from src.pipeline import engineer_features, synthetic_prices


def test_feature_engineering_outputs_target():
    prices = synthetic_prices(seed=1, n_days=80, symbol="TEST")
    features = engineer_features(prices)
    assert "target_up" in features.columns
    assert len(features) > 0

