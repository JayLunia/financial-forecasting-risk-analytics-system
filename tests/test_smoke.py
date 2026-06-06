from pathlib import Path

import pandas as pd

from src.pipeline import build_target, feature_columns, load_dataset, run, time_based_split


def test_dataset_has_real_fundamental_rows():
    data = load_dataset("data/sample/keystats.csv")
    assert len(data) > 8000
    assert {"Ticker", "stock_p_change", "SP500_p_change"}.issubset(data.columns)


def test_time_based_split_and_target_are_valid():
    data = load_dataset("data/sample/keystats.csv")
    train, test = time_based_split(data, "2011-12-31")
    assert train["Date"].max() <= pd.Timestamp("2011-12-31")
    assert test["Date"].min() > pd.Timestamp("2011-12-31")
    target = build_target(train, threshold=10.0)
    assert set(target.unique()).issubset({0, 1})


def test_feature_columns_exclude_labels():
    data = load_dataset("data/sample/keystats.csv")
    features = feature_columns(data)
    assert "stock_p_change" not in features
    assert "SP500_p_change" not in features
    assert "Trailing P/E" in features


def test_pipeline_writes_real_reports():
    run("configs/default.yaml")
    for file_name in [
        "model_summary.csv",
        "backtest_selected_stocks.csv",
        "forward_recommendations.csv",
        "analysis_report.md",
    ]:
        assert Path("reports", file_name).exists()
