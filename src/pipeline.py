from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def synthetic_prices(seed: int, n_days: int, symbol: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    returns = rng.normal(0.0006, 0.018, n_days)
    shocks = rng.choice([0, -0.06, 0.05], n_days, p=[0.96, 0.02, 0.02])
    close = 100 * np.exp(np.cumsum(returns + shocks))
    return pd.DataFrame({
        "date": pd.date_range("2021-01-01", periods=n_days, freq="B"),
        "symbol": symbol,
        "close": close,
    })


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["return_1d"] = out["close"].pct_change()
    out["ma_10"] = out["close"].rolling(10).mean()
    out["ma_30"] = out["close"].rolling(30).mean()
    out["volatility_20"] = out["return_1d"].rolling(20).std()
    out["momentum_10"] = out["close"].pct_change(10)
    out["drawdown"] = out["close"] / out["close"].cummax() - 1
    out["target_up"] = (out["close"].shift(-5) > out["close"]).astype(int)
    return out.dropna().reset_index(drop=True)


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    prices = synthetic_prices(cfg["seed"], cfg["n_days"], cfg["symbol"])
    features = engineer_features(prices)
    columns = ["return_1d", "ma_10", "ma_30", "volatility_20", "momentum_10", "drawdown"]
    x_train, x_test, y_train, y_test = train_test_split(
        features[columns], features["target_up"], test_size=cfg["test_size"],
        random_state=cfg["seed"], shuffle=False
    )
    model = GradientBoostingClassifier(random_state=cfg["seed"])
    model.fit(x_train, y_train)
    probability = model.predict_proba(features[columns])[:, 1]
    features["trend_up_probability"] = probability
    features["risk_score"] = (
        features["volatility_20"].rank(pct=True) * 0.55
        + (-features["drawdown"]).rank(pct=True) * 0.45
    )
    preds = model.predict(x_test)
    summary = pd.DataFrame([{
        "symbol": cfg["symbol"],
        "accuracy": accuracy_score(y_test, preds),
        "macro_f1": f1_score(y_test, preds, average="macro"),
        "latest_trend_up_probability": float(features["trend_up_probability"].iloc[-1]),
        "latest_risk_score": float(features["risk_score"].iloc[-1]),
    }])

    output = Path(cfg["report_path"])
    output.parent.mkdir(exist_ok=True)
    features.to_csv(output, index=False)
    summary.to_csv(output.parent / "model_summary.csv", index=False)
    print(f"Wrote dashboard data to {output.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run(parser.parse_args().config)

