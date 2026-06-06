from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline


OUTPERFORMANCE_THRESHOLD = 10.0
METRIC_COLUMNS = ["accuracy", "precision", "recall", "macro_f1", "roc_auc"]
NON_FEATURE_COLUMNS = {"Date", "Unix", "Ticker", "Price", "stock_p_change", "SP500", "SP500_p_change"}


@dataclass(frozen=True)
class ModelResult:
    name: str
    accuracy: float
    precision: float
    recall: float
    macro_f1: float
    roc_auc: float
    selected_stocks: int
    avg_selected_stock_return: float
    avg_selected_market_return: float
    selected_outperformance: float


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = data.copy()
    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.replace([np.inf, -np.inf], np.nan)


def load_dataset(path: str) -> pd.DataFrame:
    data = pd.read_csv(path)
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Ticker", "stock_p_change", "SP500_p_change"])
    features = feature_columns(data)
    data = clean_numeric_frame(data, features + ["stock_p_change", "SP500_p_change"])
    return data.sort_values("Date").reset_index(drop=True)


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in NON_FEATURE_COLUMNS]


def build_target(data: pd.DataFrame, threshold: float) -> pd.Series:
    return (data["stock_p_change"] - data["SP500_p_change"] >= threshold).astype(int)


def time_based_split(data: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(train_end_date)
    train = data[data["Date"] <= cutoff].copy()
    test = data[data["Date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set. Adjust train_end_date.")
    return train, test


def models(seed: int) -> dict[str, Pipeline]:
    return {
        "extra_trees": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", ExtraTreesClassifier(
                n_estimators=350,
                min_samples_leaf=4,
                class_weight="balanced",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "random_forest": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestClassifier(
                n_estimators=350,
                min_samples_leaf=4,
                class_weight="balanced_subsample",
                random_state=seed,
                n_jobs=-1,
            )),
        ]),
        "gradient_boosting": Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", GradientBoostingClassifier(random_state=seed)),
        ]),
    }


def evaluate_model(name: str, model: Pipeline, train: pd.DataFrame, test: pd.DataFrame, features: list[str], threshold: float) -> tuple[ModelResult, pd.DataFrame]:
    y_train = build_target(train, threshold)
    y_test = build_target(test, threshold)
    model.fit(train[features], y_train)
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    selected = test.loc[predictions == 1, ["Date", "Ticker", "stock_p_change", "SP500_p_change"]].copy()
    selected["model"] = name
    selected["outperformance"] = selected["stock_p_change"] - selected["SP500_p_change"]
    selected["predicted_outperformance_probability"] = probabilities[predictions == 1]

    selected_stocks = int(predictions.sum())
    avg_stock_return = float(selected["stock_p_change"].mean()) if selected_stocks else 0.0
    avg_market_return = float(selected["SP500_p_change"].mean()) if selected_stocks else 0.0
    selected_outperformance = avg_stock_return - avg_market_return
    result = ModelResult(
        name=name,
        accuracy=float(accuracy_score(y_test, predictions)),
        precision=float(precision_score(y_test, predictions, zero_division=0)),
        recall=float(recall_score(y_test, predictions, zero_division=0)),
        macro_f1=float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        roc_auc=float(roc_auc_score(y_test, probabilities)) if y_test.nunique() == 2 else float("nan"),
        selected_stocks=selected_stocks,
        avg_selected_stock_return=avg_stock_return,
        avg_selected_market_return=avg_market_return,
        selected_outperformance=float(selected_outperformance),
    )
    return result, selected


def train_best_model(data: pd.DataFrame, features: list[str], best_model_name: str, seed: int, threshold: float) -> Pipeline:
    model = models(seed)[best_model_name]
    model.fit(data[features], build_target(data, threshold))
    return model


def predict_forward(model: Pipeline, forward_data_path: str, features: list[str], top_n: int) -> pd.DataFrame:
    forward = pd.read_csv(forward_data_path)
    forward = clean_numeric_frame(forward, [column for column in features if column in forward.columns])
    missing = [column for column in features if column not in forward.columns]
    if missing:
        raise ValueError(f"Forward sample is missing feature columns: {missing[:5]}")
    probabilities = model.predict_proba(forward[features])[:, 1]
    output = forward[["Ticker"]].copy()
    output["predicted_outperformance_probability"] = probabilities
    output = output.sort_values("predicted_outperformance_probability", ascending=False).head(top_n)
    return output.reset_index(drop=True)


def write_reports(results: list[ModelResult], selections: list[pd.DataFrame], forward: pd.DataFrame, report_dir: Path, best_model: str, source_repo: str) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([result.__dict__ for result in results]).sort_values("macro_f1", ascending=False)
    metrics.to_csv(report_dir / "model_summary.csv", index=False)
    selected = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    selected.to_csv(report_dir / "backtest_selected_stocks.csv", index=False)
    forward.to_csv(report_dir / "forward_recommendations.csv", index=False)
    (report_dir / "risk_dashboard.csv").write_text(selected.to_csv(index=False), encoding="utf-8")

    lines = [
        "# Financial Forecasting and Risk Analytics Report",
        "",
        f"Source inspiration: {source_repo}",
        f"Best model by macro F1: `{best_model}`",
        "",
        "## Model Comparison",
        "",
        metrics[["name", *METRIC_COLUMNS, "selected_stocks", "selected_outperformance"]].to_markdown(index=False),
        "",
        "## Top Forward Recommendations",
        "",
        forward.to_markdown(index=False),
        "",
        "## Improvement Over Source Project",
        "",
        "- Preserved the original fundamentals-based stock outperformance idea.",
        "- Replaced random train/test splitting with date-based validation to reduce leakage.",
        "- Added model comparison across extra trees, random forest, and gradient boosting.",
        "- Added ROC AUC, F1, precision, recall, selected-stock return, and index-relative outperformance metrics.",
        "- Added saved CSV and Markdown reports for reproducible evaluation.",
    ]
    (report_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    data = load_dataset(cfg["keystats_path"])
    forward_columns = set(pd.read_csv(cfg["forward_sample_path"], nrows=1).columns)
    train, test = time_based_split(data, cfg["train_end_date"])
    features = [column for column in feature_columns(data) if column in forward_columns]
    if len(features) < 10:
        raise ValueError("Too few common feature columns between historical and forward datasets")
    results: list[ModelResult] = []
    selections: list[pd.DataFrame] = []
    for name, model in models(cfg["seed"]).items():
        result, selected = evaluate_model(
            name=name,
            model=model,
            train=train,
            test=test,
            features=features,
            threshold=cfg["outperformance_threshold"],
        )
        results.append(result)
        selections.append(selected)

    best = max(results, key=lambda item: item.macro_f1)
    best_model = train_best_model(data, features, best.name, cfg["seed"], cfg["outperformance_threshold"])
    forward = predict_forward(best_model, cfg["forward_sample_path"], features, cfg["top_n_recommendations"])
    write_reports(
        results=results,
        selections=selections,
        forward=forward,
        report_dir=Path(cfg["report_dir"]),
        best_model=best.name,
        source_repo=cfg["source_repo"],
    )
    print(json.dumps({"best_model": best.name, "macro_f1": best.macro_f1, "reports": cfg["report_dir"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run(parser.parse_args().config)
