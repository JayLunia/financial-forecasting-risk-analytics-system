from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
import shap
from xgboost import XGBClassifier
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import FunctionTransformer, RobustScaler


OUTPERFORMANCE_THRESHOLD = 10.0
METRIC_COLUMNS = ["accuracy", "precision", "recall", "macro_f1", "roc_auc"]
NON_FEATURE_COLUMNS = {"Date", "Unix", "Ticker", "Price", "stock_p_change", "SP500", "SP500_p_change"}
ALIASES = {
    "Quarterly Revenue Growth": "Qtrly Revenue Growth",
    "Quarterly Earnings Growth": "Qtrly Earnings Growth",
    "Net Income Avi to Common": "Net Income Avl to Common",
    "Shares Short (as of": "Shares Short",
    "Shares Short (prior month": "Shares Short (prior month)",
}


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
    avg_selected_sharpe_proxy: float


@dataclass(frozen=True)
class RegressionResult:
    name: str
    selected_stocks: int
    avg_selected_stock_return: float
    avg_selected_market_return: float
    selected_outperformance: float
    avg_selected_sharpe_proxy: float


def load_config(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def clean_numeric_frame(data: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    cleaned = data.copy()
    for column in columns:
        cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce")
    return cleaned.replace([np.inf, -np.inf], np.nan)


def standardize_columns(data: pd.DataFrame) -> pd.DataFrame:
    return data.rename(columns={source: target for source, target in ALIASES.items() if source in data.columns})


def load_dataset(path: str) -> pd.DataFrame:
    data = standardize_columns(pd.read_csv(path))
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data = data.dropna(subset=["Date", "Ticker", "stock_p_change", "SP500_p_change"])
    features = feature_columns(data)
    data = clean_numeric_frame(data, features + ["stock_p_change", "SP500_p_change"])
    return data.sort_values("Date").reset_index(drop=True)


def feature_columns(data: pd.DataFrame) -> list[str]:
    return [column for column in data.columns if column not in NON_FEATURE_COLUMNS]


def add_engineered_features(data: pd.DataFrame) -> pd.DataFrame:
    out = data.copy()
    eps = out.get("Diluted EPS")
    book = out.get("Book Value Per Share")
    price = out.get("Price")
    if eps is not None and book is not None and price is not None:
        positive = (eps > 0) & (book > 0) & (price > 0)
        out["Graham Number"] = np.nan
        out.loc[positive, "Graham Number"] = np.sqrt(22.5 * eps.loc[positive] * book.loc[positive])
        out["Graham Number / Price"] = out["Graham Number"] / price.replace(0, np.nan)
    if {"Enterprise Value", "Market Cap"}.issubset(out.columns):
        out["Enterprise Value / Market Cap"] = out["Enterprise Value"] / out["Market Cap"].replace(0, np.nan)
    if {"Total Debt", "Total Cash"}.issubset(out.columns):
        out["Debt / Cash"] = out["Total Debt"] / out["Total Cash"].replace(0, np.nan)
    if {"Operating Cash Flow", "Market Cap"}.issubset(out.columns):
        out["Operating Cash Flow / Market Cap"] = out["Operating Cash Flow"] / out["Market Cap"].replace(0, np.nan)
    if {"Levered Free Cash Flow", "Market Cap"}.issubset(out.columns):
        out["Free Cash Flow Yield"] = out["Levered Free Cash Flow"] / out["Market Cap"].replace(0, np.nan)
    if {"50-Day Moving Average", "200-Day Moving Average"}.issubset(out.columns):
        out["Moving Average Spread"] = (
            out["50-Day Moving Average"] / out["200-Day Moving Average"].replace(0, np.nan) - 1
        )
    return out.replace([np.inf, -np.inf], np.nan)


def add_missing_indicators(data: pd.DataFrame, columns: list[str], min_missing_rate: float) -> pd.DataFrame:
    out = data.copy()
    for column in columns:
        if column in out.columns and out[column].isna().mean() >= min_missing_rate:
            out[f"{column} Missing"] = out[column].isna().astype(int)
    return out


def build_target(data: pd.DataFrame, threshold: float) -> pd.Series:
    return (data["stock_p_change"] - data["SP500_p_change"] >= threshold).astype(int)


def time_based_split(data: pd.DataFrame, train_end_date: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    cutoff = pd.Timestamp(train_end_date)
    train = data[data["Date"] <= cutoff].copy()
    test = data[data["Date"] > cutoff].copy()
    if train.empty or test.empty:
        raise ValueError("Time split produced an empty train or test set. Adjust train_end_date.")
    return train, test


def models(seed: int, include_neural_network: bool = False, include_xgboost: bool = True) -> dict[str, Pipeline]:
    clipper = FunctionTransformer(lambda values: np.clip(values, -1e9, 1e9), validate=False)
    model_map = {
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
    if include_neural_network:
        model_map["mlp"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("clip", clipper),
            ("scaler", RobustScaler()),
            ("model", MLPClassifier(
                hidden_layer_sizes=(64, 24),
                alpha=0.001,
                learning_rate_init=0.001,
                max_iter=450,
                early_stopping=True,
                random_state=seed,
            )),
        ])
    if include_xgboost:
        model_map["xgboost"] = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBClassifier(
                n_estimators=300,
                max_depth=3,
                learning_rate=0.05,
                subsample=0.85,
                colsample_bytree=0.85,
                eval_metric="logloss",
                tree_method="hist",
                random_state=seed,
                n_jobs=-1,
            )),
        ])
    return model_map


def tune_extra_trees(train: pd.DataFrame, features: list[str], seed: int, threshold: float) -> Pipeline:
    base = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesClassifier(class_weight="balanced", random_state=seed, n_jobs=-1)),
    ])
    search = GridSearchCV(
        base,
        param_grid={
            "model__n_estimators": [250, 450],
            "model__min_samples_leaf": [2, 4, 8],
            "model__max_features": ["sqrt", 0.65],
        },
        scoring="f1_macro",
        cv=TimeSeriesSplit(n_splits=3),
        n_jobs=-1,
    )
    search.fit(train[features], build_target(train, threshold))
    return search.best_estimator_


def evaluate_model(name: str, model: Pipeline, train: pd.DataFrame, test: pd.DataFrame, features: list[str], threshold: float, probability_cutoff: float) -> tuple[ModelResult, pd.DataFrame]:
    y_train = build_target(train, threshold)
    y_test = build_target(test, threshold)
    model.fit(train[features], y_train)
    probabilities = model.predict_proba(test[features])[:, 1]
    predictions = (probabilities >= probability_cutoff).astype(int)

    selected = test.loc[predictions == 1, ["Date", "Ticker", "stock_p_change", "SP500_p_change"]].copy()
    selected["model"] = name
    selected["outperformance"] = selected["stock_p_change"] - selected["SP500_p_change"]
    selected["sharpe_proxy"] = selected["stock_p_change"] / selected["stock_p_change"].abs().rolling(20, min_periods=1).std().replace(0, np.nan)
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
        avg_selected_sharpe_proxy=float(selected["sharpe_proxy"].replace([np.inf, -np.inf], np.nan).mean()) if selected_stocks else 0.0,
    )
    return result, selected


def evaluate_regression(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, top_n: int) -> tuple[RegressionResult, pd.DataFrame]:
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("model", ExtraTreesRegressor(n_estimators=350, min_samples_leaf=4, random_state=seed, n_jobs=-1)),
    ])
    model.fit(train[features], train["stock_p_change"] - train["SP500_p_change"])
    scored = test[["Date", "Ticker", "stock_p_change", "SP500_p_change"]].copy()
    scored["model"] = "extra_trees_regression"
    scored["predicted_outperformance"] = model.predict(test[features])
    selected = scored.sort_values("predicted_outperformance", ascending=False).head(top_n).copy()
    selected["outperformance"] = selected["stock_p_change"] - selected["SP500_p_change"]
    selected["sharpe_proxy"] = selected["stock_p_change"] / selected["stock_p_change"].abs().std()
    result = RegressionResult(
        name="extra_trees_regression",
        selected_stocks=len(selected),
        avg_selected_stock_return=float(selected["stock_p_change"].mean()),
        avg_selected_market_return=float(selected["SP500_p_change"].mean()),
        selected_outperformance=float(selected["outperformance"].mean()),
        avg_selected_sharpe_proxy=float(selected["sharpe_proxy"].replace([np.inf, -np.inf], np.nan).mean()),
    )
    return result, selected


def train_best_model(data: pd.DataFrame, features: list[str], best_model_name: str, seed: int, threshold: float) -> Pipeline:
    model = models(seed)[best_model_name]
    model.fit(data[features], build_target(data, threshold))
    return model


def predict_forward(model: Pipeline, forward_data_path: str, features: list[str], top_n: int) -> pd.DataFrame:
    forward = add_engineered_features(standardize_columns(pd.read_csv(forward_data_path)))
    forward = clean_numeric_frame(forward, [column for column in features if column in forward.columns])
    for column in features:
        if column.endswith(" Missing") and column not in forward.columns:
            source_column = column.removesuffix(" Missing")
            forward[column] = forward[source_column].isna().astype(int) if source_column in forward.columns else 1
    missing = [column for column in features if column not in forward.columns]
    if missing:
        raise ValueError(f"Forward sample is missing feature columns: {missing[:5]}")
    probabilities = model.predict_proba(forward[features])[:, 1]
    output = forward[["Ticker"]].copy()
    output["predicted_outperformance_probability"] = probabilities
    output = output.sort_values("predicted_outperformance_probability", ascending=False).head(top_n)
    return output.reset_index(drop=True)


def feature_importance(model: Pipeline, features: list[str]) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])
    return pd.DataFrame({"feature": features, "importance": estimator.feature_importances_}).sort_values(
        "importance", ascending=False
    )


def shap_summary(model: Pipeline, sample: pd.DataFrame, features: list[str], max_rows: int) -> pd.DataFrame:
    estimator = model.named_steps["model"]
    if not hasattr(estimator, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "mean_abs_shap"])
    transformed = model.named_steps["imputer"].transform(sample[features].head(max_rows))
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    if getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, -1]
    return (
        pd.DataFrame({"feature": features, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )


def threshold_sweep(train: pd.DataFrame, test: pd.DataFrame, features: list[str], seed: int, thresholds: list[float], probability_cutoff: float) -> pd.DataFrame:
    rows = []
    for threshold in thresholds:
        model = ExtraTreesClassifier(n_estimators=300, min_samples_leaf=4, class_weight="balanced", random_state=seed, n_jobs=-1)
        result, _ = evaluate_model(
            name=f"extra_trees_threshold_{threshold:g}",
            model=Pipeline([("imputer", SimpleImputer(strategy="median")), ("model", model)]),
            train=train,
            test=test,
            features=features,
            threshold=threshold,
            probability_cutoff=probability_cutoff,
        )
        rows.append(result.__dict__)
    return pd.DataFrame(rows)


def usable_features(train: pd.DataFrame, features: list[str], min_non_missing_rate: float) -> list[str]:
    first_fold_end = max(1, len(train) // 4)
    early_train = train.iloc[:first_fold_end]
    usable = []
    for feature in features:
        if feature not in train.columns:
            continue
        series = train[feature]
        if (
            series.notna().mean() >= min_non_missing_rate
            and early_train[feature].notna().any()
            and series.nunique(dropna=True) > 1
        ):
            usable.append(feature)
    return usable


def consensus_recommendations(selections: list[pd.DataFrame], top_n: int) -> pd.DataFrame:
    if not selections:
        return pd.DataFrame(columns=["Ticker", "votes", "avg_probability"])
    frames = [selection[["Ticker", "predicted_outperformance_probability"]].copy() for selection in selections if "predicted_outperformance_probability" in selection]
    if not frames:
        return pd.DataFrame(columns=["Ticker", "votes", "avg_probability"])
    selected = pd.concat(frames, ignore_index=True)
    return (
        selected.groupby("Ticker", as_index=False)
        .agg(votes=("Ticker", "size"), avg_probability=("predicted_outperformance_probability", "mean"))
        .sort_values(["votes", "avg_probability"], ascending=False)
        .head(top_n)
    )


def portfolio_analytics(selections: list[pd.DataFrame]) -> pd.DataFrame:
    if not selections:
        return pd.DataFrame()
    selected = pd.concat(selections, ignore_index=True)
    rows = []
    for model_name, group in selected.groupby("model"):
        returns = group["stock_p_change"].dropna()
        market = group["SP500_p_change"].dropna()
        outperformance = group["outperformance"].dropna()
        rows.append({
            "model": model_name,
            "positions": int(len(group)),
            "unique_tickers": int(group["Ticker"].nunique()),
            "avg_stock_return": float(returns.mean()) if len(returns) else 0.0,
            "avg_market_return": float(market.mean()) if len(market) else 0.0,
            "avg_outperformance": float(outperformance.mean()) if len(outperformance) else 0.0,
            "return_volatility": float(returns.std()) if len(returns) > 1 else 0.0,
            "hit_rate": float((outperformance > 0).mean()) if len(outperformance) else 0.0,
            "sharpe_proxy": float(returns.mean() / returns.std()) if len(returns) > 1 and returns.std() else 0.0,
            "worst_position_return": float(returns.min()) if len(returns) else 0.0,
            "best_position_return": float(returns.max()) if len(returns) else 0.0,
        })
    return pd.DataFrame(rows).sort_values("avg_outperformance", ascending=False)


def write_reports(
    results: list[ModelResult],
    selections: list[pd.DataFrame],
    forward: pd.DataFrame,
    report_dir: Path,
    best_model: str,
    source_repo: str,
    regression_result: RegressionResult,
    regression_selection: pd.DataFrame,
    feature_importances: pd.DataFrame,
    shap_importances: pd.DataFrame,
    threshold_results: pd.DataFrame,
    consensus: pd.DataFrame,
    portfolio: pd.DataFrame,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    metrics = pd.DataFrame([result.__dict__ for result in results]).sort_values("macro_f1", ascending=False)
    metrics.to_csv(report_dir / "model_summary.csv", index=False)
    selected = pd.concat(selections, ignore_index=True) if selections else pd.DataFrame()
    selected.to_csv(report_dir / "backtest_selected_stocks.csv", index=False)
    forward.to_csv(report_dir / "forward_recommendations.csv", index=False)
    regression_selection.to_csv(report_dir / "regression_selected_stocks.csv", index=False)
    feature_importances.to_csv(report_dir / "feature_importance.csv", index=False)
    shap_importances.to_csv(report_dir / "shap_summary.csv", index=False)
    threshold_results.to_csv(report_dir / "threshold_experiment.csv", index=False)
    consensus.to_csv(report_dir / "consensus_recommendations.csv", index=False)
    portfolio.to_csv(report_dir / "portfolio_analytics.csv", index=False)
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
        "## Regression Framing",
        "",
        pd.DataFrame([regression_result.__dict__]).to_markdown(index=False),
        "",
        "## Threshold Experiment",
        "",
        threshold_results[["name", "macro_f1", "selected_stocks", "selected_outperformance"]].to_markdown(index=False),
        "",
        "## Portfolio-Level Risk Analytics",
        "",
        portfolio.to_markdown(index=False),
        "",
        "## Top SHAP Drivers",
        "",
        shap_importances.head(15).to_markdown(index=False),
        "",
        "## Consensus Picks",
        "",
        consensus.to_markdown(index=False),
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
        "- Added missing-value-aware preprocessing, engineered valuation features, neural net option, XGBoost, tuned ExtraTrees, regression framing, threshold experiments, consensus recommendations, feature importance, SHAP explanations, and portfolio-level risk analytics.",
        "- Added saved CSV and Markdown reports for reproducible evaluation.",
    ]
    (report_dir / "analysis_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(config_path: str) -> None:
    cfg = load_config(config_path)
    data = add_engineered_features(load_dataset(cfg["keystats_path"]))
    forward_preview = add_engineered_features(standardize_columns(pd.read_csv(cfg["forward_sample_path"], nrows=1)))
    forward_columns = set(forward_preview.columns)
    train, test = time_based_split(data, cfg["train_end_date"])
    base_features = [column for column in feature_columns(data) if column in forward_columns]
    data = add_missing_indicators(data, base_features, cfg["missing_indicator_threshold"])
    train, test = time_based_split(data, cfg["train_end_date"])
    features = usable_features(
        train,
        [column for column in feature_columns(data) if column in forward_columns or column.endswith(" Missing")],
        cfg["min_feature_non_missing_rate"],
    )
    if len(features) < 10:
        raise ValueError("Too few common feature columns between historical and forward datasets")
    results: list[ModelResult] = []
    selections: list[pd.DataFrame] = []
    candidate_models = models(
        cfg["seed"],
        include_neural_network=cfg.get("enable_neural_network", False),
        include_xgboost=cfg.get("enable_xgboost", True),
    )
    if cfg.get("enable_hyperparameter_tuning", True):
        candidate_models["extra_trees_tuned"] = tune_extra_trees(train, features, cfg["seed"], cfg["outperformance_threshold"])
    for name, model in candidate_models.items():
        result, selected = evaluate_model(
            name=name,
            model=model,
            train=train,
            test=test,
            features=features,
            threshold=cfg["outperformance_threshold"],
            probability_cutoff=cfg["probability_cutoff"],
        )
        results.append(result)
        selections.append(selected)

    best = max(results, key=lambda item: item.macro_f1)
    regression_result, regression_selection = evaluate_regression(train, test, features, cfg["seed"], cfg["top_n_recommendations"])
    threshold_results = threshold_sweep(
        train,
        test,
        features,
        cfg["seed"],
        cfg["threshold_experiments"],
        cfg["probability_cutoff"],
    )
    best_model = candidate_models[best.name]
    best_model.fit(data[features], build_target(data, cfg["outperformance_threshold"]))
    forward = predict_forward(best_model, cfg["forward_sample_path"], features, cfg["top_n_recommendations"])
    importances = feature_importance(best_model, features)
    shap_importances = shap_summary(best_model, test, features, cfg["shap_sample_size"])
    consensus = consensus_recommendations(selections, cfg["top_n_recommendations"])
    portfolio = portfolio_analytics(selections)
    write_reports(
        results=results,
        selections=selections,
        forward=forward,
        report_dir=Path(cfg["report_dir"]),
        best_model=best.name,
        source_repo=cfg["source_repo"],
        regression_result=regression_result,
        regression_selection=regression_selection,
        feature_importances=importances,
        shap_importances=shap_importances,
        threshold_results=threshold_results,
        consensus=consensus,
        portfolio=portfolio,
    )
    print(json.dumps({"best_model": best.name, "macro_f1": best.macro_f1, "reports": cfg["report_dir"]}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    run(parser.parse_args().config)
