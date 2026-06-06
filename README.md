# Financial Forecasting and Risk Analytics System

## Project Overview

I rebuilt this project as a real, end-to-end stock outperformance and risk analytics system using the MIT-licensed [MachineLearningStocks](https://github.com/robertmartin8/MachineLearningStocks) project as the source inspiration and dataset base.

The system uses real historical stock fundamentals and S&P 500-relative performance data, trains multiple machine learning models, evaluates them with a date-based backtest, and generates reproducible reports with model metrics, selected-stock performance, and forward recommendations.

## Motivation

My motivation was to build a project that shows both data science and business analytics thinking. In finance, a model prediction is only useful when it is connected to risk metrics, trend interpretation, and decision-ready outputs. I wanted to create a project that demonstrates forecasting, risk scoring, and analytics storytelling together.

## What This Project Does

- Uses real historical fundamental and market performance data from MachineLearningStocks.
- Predicts whether a stock will outperform the S&P 500 by more than a configurable threshold.
- Uses date-based train/test validation instead of random splitting.
- Compares extra trees, random forest, gradient boosting, and XGBoost models.
- Adds tuned ExtraTrees using time-series cross-validation.
- Adds engineered valuation features such as Graham Number / Price, Free Cash Flow Yield, Debt / Cash, and Moving Average Spread.
- Adds missing-value-aware preprocessing instead of dropping every incomplete row.
- Tests multiple outperformance thresholds.
- Adds a regression framing that predicts relative stock outperformance directly.
- Adds consensus stock recommendations across model outputs.
- Adds feature importance reporting to show what the model is using.
- Adds SHAP explanations for the best tree model so the drivers are easier to review.
- Adds portfolio-level analytics across model-selected stock baskets, including volatility, hit rate, worst position return, and a simple Sharpe-style proxy.
- Reports accuracy, precision, recall, macro F1, ROC AUC, selected-stock return, market return, and index-relative outperformance.
- Generates forward stock recommendations from the latest available fundamentals sample.
- Saves dashboard-ready CSVs and a Markdown analysis report.
- Includes a Streamlit dashboard for analyst review of models, risk, explanations, and recommendations.

## What I Did Differently

The original project used a random train/test split for backtesting, which can make results look better than they are. My version improves the project by using a chronological split, comparing multiple models including XGBoost, adding engineered fundamentals, tuning ExtraTrees, running threshold experiments, testing regression framing, and saving full evaluation artifacts.

I also turned the workflow into a reusable pipeline with config-driven paths, tests, model comparison, forward recommendations, SHAP explanations, portfolio risk analytics, an analyst dashboard, and clear attribution.

## Most Challenging Part

The most challenging part was improving the evaluation honestly. A random split mixes old and future observations, which is not how financial models are used. I changed the validation to train on earlier data and test on later data, then added metrics that show both classification quality and investment-style outperformance.

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- XGBoost
- SHAP
- Streamlit
- Plotly
- PyYAML
- tabulate

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --config configs/default.yaml
```

Run tests:

```bash
python -m pytest -q
```

Open the analyst dashboard after generating the reports:

```bash
streamlit run app/streamlit_app.py
```

## Project Structure

```text
configs/        model and pipeline settings
data/sample/    optional sample data
notebooks/      exploratory analysis
app/            Streamlit analyst dashboard
reports/        generated CSV and Markdown evaluation reports
src/            feature engineering, modeling, and reporting
tests/          smoke tests
```

## Outputs

```text
reports/model_summary.csv             model comparison metrics
reports/backtest_selected_stocks.csv  selected stocks from the date-based backtest
reports/forward_recommendations.csv   top current stock recommendations
reports/regression_selected_stocks.csv regression-based selected stocks
reports/threshold_experiment.csv      threshold sensitivity results
reports/consensus_recommendations.csv stocks repeatedly selected by models
reports/feature_importance.csv        ranked model feature importances
reports/shap_summary.csv              SHAP-ranked model drivers
reports/portfolio_analytics.csv       portfolio-level risk and return summary
reports/analysis_report.md            readable report with metrics and improvements
```

## Latest Results

The default run compares ExtraTrees, tuned ExtraTrees, Random Forest, Gradient Boosting, and XGBoost on a chronological holdout set.

- Best classifier by macro F1: `extra_trees`
- Best classifier macro F1: `0.560`
- Best classifier selected-stock outperformance: `21.53` percentage points
- XGBoost selected-stock outperformance: `23.56` percentage points
- Best portfolio outperformance by model: `random_forest` at `27.21` percentage points with a `0.768` hit rate
- Top SHAP drivers: Total Debt/Equity missingness, Moving Average Spread, Beta, quarterly growth missingness, and Market Cap
- Strongest threshold experiment: `15%` outperformance threshold
- Threshold experiment macro F1: `0.567`
- Threshold experiment selected-stock outperformance: `26.81` percentage points

These results are generated by:

```bash
python -m src.pipeline --config configs/default.yaml
```

## Source and Attribution

This project is an improved derivative inspired by:

- Repository: https://github.com/robertmartin8/MachineLearningStocks
- License: MIT
- Original author: Robert Andrew Martin

The original MIT license is preserved in `LICENSE-MachineLearningStocks.txt`.

## Future Improvements

- Add walk-forward retraining across multiple market regimes.
- Add transaction costs, position sizing, and sector concentration limits.
- Add SEC filing-derived features from annual and quarterly reports.
- Add LightGBM as an optional comparator for faster large-scale experiments.

## Disclaimer

This project is for education and portfolio demonstration only. It is not investment advice.
