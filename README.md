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
- Compares extra trees, random forest, and gradient boosting models.
- Reports accuracy, precision, recall, macro F1, ROC AUC, selected-stock return, market return, and index-relative outperformance.
- Generates forward stock recommendations from the latest available fundamentals sample.
- Saves dashboard-ready CSVs and a Markdown analysis report.

## What I Did Differently

I started from a strong public GitHub project instead of building a toy sample from scratch. The original project used a random train/test split for backtesting, which can make results look better than they are. My version improves the project by using a chronological split, comparing multiple models, and saving full evaluation artifacts.

I also turned the workflow into a reusable pipeline with config-driven paths, tests, model comparison, forward recommendations, and clear attribution.

## Most Challenging Part

The most challenging part was improving the evaluation honestly. A random split mixes old and future observations, which is not how financial models are used. I changed the validation to train on earlier data and test on later data, then added metrics that show both classification quality and investment-style outperformance.

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
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

## Project Structure

```text
configs/        model and pipeline settings
data/sample/    optional sample data
notebooks/      exploratory analysis
reports/        generated CSV and Markdown evaluation reports
src/            feature engineering, modeling, and reporting
tests/          smoke tests
```

## Outputs

```text
reports/model_summary.csv             model comparison metrics
reports/backtest_selected_stocks.csv  selected stocks from the date-based backtest
reports/forward_recommendations.csv   top current stock recommendations
reports/analysis_report.md            readable report with metrics and improvements
```

## Source and Attribution

This project is an improved derivative inspired by:

- Repository: https://github.com/robertmartin8/MachineLearningStocks
- License: MIT
- Original author: Robert Andrew Martin

The original MIT license is preserved in `LICENSE-MachineLearningStocks.txt`.

## Future Improvements

- Add walk-forward cross-validation across multiple cutoff dates.
- Add XGBoost or LightGBM model comparison.
- Add feature importance and SHAP explanations.
- Add portfolio-level risk analytics.
- Build a Streamlit dashboard for analyst review.

## Disclaimer

This project is for education and portfolio demonstration only. It is not investment advice.
