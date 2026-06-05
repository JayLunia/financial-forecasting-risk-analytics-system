# Financial Forecasting and Risk Analytics System

## Project Overview

I built this project as an end-to-end machine learning system for stock trend prediction and risk analytics. The goal was to combine data analysis, feature engineering, predictive modeling, and dashboard-ready outputs in one practical financial analytics workflow.

The current version uses synthetic market data so the project can run locally without API keys. The pipeline is designed so real historical stock data from Yahoo Finance, Alpha Vantage, Polygon, or other providers can be added later.

## Motivation

My motivation was to build a project that shows both data science and business analytics thinking. In finance, a model prediction is only useful when it is connected to risk metrics, trend interpretation, and decision-ready outputs. I wanted to create a project that demonstrates forecasting, risk scoring, and analytics storytelling together.

## What This Project Does

- Generates synthetic stock price history for local testing.
- Engineers technical indicators such as returns, moving averages, volatility, momentum, and drawdown.
- Trains a trend classification model.
- Produces trend-up probability scores.
- Calculates a risk score using volatility and drawdown.
- Saves dashboard-ready CSV outputs.
- Produces a model summary with accuracy, macro F1, latest trend probability, and latest risk score.

## What I Did Differently

I designed the project around decision support instead of only price prediction. The output includes both model performance and business-friendly risk analytics. This makes it easier to use in dashboards, reports, or analyst workflows.

I also used a no-API synthetic data mode so the pipeline can be tested anywhere before connecting external financial data providers.

## Most Challenging Part

The most challenging part was avoiding a project that only predicts stock prices without context. Financial forecasting is noisy, so I added risk-oriented analytics such as volatility, drawdown, and trend probability to make the output more useful and realistic.

## Tech Stack

- Python
- NumPy
- Pandas
- scikit-learn
- PyYAML

## Quick Start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m src.pipeline --config configs/default.yaml
```

## Project Structure

```text
configs/        model and pipeline settings
data/sample/    optional sample data
notebooks/      exploratory analysis
reports/        generated dashboard CSV files
src/            feature engineering, modeling, and reporting
tests/          smoke tests
```

## Future Improvements

- Add live historical stock data ingestion.
- Add LSTM, Prophet, or transformer forecasting models.
- Add backtesting and walk-forward validation.
- Add portfolio-level risk analytics.
- Build a Streamlit dashboard for analyst review.

## Disclaimer

This project is for education and portfolio demonstration only. It is not investment advice.

