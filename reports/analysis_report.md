# Financial Forecasting and Risk Analytics Report

Source inspiration: https://github.com/robertmartin8/MachineLearningStocks
Best model by macro F1: `random_forest`

## Model Comparison

| name              |   accuracy |   precision |    recall |   macro_f1 |   roc_auc |   selected_stocks |   selected_outperformance |
|:------------------|-----------:|------------:|----------:|-----------:|----------:|------------------:|--------------------------:|
| random_forest     |   0.691162 |    0.618834 | 0.204444  |   0.554314 |  0.643303 |               223 |                   23.7217 |
| extra_trees_tuned |   0.683714 |    0.577869 | 0.208889  |   0.550983 |  0.627495 |               244 |                   19.873  |
| extra_trees       |   0.670804 |    0.52069  | 0.223704  |   0.548249 |  0.622175 |               290 |                   19.615  |
| lightgbm          |   0.667329 |    0.509225 | 0.204444  |   0.537182 |  0.609396 |               271 |                   16.8203 |
| xgboost           |   0.671301 |    0.544828 | 0.117037  |   0.493162 |  0.567006 |               145 |                   21.163  |
| gradient_boosting |   0.675273 |    0.601942 | 0.0918519 |   0.479076 |  0.577153 |               103 |                   28.2177 |

## Regression Framing

| name                   |   selected_stocks |   avg_selected_stock_return |   avg_selected_market_return |   selected_outperformance |   avg_selected_sharpe_proxy |
|:-----------------------|------------------:|----------------------------:|-----------------------------:|--------------------------:|----------------------------:|
| extra_trees_regression |                25 |                     75.8732 |                      23.6772 |                    52.196 |                     1.25079 |

## Threshold Experiment

| name                     |   macro_f1 |   selected_stocks |   selected_outperformance |
|:-------------------------|-----------:|------------------:|--------------------------:|
| extra_trees_threshold_5  |   0.526748 |               433 |                   14.1988 |
| extra_trees_threshold_10 |   0.548688 |               285 |                   19.7547 |
| extra_trees_threshold_15 |   0.566262 |               204 |                   24.7523 |

## Portfolio-Level Risk Analytics

| model             |   positions |   unique_tickers |   avg_stock_return |   avg_market_return |   avg_outperformance |   return_volatility |   hit_rate |   sharpe_proxy |   worst_position_return |   best_position_return |
|:------------------|------------:|-----------------:|-------------------:|--------------------:|---------------------:|--------------------:|-----------:|---------------:|------------------------:|-----------------------:|
| gradient_boosting |         103 |               48 |            48.8917 |             20.674  |              28.2177 |             60.472  |   0.718447 |       0.8085   |                  -22.05 |                 447.48 |
| random_forest     |         223 |               78 |            43.6238 |             19.9021 |              23.7217 |             47.7188 |   0.762332 |       0.914185 |                  -39.71 |                 447.48 |
| xgboost           |         145 |               51 |            41.78   |             20.617  |              21.163  |             52.7342 |   0.675862 |       0.792276 |                  -54.8  |                 375.13 |
| extra_trees_tuned |         244 |               89 |            39.7973 |             19.9242 |              19.873  |             45.2781 |   0.745902 |       0.878951 |                  -52.13 |                 447.48 |
| extra_trees       |         290 |              100 |            39.9066 |             20.2916 |              19.615  |             51.1224 |   0.67931  |       0.780608 |                  -73.79 |                 447.48 |
| lightgbm          |         271 |              105 |            36.8975 |             20.0771 |              16.8203 |             36.1528 |   0.723247 |       1.0206   |                  -52.13 |                 236.27 |

## Constrained Portfolio Construction

| model             |   constrained_positions |   invested_weight |   total_transaction_cost_pct |   net_portfolio_return_pct |   benchmark_portfolio_return_pct |   net_outperformance_pct |   max_single_position_weight |   max_sector_weight |
|:------------------|------------------------:|------------------:|-----------------------------:|---------------------------:|---------------------------------:|-------------------------:|-----------------------------:|--------------------:|
| gradient_boosting |                      62 |          0.648467 |                     0.162117 |                    29.0978 |                          13.1867 |                  15.9111 |                   0.0136114  |                0.25 |
| xgboost           |                      88 |          0.653752 |                     0.163438 |                    27.6027 |                          13.5652 |                  14.0376 |                   0.00980785 |                0.25 |
| random_forest     |                     121 |          0.585695 |                     0.146424 |                    25.3466 |                          11.592  |                  13.7546 |                   0.00684096 |                0.25 |
| lightgbm          |                     133 |          0.552596 |                     0.138149 |                    23.6778 |                          11.4271 |                  12.2507 |                   0.00538719 |                0.25 |
| extra_trees_tuned |                     140 |          0.624758 |                     0.156189 |                    24.1486 |                          12.4506 |                  11.6981 |                   0.00626334 |                0.25 |
| extra_trees       |                     165 |          0.611201 |                     0.1528   |                    23.8893 |                          12.3033 |                  11.586  |                   0.00545447 |                0.25 |

## Top SHAP Drivers

| feature                  |   mean_abs_shap |
|:-------------------------|----------------:|
| Graham Number / Price    |      0.0265956  |
| Total Debt/Equity        |      0.0148178  |
| Enterprise Value         |      0.0147335  |
| Market Cap               |      0.0123458  |
| Net Income Avl to Common |      0.0120621  |
| % Held by Institutions   |      0.0120329  |
| Enterprise Value/Revenue |      0.0118872  |
| Gross Profit             |      0.0117677  |
| Price/Sales              |      0.0115686  |
| 50-Day Moving Average    |      0.0111046  |
| Profit Margin            |      0.010699   |
| PEG Ratio                |      0.0105417  |
| Moving Average Spread    |      0.01009    |
| Total Debt               |      0.00993769 |
| Operating Margin         |      0.00916163 |

## Consensus Picks

| Ticker   |   votes |   avg_probability |
|:---------|--------:|------------------:|
| ma       |      54 |          0.777261 |
| aapl     |      53 |          0.817615 |
| dfs      |      51 |          0.620792 |
| bac      |      45 |          0.643906 |
| wmt      |      42 |          0.716613 |
| pfe      |      40 |          0.741159 |
| fslr     |      40 |          0.562863 |
| jpm      |      34 |          0.596709 |
| hov      |      30 |          0.641765 |
| ges      |      29 |          0.565926 |
| s        |      26 |          0.667938 |
| znga     |      26 |          0.644801 |
| amd      |      26 |          0.602539 |
| gnw      |      25 |          0.541832 |
| aeo      |      24 |          0.58453  |
| rsh      |      24 |          0.569394 |
| goog     |      23 |          0.558758 |
| v        |      22 |          0.711329 |
| siri     |      21 |          0.601676 |
| ge       |      21 |          0.579127 |
| gme      |      19 |          0.609968 |
| wfc      |      18 |          0.658327 |
| sfly     |      16 |          0.668615 |
| mpc      |      16 |          0.665245 |
| grpn     |      15 |          0.685404 |

## Top Forward Recommendations

| Ticker   | Sector   |   predicted_outperformance_probability |
|:---------|:---------|---------------------------------------:|
| JCP      | Unknown  |                               0.765783 |
| HOV      | Unknown  |                               0.661563 |
| ALTR     | Unknown  |                               0.647741 |
| AVP      | Unknown  |                               0.64379  |
| GTN      | Unknown  |                               0.637618 |
| SHLD     | Unknown  |                               0.631936 |
| FSLR     | Unknown  |                               0.631103 |
| GRPN     | Unknown  |                               0.630531 |
| SD       | Unknown  |                               0.627364 |
| ZNGA     | Unknown  |                               0.615933 |
| EXPR     | Unknown  |                               0.6155   |
| SCHL     | Unknown  |                               0.604794 |
| NBR      | Unknown  |                               0.601901 |
| GES      | Unknown  |                               0.591832 |
| GME      | Unknown  |                               0.591151 |
| CAMP     | Unknown  |                               0.58236  |
| RDC      | Unknown  |                               0.580286 |
| FTR      | Unknown  |                               0.576748 |
| QDEL     | Unknown  |                               0.575459 |
| NE       | Unknown  |                               0.572549 |
| PBI      | Unknown  |                               0.565903 |
| WIN      | Unknown  |                               0.564283 |
| ANF      | Unknown  |                               0.564226 |
| LIFE     | Unknown  |                               0.561957 |
| GT       | Unknown  |                               0.560953 |

## Improvement Over Source Project

- Preserved the original fundamentals-based stock outperformance idea.
- Replaced random train/test splitting with date-based validation to reduce leakage.
- Added model comparison across extra trees, random forest, gradient boosting, XGBoost, and optional LightGBM.
- Added ROC AUC, F1, precision, recall, selected-stock return, and index-relative outperformance metrics.
- Added missing-value-aware preprocessing, engineered valuation features, SEC filing-derived feature joins, neural net option, XGBoost, optional LightGBM, tuned ExtraTrees, regression framing, threshold experiments, consensus recommendations, feature importance, SHAP explanations, transaction costs, position sizing, sector concentration limits, and portfolio-level risk analytics.
- Added saved CSV and Markdown reports for reproducible evaluation.
