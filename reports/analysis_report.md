# Financial Forecasting and Risk Analytics Report

Source inspiration: https://github.com/robertmartin8/MachineLearningStocks
Best model by macro F1: `extra_trees`

## Model Comparison

| name              |   accuracy |   precision |   recall |   macro_f1 |   roc_auc |   selected_stocks |   selected_outperformance |
|:------------------|-----------:|------------:|---------:|-----------:|----------:|------------------:|--------------------------:|
| extra_trees       |   0.680238 |    0.554386 | 0.234074 |   0.559629 |  0.630819 |               285 |                   21.5314 |
| random_forest     |   0.695631 |    0.656566 | 0.192593 |   0.551764 |  0.634135 |               198 |                   27.2118 |
| extra_trees_tuned |   0.685204 |    0.591928 | 0.195556 |   0.545715 |  0.630882 |               223 |                   23.1245 |
| xgboost           |   0.671797 |    0.55     | 0.114074 |   0.491615 |  0.582634 |               140 |                   23.5582 |
| gradient_boosting |   0.672294 |    0.561983 | 0.100741 |   0.483323 |  0.565861 |               121 |                   26.4355 |

## Regression Framing

| name                   |   selected_stocks |   avg_selected_stock_return |   avg_selected_market_return |   selected_outperformance |   avg_selected_sharpe_proxy |
|:-----------------------|------------------:|----------------------------:|-----------------------------:|--------------------------:|----------------------------:|
| extra_trees_regression |                25 |                       80.47 |                       24.266 |                    56.204 |                     1.35211 |

## Threshold Experiment

| name                     |   macro_f1 |   selected_stocks |   selected_outperformance |
|:-------------------------|-----------:|------------------:|--------------------------:|
| extra_trees_threshold_5  |   0.526743 |               418 |                   15.9511 |
| extra_trees_threshold_10 |   0.558459 |               277 |                   21.5647 |
| extra_trees_threshold_15 |   0.567048 |               193 |                   26.807  |

## Portfolio-Level Risk Analytics

| model             |   positions |   unique_tickers |   avg_stock_return |   avg_market_return |   avg_outperformance |   return_volatility |   hit_rate |   sharpe_proxy |   worst_position_return |   best_position_return |
|:------------------|------------:|-----------------:|-------------------:|--------------------:|---------------------:|--------------------:|-----------:|---------------:|------------------------:|-----------------------:|
| random_forest     |         198 |               67 |            46.9725 |             19.7608 |              27.2118 |             54.2978 |   0.767677 |       0.865091 |                  -39.71 |                 447.48 |
| gradient_boosting |         121 |               48 |            46.8896 |             20.454  |              26.4355 |             64.0281 |   0.710744 |       0.732328 |                  -42.86 |                 447.48 |
| xgboost           |         140 |               48 |            43.8306 |             20.2724 |              23.5582 |             55.8847 |   0.642857 |       0.784303 |                  -54.8  |                 375.13 |
| extra_trees_tuned |         223 |               84 |            43.0584 |             19.9339 |              23.1245 |             51.7104 |   0.753363 |       0.832684 |                  -52.13 |                 447.48 |
| extra_trees       |         285 |              108 |            41.7986 |             20.2672 |              21.5314 |             52.7472 |   0.687719 |       0.792432 |                  -54.8  |                 447.48 |

## Top SHAP Drivers

| feature                       |   mean_abs_shap |
|:------------------------------|----------------:|
| Total Debt/Equity Missing     |      0.0154703  |
| Moving Average Spread         |      0.0119046  |
| Beta                          |      0.0118184  |
| Qtrly Earnings Growth Missing |      0.0116826  |
| Market Cap                    |      0.0112315  |
| Qtrly Revenue Growth Missing  |      0.0109443  |
| Enterprise Value              |      0.00997114 |
| Graham Number / Price         |      0.00965655 |
| Operating Margin              |      0.00861956 |
| 50-Day Moving Average         |      0.00850408 |
| 200-Day Moving Average        |      0.00844921 |
| Operating Cash Flow Missing   |      0.00839763 |
| Gross Profit                  |      0.00833723 |
| Shares Outstanding            |      0.00781543 |
| Float                         |      0.00745698 |

## Consensus Picks

| Ticker   |   votes |   avg_probability |
|:---------|--------:|------------------:|
| ma       |      45 |          0.762507 |
| aapl     |      43 |          0.789715 |
| wmt      |      43 |          0.666499 |
| dfs      |      43 |          0.606128 |
| fslr     |      33 |          0.544442 |
| bac      |      31 |          0.589145 |
| s        |      30 |          0.621466 |
| hov      |      24 |          0.606208 |
| znga     |      23 |          0.579107 |
| amd      |      22 |          0.579076 |
| ges      |      22 |          0.529116 |
| rsh      |      19 |          0.547265 |
| aeo      |      18 |          0.576923 |
| pfe      |      17 |          0.668884 |
| siri     |      17 |          0.588504 |
| gme      |      17 |          0.577827 |
| gnw      |      17 |          0.548953 |
| v        |      15 |          0.703791 |
| wfc      |      15 |          0.669659 |
| grpn     |      15 |          0.624717 |
| goog     |      14 |          0.553034 |
| mpc      |      13 |          0.641526 |
| sfly     |      12 |          0.675581 |
| nok      |      12 |          0.574571 |
| low      |      12 |          0.534303 |

## Top Forward Recommendations

| Ticker   |   predicted_outperformance_probability |
|:---------|---------------------------------------:|
| DSW      |                               0.700333 |
| JCP      |                               0.690768 |
| SHLD     |                               0.676088 |
| KLAC     |                               0.668864 |
| LRCX     |                               0.662137 |
| EA       |                               0.650832 |
| TIF      |                               0.650723 |
| WIN      |                               0.650699 |
| AVP      |                               0.649355 |
| SD       |                               0.645015 |
| WAT      |                               0.644509 |
| HOV      |                               0.642743 |
| RL       |                               0.630534 |
| BIIB     |                               0.623816 |
| BLK      |                               0.620758 |
| LIFE     |                               0.615713 |
| A        |                               0.611523 |
| NTAP     |                               0.609911 |
| ROK      |                               0.608326 |
| ZNGA     |                               0.60222  |
| MCO      |                               0.600058 |
| XLNX     |                               0.597115 |
| IFF      |                               0.59689  |
| APD      |                               0.595654 |
| HAS      |                               0.595576 |

## Improvement Over Source Project

- Preserved the original fundamentals-based stock outperformance idea.
- Replaced random train/test splitting with date-based validation to reduce leakage.
- Added model comparison across extra trees, random forest, and gradient boosting.
- Added ROC AUC, F1, precision, recall, selected-stock return, and index-relative outperformance metrics.
- Added missing-value-aware preprocessing, engineered valuation features, neural net option, XGBoost, tuned ExtraTrees, regression framing, threshold experiments, consensus recommendations, feature importance, SHAP explanations, and portfolio-level risk analytics.
- Added saved CSV and Markdown reports for reproducible evaluation.
