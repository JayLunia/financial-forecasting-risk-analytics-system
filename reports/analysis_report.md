# Financial Forecasting and Risk Analytics Report

Source inspiration: https://github.com/robertmartin8/MachineLearningStocks
Best model by macro F1: `extra_trees`

## Model Comparison

| name              |   accuracy |   precision |   recall |   macro_f1 |   roc_auc |   selected_stocks |   selected_outperformance |
|:------------------|-----------:|------------:|---------:|-----------:|----------:|------------------:|--------------------------:|
| extra_trees       |   0.680238 |    0.554386 | 0.234074 |   0.559629 |  0.630819 |               285 |                   21.5314 |
| random_forest     |   0.695631 |    0.656566 | 0.192593 |   0.551764 |  0.634135 |               198 |                   27.2118 |
| extra_trees_tuned |   0.685204 |    0.591928 | 0.195556 |   0.545715 |  0.630882 |               223 |                   23.1245 |
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

## Consensus Picks

| Ticker   |   votes |   avg_probability |
|:---------|--------:|------------------:|
| dfs      |      43 |          0.606128 |
| wmt      |      39 |          0.679861 |
| ma       |      36 |          0.736124 |
| aapl     |      35 |          0.782102 |
| fslr     |      26 |          0.544304 |
| bac      |      24 |          0.584853 |
| s        |      22 |          0.641484 |
| hov      |      19 |          0.599092 |
| znga     |      19 |          0.576293 |
| ges      |      19 |          0.529858 |
| siri     |      17 |          0.588504 |
| amd      |      17 |          0.569907 |
| rsh      |      17 |          0.54567  |
| pfe      |      16 |          0.678441 |
| wfc      |      14 |          0.669226 |
| aeo      |      14 |          0.583943 |
| goog     |      14 |          0.553034 |
| gme      |      13 |          0.548609 |
| v        |      12 |          0.692119 |
| grpn     |      12 |          0.601608 |
| nok      |      12 |          0.574571 |
| low      |      12 |          0.534303 |
| sfly     |      10 |          0.681326 |
| mpc      |      10 |          0.662335 |
| nflx     |      10 |          0.522937 |

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
- Added missing-value-aware preprocessing, engineered valuation features, SVM, neural net, tuned ExtraTrees, regression framing, threshold experiments, consensus recommendations, and feature importance.
- Added saved CSV and Markdown reports for reproducible evaluation.
