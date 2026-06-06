# Financial Forecasting and Risk Analytics Report

Source inspiration: https://github.com/robertmartin8/MachineLearningStocks
Best model by macro F1: `extra_trees`

## Model Comparison

| name              |   accuracy |   precision |    recall |   macro_f1 |   roc_auc |   selected_stocks |   selected_outperformance |
|:------------------|-----------:|------------:|----------:|-----------:|----------:|------------------:|--------------------------:|
| extra_trees       |   0.683714 |    0.557229 | 0.274074  |   0.578285 |  0.632756 |               332 |                   19.7618 |
| random_forest     |   0.681728 |    0.577273 | 0.188148  |   0.539601 |  0.626366 |               220 |                   20.1044 |
| gradient_boosting |   0.669811 |    0.542373 | 0.0948148 |   0.477924 |  0.551046 |               118 |                   18.1195 |

## Top Forward Recommendations

| Ticker   |   predicted_outperformance_probability |
|:---------|---------------------------------------:|
| BLK      |                               0.721693 |
| BIIB     |                               0.692329 |
| EXPR     |                               0.690379 |
| NBR      |                               0.660272 |
| JCP      |                               0.646991 |
| RDC      |                               0.637703 |
| GTN      |                               0.633856 |
| AVP      |                               0.633262 |
| HOV      |                               0.633262 |
| GNW      |                               0.629007 |
| NE       |                               0.619771 |
| FTR      |                               0.618019 |
| AMP      |                               0.616821 |
| WIN      |                               0.615986 |
| FSLR     |                               0.611231 |
| CAMP     |                               0.607177 |
| CLF      |                               0.604343 |
| SD       |                               0.601417 |
| LH       |                               0.600994 |
| GCI      |                               0.600223 |
| GRPN     |                               0.597183 |
| SFLY     |                               0.596257 |
| GME      |                               0.594388 |
| KR       |                               0.591774 |
| RL       |                               0.590532 |

## Improvement Over Source Project

- Preserved the original fundamentals-based stock outperformance idea.
- Replaced random train/test splitting with date-based validation to reduce leakage.
- Added model comparison across extra trees, random forest, and gradient boosting.
- Added ROC AUC, F1, precision, recall, selected-stock return, and index-relative outperformance metrics.
- Added saved CSV and Markdown reports for reproducible evaluation.
