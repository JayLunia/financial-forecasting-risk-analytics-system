from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


REPORT_DIR = Path(__file__).resolve().parents[1] / "reports"


@st.cache_data
def read_report(name: str) -> pd.DataFrame:
    path = REPORT_DIR / name
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def metric_card(label: str, value: object) -> None:
    st.metric(label, value if isinstance(value, str) else f"{value:.3f}")


st.set_page_config(page_title="Stock Risk Analytics", layout="wide")
st.title("Financial Forecasting and Risk Analytics")
st.caption("Real fundamentals-based model comparison, SHAP explanations, and portfolio-level review.")

models = read_report("model_summary.csv")
thresholds = read_report("threshold_experiment.csv")
portfolio = read_report("portfolio_analytics.csv")
constrained_portfolio = read_report("portfolio_construction.csv")
constrained_summary = read_report("constrained_portfolio_summary.csv")
recommendations = read_report("forward_recommendations.csv")
consensus = read_report("consensus_recommendations.csv")
shap_summary = read_report("shap_summary.csv")
feature_importance = read_report("feature_importance.csv")

if models.empty:
    st.warning("Run `python -m src.pipeline --config configs/default.yaml` before opening the dashboard.")
    st.stop()

best = models.sort_values("macro_f1", ascending=False).iloc[0]
cols = st.columns(4)
with cols[0]:
    metric_card("Best Model", str(best["name"]))
with cols[1]:
    metric_card("Macro F1", best["macro_f1"])
with cols[2]:
    metric_card("ROC AUC", best["roc_auc"])
with cols[3]:
    metric_card("Selected Outperformance", best["selected_outperformance"])

tab_models, tab_portfolio, tab_explain, tab_recommendations = st.tabs([
    "Model Results",
    "Portfolio Risk",
    "Explainability",
    "Recommendations",
])

with tab_models:
    st.subheader("Model Comparison")
    st.dataframe(models, use_container_width=True)
    st.plotly_chart(
        px.bar(models, x="name", y="macro_f1", color="selected_outperformance", title="Macro F1 by model"),
        use_container_width=True,
    )
    if not thresholds.empty:
        st.subheader("Outperformance Threshold Experiment")
        st.dataframe(thresholds, use_container_width=True)
        st.plotly_chart(
            px.line(thresholds, x="name", y=["macro_f1", "selected_outperformance"], markers=True),
            use_container_width=True,
        )

with tab_portfolio:
    st.subheader("Portfolio-Level Risk Analytics")
    st.dataframe(portfolio, use_container_width=True)
    if not portfolio.empty:
        st.plotly_chart(
            px.scatter(
                portfolio,
                x="return_volatility",
                y="avg_outperformance",
                size="positions",
                color="hit_rate",
                hover_name="model",
                title="Risk vs outperformance by model",
            ),
            use_container_width=True,
        )
    st.subheader("Constrained Portfolio Construction")
    st.dataframe(constrained_summary, use_container_width=True)
    if not constrained_summary.empty:
        st.plotly_chart(
            px.bar(
                constrained_summary,
                x="model",
                y="net_outperformance_pct",
                color="total_transaction_cost_pct",
                title="Net outperformance after transaction costs",
            ),
            use_container_width=True,
        )
    if not constrained_portfolio.empty:
        st.subheader("Position Sizing")
        st.dataframe(constrained_portfolio, use_container_width=True)
        sector_view = (
            constrained_portfolio.groupby(["model", "Sector"], as_index=False)["position_weight"].sum()
        )
        st.plotly_chart(
            px.bar(
                sector_view,
                x="model",
                y="position_weight",
                color="Sector",
                title="Sector weights after concentration limits",
            ),
            use_container_width=True,
        )

with tab_explain:
    st.subheader("SHAP Summary")
    st.dataframe(shap_summary.head(25), use_container_width=True)
    if not shap_summary.empty:
        st.plotly_chart(
            px.bar(shap_summary.head(20), x="mean_abs_shap", y="feature", orientation="h", title="Top SHAP drivers"),
            use_container_width=True,
        )
    st.subheader("Tree Feature Importance")
    st.dataframe(feature_importance.head(25), use_container_width=True)

with tab_recommendations:
    st.subheader("Forward Recommendations")
    st.dataframe(recommendations, use_container_width=True)
    st.subheader("Consensus Recommendations")
    st.dataframe(consensus, use_container_width=True)
