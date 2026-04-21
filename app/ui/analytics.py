from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from app.config.constants import USD_TO_INR
from app.core.utils import format_currency
from app.services.data_loader import load_test_metrics, load_test_predictions


def render_analytics_tab() -> None:
    st.header("Model Performance")

    metrics = load_test_metrics()
    preds = load_test_predictions()

    if not metrics or preds.empty:
        st.warning("No evaluation artifacts found. Run scripts/train_pipeline.py first.")
        return

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.metric("Average Error", format_currency(metrics.get("test_mae", 0) * USD_TO_INR))
    with c2:
        st.metric("RMSE", format_currency(metrics.get("test_rmse", 0) * USD_TO_INR))
    with c3:
        st.metric("R2", f"{metrics.get('test_r2', 0) * 100:.2f}%")
    with c4:
        st.metric("Dataset Size", str(metrics.get("dataset_size", "unknown")))
    with c5:
        st.metric("Last Trained", str(metrics.get("trained_at", "unknown")))

    st.caption("Metrics are computed on a held-out test dataset, not training data.")

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=preds["actual"] * USD_TO_INR,
            y=preds["predicted"] * USD_TO_INR,
            mode="markers",
            marker={"size": 6, "opacity": 0.6},
            name="Predictions",
        )
    )

    min_val = min((preds["actual"] * USD_TO_INR).min(), (preds["predicted"] * USD_TO_INR).min())
    max_val = max((preds["actual"] * USD_TO_INR).max(), (preds["predicted"] * USD_TO_INR).max())
    fig.add_trace(
        go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode="lines",
            line={"dash": "dash", "color": "red"},
            name="Perfect fit",
        )
    )
    fig.update_layout(
        title="Actual vs Predicted (Saved Test Set)",
        xaxis_title="Actual Sales (INR)",
        yaxis_title="Predicted Sales (INR)",
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    if "actual" in preds.columns and "predicted" in preds.columns:
        residuals = (preds["actual"] - preds["predicted"]) * USD_TO_INR
        fig_res = go.Figure()
        fig_res.add_trace(
            go.Scatter(
                x=preds["predicted"] * USD_TO_INR,
                y=residuals,
                mode="markers",
                marker={"size": 6, "opacity": 0.55},
                name="Residuals",
            )
        )
        fig_res.add_hline(y=0, line_dash="dash", line_color="red")
        fig_res.update_layout(
            title="Residual Plot (Saved Test Set)",
            xaxis_title="Predicted Sales (INR)",
            yaxis_title="Residual (Actual - Predicted, INR)",
            height=420,
        )
        st.plotly_chart(fig_res, use_container_width=True)

    st.info("How to read: points closer to red line indicate better prediction quality.")
