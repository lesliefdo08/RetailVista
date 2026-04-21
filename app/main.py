from __future__ import annotations

import streamlit as st

from app.core.model import load_model_bundle
from app.ui.analytics import render_analytics_tab
from app.ui.prediction import render_prediction_tab


def run_app() -> None:
    st.set_page_config(page_title="RetailVista", page_icon="RV", layout="wide")

    model, preprocessor, _ = load_model_bundle()

    st.title("RetailVista")
    tabs = st.tabs(["Get Sales Estimate", "Model Performance", "About This Tool"])

    with tabs[0]:
        render_prediction_tab(model, preprocessor)
    with tabs[1]:
        render_analytics_tab()
    with tabs[2]:
        st.header("About")
        st.markdown(
            "### What RetailVista Is\n"
            "RetailVista is an ML-powered retail forecasting tool that estimates product-level monthly sales and translates model outputs into practical business guidance.\n\n"
            "### Problem It Solves\n"
            "Small retailers often make pricing, assortment, and shelf placement decisions using intuition only. RetailVista provides data-driven estimates and decision support so store operators can test choices with evidence.\n\n"
            "### Key Features\n"
            "- XGBoost prediction engine for sales estimation\n"
            "- Scenario simulation for what-if planning\n"
            "- AI Business Advisor for contextual narrative insights\n"
            "- Explainability layer with SHAP-based local reasoning and robust fallback\n\n"
            "### Technical Stack\n"
            "- Streamlit (application UI)\n"
            "- XGBoost (regression model)\n"
            "- Pandas and NumPy (data processing)\n"
            "- OpenRouter (LLM integration layer with fail-safe fallback)\n\n"
            "### How AI Is Used In This Tool\n"
            "- Machine Learning (XGBoost): predicts numerical sales values from structured product and outlet features.\n"
            "- LLM Advisor: explains prediction drivers, suggests actions, and highlights risks in business language.\n\n"
            "### Important Honesty Notes\n"
            "- This is an educational prototype.\n"
            "- Predictions are based on a historical dataset and may not reflect live market shifts.\n"
            "- External drivers like competitor actions, weather, festivals, and macroeconomic shocks are not modeled directly."
        )
        st.caption("Insights are based on historical patterns and should support-not replace-business decisions.")


if __name__ == "__main__":
    run_app()
