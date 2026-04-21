from __future__ import annotations

import pandas as pd
import streamlit as st

from app.config.constants import REQUIRED_COLUMNS, USD_TO_INR
from app.core.benchmarking import calculate_benchmarks
from app.core.confidence import calculate_confidence_level
from app.core.explain import shap_local_explanation
from app.core.model import predict_sales
from app.core.preprocess import prepare_features, validate_input_schema
from app.core.simulation import simulate_scenarios
from app.core.utils import format_currency
from app.services.data_loader import load_dataset, load_dataset_stats, load_feature_stats
from app.services.llm_service import LLMService
from app.ui.batch import render_batch_prediction


def _generate_rule_based_insights(input_data: pd.DataFrame) -> list[str]:
    insights = []
    vis = float(input_data["Item_Visibility"].iloc[0])
    mrp = float(input_data["Item_MRP"].iloc[0])
    loc = str(input_data["Outlet_Location_Type"].iloc[0])

    if vis < 0.05:
        insights.append("Rule-based insight: Improve shelf placement to increase product visibility.")
    if mrp > 200:
        insights.append("Rule-based insight: Evaluate promotional pricing or bundle discounts for premium-priced products.")
    if loc == "Tier 3":
        insights.append("Rule-based insight: Focus on local campaigns and community promotions in small-town markets.")

    if not insights:
        insights.append("Rule-based insight: Inputs look balanced; prioritize execution quality and in-store visibility tests.")
    return insights[:3]


def render_prediction_tab(model, preprocessor) -> None:
    st.header("Get Sales Estimate")
    st.info(
        "This tool helps shop owners and students estimate monthly sales and understand how to improve them."
    )

    with st.expander("Before You Start", expanded=False):
        st.markdown(
            "- Product Visibility: How easy it is to notice on shelves.\n"
            "- Store Size: Small, Medium, or High capacity outlet.\n"
            "- City Type: Metro, developing city, or small town market.\n"
            "- MRP: Enter retail price in the original dataset-like scale.\n"
            "- Note: Correlation does not imply causation."
        )

    mode = st.radio("Input mode", ["Single Product", "CSV Upload"], horizontal=True)
    if mode == "CSV Upload":
        render_batch_prediction(model, preprocessor)
        return

    dataset = load_dataset()

    col1, col2 = st.columns(2)
    with col1:
        item_weight = st.number_input("Product Weight", min_value=0.0, value=10.0)
        item_fat_content = st.selectbox("Fat Content", sorted(dataset["Item_Fat_Content"].dropna().unique()))
        visibility_label = st.select_slider("Product Visibility", options=["Low", "Medium", "High"], value="Medium")
        vis_map = {"Low": 0.03, "Medium": 0.08, "High": 0.15}
        item_visibility = vis_map[visibility_label]
        item_type = st.selectbox("Product Category", sorted(dataset["Item_Type"].dropna().unique()))
        item_mrp = st.number_input("MRP", min_value=0.0, value=100.0, help="Use the same scale as historical dataset values.")
    with col2:
        outlet_year = st.number_input("Store Establishment Year", min_value=1980, max_value=2030, value=2000)
        outlet_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        city_type = st.selectbox("City Type", ["Metro", "Developing City", "Small Town"])
        city_map = {"Metro": "Tier 1", "Developing City": "Tier 2", "Small Town": "Tier 3"}
        outlet_location_type = city_map[city_type]
        store_format = st.selectbox(
            "Store Format",
            ["Small Grocery", "Supermarket Type 1", "Supermarket Type 2", "Supermarket Type 3"],
        )
        format_map = {
            "Small Grocery": "Grocery Store",
            "Supermarket Type 1": "Supermarket Type1",
            "Supermarket Type 2": "Supermarket Type2",
            "Supermarket Type 3": "Supermarket Type3",
        }
        outlet_type = format_map[store_format]

    if not st.button("Estimate Sales", type="primary", use_container_width=True):
        return

    input_data = pd.DataFrame(
        {
            "Item_Weight": [item_weight],
            "Item_Fat_Content": [item_fat_content],
            "Item_Visibility": [item_visibility],
            "Item_Type": [item_type],
            "Item_MRP": [item_mrp],
            "Outlet_Establishment_Year": [outlet_year],
            "Outlet_Size": [outlet_size],
            "Outlet_Location_Type": [outlet_location_type],
            "Outlet_Type": [outlet_type],
        }
    )

    schema_errors = validate_input_schema(input_data)
    if schema_errors["missing_columns"] or schema_errors["invalid_types"]:
        st.error("Input validation failed. Please review your values.")
        return

    features = prepare_features(input_data)
    prediction = float(predict_sales(model, preprocessor, features)[0])
    prediction_inr = prediction * USD_TO_INR

    stats = load_feature_stats()
    confidence_level, confidence_note = calculate_confidence_level(features, stats.get("numeric_bounds", {}))

    st.subheader("Business Summary")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric("Model Prediction (INR)", format_currency(prediction_inr))
    with c2:
        st.metric("Expected Range", f"{format_currency(prediction_inr * 0.85)} - {format_currency(prediction_inr * 1.15)}")
    with c3:
        st.metric("Confidence", confidence_level)
    with c4:
        st.metric("Primary Lever", "Visibility / Pricing")
    st.caption(confidence_note)

    with st.expander("Why This Sales Estimate?", expanded=False):
        local_exp, method = shap_local_explanation(model, preprocessor, features)
        st.caption(f"Method: {method}")
        st.caption("Model Prediction Insight")
        for row in local_exp:
            st.write(f"- {row['feature']}: {row['direction']} prediction")

    with st.expander("Scenario Simulation (Not Retraining)", expanded=False):
        scenarios = simulate_scenarios(model, preprocessor, input_data, USD_TO_INR)
        if not scenarios:
            st.info("No actionable scenario generated from current inputs.")
        for sc in scenarios:
            st.write(f"- {sc['scenario']}")
            st.write(
                f"  Original: {format_currency(sc['original_sales'])} | "
                f"New: {format_currency(sc['new_sales'])} | Improvement: {sc['improvement_pct']:.2f}%"
            )

    with st.expander("Category Benchmarks", expanded=False):
        benchmarks = calculate_benchmarks(dataset, input_data, prediction, USD_TO_INR)
        for b in benchmarks:
            st.write(
                f"- {b['segment']}: {b['status']} by {b['difference_pct']:.2f}% "
                f"(Avg {format_currency(b['average'])}, Estimate {format_currency(b['estimate'])})"
            )

    st.subheader("How Can You Improve These Sales?")
    st.caption("Rule-Based Insight")
    for tip in _generate_rule_based_insights(input_data):
        st.info(tip)

    st.subheader("AI Business Advisor")
    st.caption("AI-Generated Insight")
    llm = LLMService()
    dataset_stats = load_dataset_stats()
    ai = llm.generate_insights(
        input_data.iloc[0].to_dict(),
        prediction_inr,
        dataset_stats,
    )
    st.caption(f"Advisor source: {ai.source}")
    st.write(f"Explanation: {ai.explanation}")
    st.write(f"Recommendation: {ai.recommendation}")
    st.write(f"Risk: {ai.risk}")
    st.caption("Insights are based on historical patterns and should support-not replace-business decisions.")
