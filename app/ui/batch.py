from __future__ import annotations

import streamlit as st

from app.config.constants import MAX_UPLOAD_ROWS, REQUIRED_COLUMNS
from app.core.model import predict_sales
from app.core.preprocess import prepare_features, validate_input_schema
from app.core.utils import format_currency


def render_batch_prediction(model, preprocessor) -> None:
    st.subheader("Batch Prediction")
    st.caption("Upload CSV with the required schema.")
    st.code(", ".join(REQUIRED_COLUMNS))

    uploaded = st.file_uploader("Upload CSV", type=["csv"], key="batch_upload")
    if uploaded is None:
        return

    try:
        import pandas as pd

        df = pd.read_csv(uploaded)
    except Exception:
        st.error("Could not parse CSV file. Please upload a valid CSV.")
        return

    if len(df) > MAX_UPLOAD_ROWS:
        st.error(f"File too large. Maximum allowed rows: {MAX_UPLOAD_ROWS}")
        return

    schema_errors = validate_input_schema(df)
    if schema_errors["missing_columns"]:
        st.error(f"Missing required columns: {', '.join(schema_errors['missing_columns'])}")
        return
    if schema_errors["invalid_types"]:
        st.error(f"Invalid numeric values in: {', '.join(schema_errors['invalid_types'])}")
        return

    st.success(f"Loaded {len(df)} rows")
    st.dataframe(df.head(), use_container_width=True)

    if st.button("Run Batch Prediction", type="primary"):
        features = prepare_features(df)
        preds = predict_sales(model, preprocessor, features)
        out = df.copy()
        out["Estimated_Monthly_Sales"] = preds
        out["Estimated_Monthly_Sales_INR"] = out["Estimated_Monthly_Sales"].map(format_currency)
        st.dataframe(out, use_container_width=True)
        st.download_button(
            "Download Results CSV",
            out.to_csv(index=False),
            file_name="retailvista_batch_predictions.csv",
            mime="text/csv",
        )
