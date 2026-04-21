from __future__ import annotations

from typing import Any, Dict, Tuple

import joblib
import streamlit as st

from app.config.constants import METRICS_PKL_PATH, MODEL_PATH, PREPROCESSOR_PATH


@st.cache_resource
def load_model_bundle() -> Tuple[Any, Any, Dict[str, float]]:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
    metrics = joblib.load(METRICS_PKL_PATH) if METRICS_PKL_PATH.exists() else {}
    return model, preprocessor, metrics


def predict_sales(model: Any, preprocessor: Any, features):
    transformed = preprocessor.transform(features)
    return model.predict(transformed)
