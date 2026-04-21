from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import pandas as pd
import streamlit as st

from app.config.constants import (
    DATA_PATH,
    FEATURE_STATS_PATH,
    TEST_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
)


@st.cache_data
def load_dataset() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH)


@st.cache_data
def load_dataset_stats() -> Dict[str, Any]:
    df = load_dataset()
    numeric_cols = [
        "Item_Weight",
        "Item_Visibility",
        "Item_MRP",
        "Outlet_Establishment_Year",
        "Item_Outlet_Sales",
    ]
    stats: Dict[str, Any] = {"row_count": int(len(df)), "numeric": {}}
    for col in numeric_cols:
        if col not in df.columns:
            continue
        series = pd.to_numeric(df[col], errors="coerce")
        stats["numeric"][col] = {
            "min": float(series.min()),
            "max": float(series.max()),
            "mean": float(series.mean()),
        }
    return stats


@st.cache_data
def load_test_metrics() -> Dict[str, Any]:
    if not TEST_METRICS_PATH.exists():
        return {}
    return json.loads(TEST_METRICS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_feature_stats() -> Dict[str, Any]:
    if not FEATURE_STATS_PATH.exists():
        return {}
    return json.loads(FEATURE_STATS_PATH.read_text(encoding="utf-8"))


@st.cache_data
def load_test_predictions() -> pd.DataFrame:
    if not TEST_PREDICTIONS_PATH.exists():
        return pd.DataFrame()
    return pd.read_csv(TEST_PREDICTIONS_PATH)


def ensure_artifacts_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
