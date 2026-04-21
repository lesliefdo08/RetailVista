from __future__ import annotations

from typing import Dict, List, Tuple

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from app.config.constants import REQUIRED_COLUMNS


def get_feature_columns() -> Tuple[List[str], List[str]]:
    numeric_features = [
        "Item_Weight",
        "Item_Visibility",
        "Item_MRP",
        "Outlet_Establishment_Year",
    ]
    categorical_features = [
        "Item_Fat_Content",
        "Item_Type",
        "Outlet_Size",
        "Outlet_Location_Type",
        "Outlet_Type",
    ]
    return numeric_features, categorical_features


def create_preprocessor() -> ColumnTransformer:
    numeric_features, categorical_features = get_feature_columns()
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_features),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=True), categorical_features),
        ],
        remainder="drop",
    )


def validate_input_schema(df: pd.DataFrame) -> Dict[str, List[str]]:
    errors: Dict[str, List[str]] = {"missing_columns": [], "invalid_types": []}
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        errors["missing_columns"] = missing
        return errors

    numeric_features, _ = get_feature_columns()
    for col in numeric_features:
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.isna().any():
            errors["invalid_types"].append(col)

    return errors


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    numeric_features, categorical_features = get_feature_columns()
    all_features = numeric_features + categorical_features
    df_features = df[all_features].copy()

    for col in numeric_features:
        df_features[col] = pd.to_numeric(df_features[col], errors="coerce")

    df_features[numeric_features] = df_features[numeric_features].fillna(
        df_features[numeric_features].median(numeric_only=True)
    )

    for col in categorical_features:
        df_features[col] = df_features[col].astype(str).replace({"nan": "Unknown"}).fillna("Unknown")

    return df_features
