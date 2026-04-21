from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from app.core.preprocess import get_feature_columns


def global_feature_factors(model: Any, preprocessor: Any, top_n: int = 3) -> List[Dict[str, float]]:
    importances = model.feature_importances_
    numeric_features, categorical_features = get_feature_columns()
    feature_names = numeric_features + list(
        preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
    )
    ranked = sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)[:top_n]
    return [{"feature": name, "importance": float(score)} for name, score in ranked]


def shap_local_explanation(model: Any, preprocessor: Any, input_features: pd.DataFrame) -> Tuple[List[Dict[str, float]], str]:
    try:
        import shap

        transformed = preprocessor.transform(input_features)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(transformed)

        if isinstance(shap_values, list):
            values = shap_values[0]
        else:
            values = shap_values

        row_values = np.array(values[0]).flatten()

        numeric_features, categorical_features = get_feature_columns()
        names = numeric_features + list(
            preprocessor.named_transformers_["cat"].get_feature_names_out(categorical_features)
        )

        ranked = sorted(zip(names, row_values), key=lambda x: abs(x[1]), reverse=True)[:3]
        explanation = []
        for name, val in ranked:
            direction = "increases" if val >= 0 else "decreases"
            explanation.append(
                {
                    "feature": name,
                    "impact": float(val),
                    "direction": direction,
                }
            )

        return explanation, "SHAP local explanation"
    except Exception:
        factors = global_feature_factors(model, preprocessor, top_n=3)
        fallback = [
            {
                "feature": f["feature"],
                "impact": f["importance"],
                "direction": "influences",
            }
            for f in factors
        ]
        return fallback, "Global feature importance fallback"
