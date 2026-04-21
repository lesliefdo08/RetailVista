from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from app.core.preprocess import get_feature_columns


def compute_feature_stats(train_df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
    numeric_features, _ = get_feature_columns()
    stats: Dict[str, Dict[str, float]] = {}
    for col in numeric_features:
        q1 = float(train_df[col].quantile(0.25))
        q3 = float(train_df[col].quantile(0.75))
        iqr = q3 - q1
        stats[col] = {
            "q1": q1,
            "q3": q3,
            "iqr": iqr,
            "lower": q1 - 1.5 * iqr,
            "upper": q3 + 1.5 * iqr,
        }
    return stats


def calculate_confidence_level(input_df: pd.DataFrame, feature_stats: Dict[str, Dict[str, float]]) -> Tuple[str, str]:
    if not feature_stats:
        return "Medium", "No training stats found; confidence defaults to Medium."

    scores = []
    outlier_cols = []
    for col, stat in feature_stats.items():
        if col not in input_df.columns:
            continue
        value = float(input_df[col].iloc[0])
        if stat["lower"] <= value <= stat["upper"]:
            scores.append(1.0)
        else:
            scores.append(0.5)
            outlier_cols.append(col)

    avg = float(np.mean(scores)) if scores else 0.75
    if avg >= 0.9:
        level = "High"
    elif avg >= 0.7:
        level = "Medium"
    else:
        level = "Low"

    if outlier_cols:
        note = f"Extreme input values detected for: {', '.join(outlier_cols)}"
    else:
        note = "Input values are within typical training ranges."
    return level, note
