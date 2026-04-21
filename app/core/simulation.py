from __future__ import annotations

from typing import Any, Dict, List

from app.core.model import predict_sales
from app.core.preprocess import prepare_features


def simulate_scenarios(model: Any, preprocessor: Any, input_data, usd_to_inr: float) -> List[Dict[str, float]]:
    scenarios: List[Dict[str, float]] = []

    base_features = prepare_features(input_data)
    original_pred = float(predict_sales(model, preprocessor, base_features)[0])

    visibility_levels = {
        "Low": 0.03,
        "Medium": 0.08,
        "High": 0.15,
    }
    current_vis = float(input_data["Item_Visibility"].iloc[0])
    for label, value in visibility_levels.items():
        if value <= current_vis:
            continue
        cand = input_data.copy()
        cand["Item_Visibility"] = value
        pred = float(predict_sales(model, preprocessor, prepare_features(cand))[0])
        scenarios.append(
            {
                "scenario": f"Increase product visibility to {label}",
                "original_sales": original_pred * usd_to_inr,
                "new_sales": pred * usd_to_inr,
                "improvement_pct": ((pred - original_pred) / original_pred) * 100.0,
            }
        )

    size_order = ["Small", "Medium", "High"]
    current_size = str(input_data["Outlet_Size"].iloc[0])
    if current_size in size_order and size_order.index(current_size) < len(size_order) - 1:
        next_size = size_order[size_order.index(current_size) + 1]
        cand = input_data.copy()
        cand["Outlet_Size"] = next_size
        pred = float(predict_sales(model, preprocessor, prepare_features(cand))[0])
        scenarios.append(
            {
                "scenario": f"Upgrade store size to {next_size}",
                "original_sales": original_pred * usd_to_inr,
                "new_sales": pred * usd_to_inr,
                "improvement_pct": ((pred - original_pred) / original_pred) * 100.0,
            }
        )

    return scenarios
