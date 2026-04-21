from __future__ import annotations

from typing import Dict, List

import pandas as pd


def calculate_benchmarks(df: pd.DataFrame, input_data: pd.DataFrame, prediction: float, usd_to_inr: float) -> List[Dict[str, float]]:
    results: List[Dict[str, float]] = []

    item_type = str(input_data["Item_Type"].iloc[0])
    item_avg = float(df[df["Item_Type"] == item_type]["Item_Outlet_Sales"].mean())
    if item_avg > 0:
        diff = ((prediction - item_avg) / item_avg) * 100.0
        results.append(
            {
                "segment": f"{item_type} category",
                "average": item_avg * usd_to_inr,
                "estimate": prediction * usd_to_inr,
                "status": "Above Average" if diff >= 0 else "Below Average",
                "difference_pct": abs(diff),
            }
        )

    location = str(input_data["Outlet_Location_Type"].iloc[0])
    location_avg = float(df[df["Outlet_Location_Type"] == location]["Item_Outlet_Sales"].mean())
    if location_avg > 0:
        diff = ((prediction - location_avg) / location_avg) * 100.0
        results.append(
            {
                "segment": f"{location} outlets",
                "average": location_avg * usd_to_inr,
                "estimate": prediction * usd_to_inr,
                "status": "Above Average" if diff >= 0 else "Below Average",
                "difference_pct": abs(diff),
            }
        )

    return results
