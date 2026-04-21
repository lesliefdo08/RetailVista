from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from typing import Any, Dict

import streamlit as st


@dataclass
class LLMResponse:
    explanation: str
    recommendation: str
    risk: str
    source: str


class LLMService:
    def __init__(self) -> None:
        self.api_key = os.getenv("OPENROUTER_API_KEY", "PASTE_YOUR_API_KEY_HERE")
        self.model = os.getenv("OPENROUTER_MODEL", "mistralai/mixtral-8x7b")

    def _is_api_key_usable(self) -> bool:
        return bool(self.api_key and self.api_key != "PASTE_YOUR_API_KEY_HERE")

    def build_prompt(self, input_data: Dict[str, Any], prediction: float, stats: Dict[str, Any]) -> str:
        numeric_stats = stats.get("numeric", {})

        def _summary(col: str) -> str:
            s = numeric_stats.get(col, {})
            if not s:
                return "n/a"
            return f"min={s.get('min', 0):.3f}, mean={s.get('mean', 0):.3f}, max={s.get('max', 0):.3f}"

        return (
            "You are an AI business advisor for a retail forecasting tool. "
            "Return STRICT JSON with keys: explanation, recommendation, risk. "
            "Do not include markdown, bullets, or extra keys. "
            "Use specific, non-generic language and tie reasoning to numeric context.\n\n"
            f"Predicted monthly sales (INR): {prediction:.2f}\n"
            f"Input features: {json.dumps(input_data, ensure_ascii=True)}\n\n"
            "Dataset numeric ranges (historical):\n"
            f"- Item_MRP: {_summary('Item_MRP')}\n"
            f"- Item_Visibility: {_summary('Item_Visibility')}\n"
            f"- Item_Weight: {_summary('Item_Weight')}\n"
            f"- Outlet_Establishment_Year: {_summary('Outlet_Establishment_Year')}\n"
            f"- Item_Outlet_Sales: {_summary('Item_Outlet_Sales')}\n\n"
            "Instructions:\n"
            "1) Explain WHY this prediction is high/medium/low using the actual input values.\n"
            "2) Compare the key drivers to dataset means or ranges.\n"
            "3) Give one specific recommendation that is feasible for a small retailer.\n"
            "4) Provide one realistic risk tied to model limits or uncertain external factors.\n"
            "5) Avoid generic statements and avoid repeating the same phrase patterns."
        )

    def _pick(self, options: list[str], signature: str, salt: str) -> str:
        idx = int(hashlib.sha256(f"{signature}|{salt}".encode("utf-8")).hexdigest(), 16) % len(options)
        return options[idx]

    def local_fallback(self, input_data: Dict[str, Any], prediction: float, stats: Dict[str, Any]) -> Dict[str, str]:
        numeric = stats.get("numeric", {})
        mrp_mean = float(numeric.get("Item_MRP", {}).get("mean", 0) or 0)
        vis_mean = float(numeric.get("Item_Visibility", {}).get("mean", 0) or 0)
        sales_mean = float(numeric.get("Item_Outlet_Sales", {}).get("mean", 0) or 0)

        mrp = float(input_data.get("Item_MRP", 0) or 0)
        vis = float(input_data.get("Item_Visibility", 0) or 0)
        outlet_type = str(input_data.get("Outlet_Type", "Unknown"))
        location = str(input_data.get("Outlet_Location_Type", "Unknown"))

        signature = json.dumps(input_data, sort_keys=True)
        expl_parts = []
        rec_parts = []
        risk_parts = []

        if mrp_mean > 0 and mrp >= 1.2 * mrp_mean:
            expl_parts.append(
                self._pick(
                    [
                        "Price is materially above the historical average, so premium pricing is a primary revenue driver.",
                        "The product sits in a higher price band than typical records, which lifts expected sales value per unit.",
                    ],
                    signature,
                    "exp_price_high",
                )
            )
            rec_parts.append(
                self._pick(
                    [
                        "Protect margin with value-focused messaging rather than deep discounting.",
                        "Test a light bundle offer to increase basket size without eroding premium positioning.",
                    ],
                    signature,
                    "rec_price_high",
                )
            )
        elif mrp_mean > 0 and mrp <= 0.85 * mrp_mean:
            expl_parts.append(
                self._pick(
                    [
                        "Price is below the dataset average, so the forecast relies more on volume than ticket size.",
                        "Lower-than-average pricing supports accessibility, but caps per-item revenue.",
                    ],
                    signature,
                    "exp_price_low",
                )
            )
            rec_parts.append(
                self._pick(
                    [
                        "Use multi-buy offers to convert price advantage into higher throughput.",
                        "Improve cross-sell placement near complementary items to monetize volume traffic.",
                    ],
                    signature,
                    "rec_price_low",
                )
            )
        else:
            expl_parts.append("Pricing is close to historical center, so non-price factors are likely driving most of the variation.")

        if vis_mean > 0 and vis <= 0.85 * vis_mean:
            expl_parts.append(
                self._pick(
                    [
                        "Visibility is below typical levels, which can suppress conversion despite acceptable pricing.",
                        "The product is less discoverable than average, limiting demand capture at shelf level.",
                    ],
                    signature,
                    "exp_vis_low",
                )
            )
            rec_parts.append(
                self._pick(
                    [
                        "Move the SKU to eye-level or high-traffic aisles for a short A/B test window.",
                        "Allocate end-cap or checkout adjacency for one cycle and track weekly uplift.",
                    ],
                    signature,
                    "rec_vis_low",
                )
            )
            risk_parts.append("Low visibility increases forecast sensitivity; realized sales may undershoot if placement remains unchanged.")
        elif vis_mean > 0 and vis >= 1.15 * vis_mean:
            expl_parts.append("Visibility is stronger than average, supporting customer discovery and improving conversion odds.")
            rec_parts.append("Keep display quality consistent and avoid placement drift during replenishment cycles.")

        if sales_mean > 0:
            if prediction > 1.2 * sales_mean:
                expl_parts.append("Predicted sales are above the historical mean, indicating a comparatively strong expected outcome.")
                risk_parts.append("Higher forecasts can amplify stock-out risk; inventory buffers should be validated before promotions.")
            elif prediction < 0.8 * sales_mean:
                expl_parts.append("Predicted sales are below the historical mean, signaling underperformance risk under current conditions.")
                risk_parts.append("If local demand is weaker than historical patterns, markdown pressure may rise.")

        if location == "Tier 3":
            rec_parts.append("Use local event-led promotions and community channels that typically perform better in small-town contexts.")
        if outlet_type == "Grocery Store":
            risk_parts.append("Smaller-format stores may face assortment and footfall constraints that are not fully captured by static features.")

        if not risk_parts:
            risk_parts.append("External factors such as seasonality, competitor campaigns, and local events are not directly modeled.")

        explanation = " ".join(dict.fromkeys(expl_parts))
        recommendation = " ".join(dict.fromkeys(rec_parts))
        risk = " ".join(dict.fromkeys(risk_parts))

        return {
            "explanation": explanation,
            "recommendation": recommendation,
            "risk": risk,
            "source": "Local fallback",
        }

    def _parse_or_fallback(self, raw_text: str, input_data: Dict[str, Any], prediction: float, stats: Dict[str, Any], source: str) -> Dict[str, str]:
        try:
            cleaned = raw_text.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.strip("`")
                cleaned = cleaned.replace("json\n", "", 1).strip()
            data = json.loads(cleaned)
            return {
                "explanation": data.get("explanation", ""),
                "recommendation": data.get("recommendation", ""),
                "risk": data.get("risk", data.get("risk_note", "")),
                "source": source,
            }
        except Exception:
            return self.local_fallback(input_data, prediction, stats)

    def call_openrouter(self, prompt: str) -> Dict[str, str]:
        cache_key = hashlib.sha256(prompt.encode("utf-8")).hexdigest()
        cache = st.session_state.setdefault("llm_prompt_cache", {})
        if cache_key in cache:
            return cache[cache_key]

        import requests

        response = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": "You are a practical retail analytics advisor. Respond in valid JSON only.",
                    },
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.25,
            },
            timeout=15,
        )
        response.raise_for_status()
        body = response.json()
        text = body["choices"][0]["message"]["content"]
        cache[cache_key] = {"content": text}
        return cache[cache_key]

    def generate_insights(self, input_data: Dict[str, Any], prediction_inr: float, stats: Dict[str, Any]) -> LLMResponse:
        prompt = self.build_prompt(input_data, prediction_inr, stats)

        if self._is_api_key_usable():
            try:
                raw = self.call_openrouter(prompt)
                result = self._parse_or_fallback(raw.get("content", "{}"), input_data, prediction_inr, stats, source="OpenRouter")
            except Exception:
                result = self.local_fallback(input_data, prediction_inr, stats)
        else:
            result = self.local_fallback(input_data, prediction_inr, stats)

        return LLMResponse(
            explanation=result["explanation"],
            recommendation=result["recommendation"],
            risk=result["risk"],
            source=result["source"],
        )
