# RetailVista

RetailVista is an ML-powered retail forecasting application that predicts monthly product sales and provides decision-support insights for store operators.

## Overview

RetailVista combines a supervised regression model with explainability and advisory layers:

- Predicts sales using structured product and outlet attributes
- Provides scenario simulation for what-if planning
- Surfaces explainability-driven factors for each estimate
- Adds AI-assisted business interpretation for non-technical users

This project is structured for demo, submission, and deployment on Streamlit.

## Features

- ML Prediction (XGBoost)
- AI Business Advisor (LLM-powered via OpenRouter with local fail-safe)
- Scenario Simulation
- Explainability (SHAP with robust fallback)
- Dynamic model performance dashboard based on held-out artifacts

## Tech Stack

- Python
- Streamlit
- XGBoost
- scikit-learn
- Pandas
- NumPy
- Plotly
- SHAP
- OpenRouter API

## Project Structure

```
RetailVista/
├── app/
│   ├── config/
│   ├── core/
│   ├── services/
│   ├── ui/
│   └── main.py
├── artifacts/
│   ├── feature_stats.json
│   ├── test_metrics.json
│   └── test_predictions.csv
├── scripts/
│   └── train_pipeline.py
├── app.py
├── requirements.txt
└── README.md
```

## Setup Instructions

1. Install dependencies

```bash
pip install -r requirements.txt
```

2. Train model and generate artifacts

```bash
python scripts/train_pipeline.py
```

3. Run the Streamlit app

```bash
streamlit run app.py
```

## Environment Variables

Set the following variable to enable real LLM mode:

```bash
OPENROUTER_API_KEY=PASTE_YOUR_API_KEY_HERE
```

Optional model override:

```bash
OPENROUTER_MODEL=mistralai/mixtral-8x7b
```

## Screenshots

- Add home/prediction screen screenshot
- Add model performance screen screenshot
- Add AI advisor output screenshot

## Streamlit Deployment Notes

- Entry file: `app.py`
- Paths are relative/project-root aware through centralized constants
- Required dependencies are listed in `requirements.txt`

## Disclaimer

- This is an educational prototype.
- Predictions are based on historical dataset patterns.
- Insights should support business decisions, not replace domain judgment.
- The model does not directly include real-time external factors like competitor campaigns, weather, or live pricing shocks.

## Author

Leslie Fernando
