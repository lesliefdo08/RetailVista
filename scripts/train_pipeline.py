from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor

from app.config.constants import (
    ARTIFACTS_DIR,
    CV_METRICS_PATH,
    DATA_PATH,
    FEATURE_STATS_PATH,
    MODEL_PATH,
    METRICS_PKL_PATH,
    PREPROCESSOR_PATH,
    TEST_METRICS_PATH,
    TEST_PREDICTIONS_PATH,
)
from app.core.confidence import compute_feature_stats
from app.core.preprocess import create_preprocessor, get_feature_columns, prepare_features


def _rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def _mape(y_true, y_pred) -> float:
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    denom = np.where(np.abs(y_true_arr) < 1e-8, np.nan, np.abs(y_true_arr))
    ape = np.abs((y_true_arr - y_pred_arr) / denom)
    return float(np.nanmean(ape) * 100.0)


def main() -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)
    X = prepare_features(df)
    y = df["Item_Outlet_Sales"].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    tuning_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            (
                "model",
                XGBRegressor(
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    param_distributions = {
        "model__n_estimators": [200, 300, 400, 500, 600],
        "model__max_depth": [4, 5, 6, 7, 8],
        "model__learning_rate": [0.03, 0.05, 0.07, 0.1],
        "model__subsample": [0.8, 0.9, 1.0],
        "model__colsample_bytree": [0.8, 0.9, 1.0],
        "model__min_child_weight": [1, 3, 5],
    }

    search = RandomizedSearchCV(
        estimator=tuning_pipeline,
        param_distributions=param_distributions,
        n_iter=20,
        scoring="neg_root_mean_squared_error",
        cv=KFold(n_splits=5, shuffle=True, random_state=42),
        random_state=42,
        n_jobs=-1,
        refit=True,
        verbose=0,
    )
    search.fit(X_train, y_train)

    best_pipeline = search.best_estimator_
    preprocessor = best_pipeline.named_steps["preprocessor"]
    model = best_pipeline.named_steps["model"]

    X_train_t = preprocessor.transform(X_train)
    X_test_t = preprocessor.transform(X_test)

    train_pred = model.predict(X_train_t)
    test_pred = model.predict(X_test_t)

    metrics = {
        "train_rmse": _rmse(y_train, train_pred),
        "train_mae": float(mean_absolute_error(y_train, train_pred)),
        "train_mape": _mape(y_train, train_pred),
        "train_median_ae": float(median_absolute_error(y_train, train_pred)),
        "train_r2": float(r2_score(y_train, train_pred)),
        "test_rmse": _rmse(y_test, test_pred),
        "test_mae": float(mean_absolute_error(y_test, test_pred)),
        "test_mape": _mape(y_test, test_pred),
        "test_median_ae": float(median_absolute_error(y_test, test_pred)),
        "test_r2": float(r2_score(y_test, test_pred)),
        "dataset_size": int(len(df)),
        "trained_at": datetime.now(timezone.utc).isoformat(),
        "target": "Item_Outlet_Sales",
        "feature_count": len(get_feature_columns()[0]) + len(get_feature_columns()[1]),
        "selected_params": {
            "n_estimators": int(model.get_params().get("n_estimators", 0)),
            "max_depth": int(model.get_params().get("max_depth", 0)),
            "learning_rate": float(model.get_params().get("learning_rate", 0.0)),
            "subsample": float(model.get_params().get("subsample", 0.0)),
            "colsample_bytree": float(model.get_params().get("colsample_bytree", 0.0)),
            "min_child_weight": float(model.get_params().get("min_child_weight", 0.0)),
        },
    }

    cv_model = XGBRegressor(
        n_estimators=metrics["selected_params"]["n_estimators"],
        max_depth=metrics["selected_params"]["max_depth"],
        learning_rate=metrics["selected_params"]["learning_rate"],
        subsample=metrics["selected_params"]["subsample"],
        colsample_bytree=metrics["selected_params"]["colsample_bytree"],
        min_child_weight=metrics["selected_params"]["min_child_weight"],
        objective="reg:squarederror",
        random_state=42,
        n_jobs=-1,
    )
    cv_pipeline = Pipeline(
        steps=[
            ("preprocessor", create_preprocessor()),
            ("model", cv_model),
        ]
    )
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    cv_res = cross_validate(
        cv_pipeline,
        X,
        y,
        cv=cv,
        scoring={
            "rmse": "neg_root_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        n_jobs=-1,
    )
    cv_metrics = {
        "cv_rmse_mean": float(-cv_res["test_rmse"].mean()),
        "cv_mae_mean": float(-cv_res["test_mae"].mean()),
        "cv_r2_mean": float(cv_res["test_r2"].mean()),
        "cv_rmse_std": float(cv_res["test_rmse"].std()),
        "cv_mae_std": float(cv_res["test_mae"].std()),
        "cv_r2_std": float(cv_res["test_r2"].std()),
    }

    feature_stats = {
        "numeric_bounds": compute_feature_stats(X_train),
        "computed_from": "training_split_only",
    }

    test_predictions = pd.DataFrame(
        {
            "actual": y_test.values,
            "predicted": test_pred,
        }
    )

    joblib.dump(model, MODEL_PATH)
    joblib.dump(preprocessor, PREPROCESSOR_PATH)
    joblib.dump(metrics, METRICS_PKL_PATH)

    TEST_METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    CV_METRICS_PATH.write_text(json.dumps(cv_metrics, indent=2), encoding="utf-8")
    FEATURE_STATS_PATH.write_text(json.dumps(feature_stats, indent=2), encoding="utf-8")
    test_predictions.to_csv(TEST_PREDICTIONS_PATH, index=False)

    print("Training complete.")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved preprocessor: {PREPROCESSOR_PATH}")
    print(f"Saved metrics: {TEST_METRICS_PATH}")
    print(f"Saved CV metrics: {CV_METRICS_PATH}")
    print(f"Saved feature stats: {FEATURE_STATS_PATH}")
    print(f"Saved test predictions: {TEST_PREDICTIONS_PATH}")


if __name__ == "__main__":
    main()
