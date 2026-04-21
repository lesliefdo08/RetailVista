from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "supermarket_sales.csv"
MODEL_DIR = BASE_DIR / "model"
ARTIFACTS_DIR = BASE_DIR / "artifacts"

MODEL_PATH = MODEL_DIR / "model.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "preprocessor.pkl"
METRICS_PKL_PATH = MODEL_DIR / "metrics.pkl"

TEST_METRICS_PATH = ARTIFACTS_DIR / "test_metrics.json"
TEST_PREDICTIONS_PATH = ARTIFACTS_DIR / "test_predictions.csv"
FEATURE_STATS_PATH = ARTIFACTS_DIR / "feature_stats.json"
CV_METRICS_PATH = ARTIFACTS_DIR / "cv_metrics.json"

USD_TO_INR = 83.0
MAX_UPLOAD_ROWS = 50000
REQUIRED_COLUMNS = [
    "Item_Weight",
    "Item_Fat_Content",
    "Item_Visibility",
    "Item_Type",
    "Item_MRP",
    "Outlet_Establishment_Year",
    "Outlet_Size",
    "Outlet_Location_Type",
    "Outlet_Type",
]
