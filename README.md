# RetailVista: Supermarket Sales Prediction

**Author:** Leslie Fernando

RetailVista is a machine learning application that predicts supermarket sales (`Item_Outlet_Sales`) based on product and outlet characteristics. This project demonstrates a complete ML workflow including data preprocessing, model training with proper evaluation, and a simple web interface for predictions.

## Target Variable

**Item_Outlet_Sales** - The sales amount (in dollars) for each product at a specific outlet.

This is a regression problem predicting continuous sales values based on product features (weight, fat content, visibility, type, MRP) and outlet features (establishment year, size, location type, outlet type).

## Project Structure

```
RetailVista/
├── data/
│   └── supermarket_sales.csv       # Dataset (8,523 records, 13 columns)
├── model/
│   ├── model.pkl                   # Trained XGBoost model
│   ├── preprocessor.pkl            # Fitted preprocessing pipeline
│   └── metrics.pkl                 # Evaluation metrics
├── notebooks/
│   └── train_model.ipynb           # Model training notebook
├── preprocess.py                   # Shared preprocessing module
├── app.py                          # Streamlit web application
└── requirements.txt                # Python dependencies
```

## Features Used

The model uses **8 features** (excludes ID columns and target variable to prevent data leakage):

**Product Features (5):**
- `Item_Weight` - Weight of the product in kg
- `Item_Fat_Content` - Whether the product is low fat or regular
- `Item_Visibility` - Display area allocated to product (0-1)
- `Item_Type` - Category (Dairy, Soft Drinks, Meat, etc.)
- `Item_MRP` - Maximum Retail Price

**Outlet Features (4):**
- `Outlet_Establishment_Year` - Year the outlet was established
- `Outlet_Size` - Size of the outlet (Small/Medium/High)
- `Outlet_Location_Type` - Location tier (Tier 1/2/3)
- `Outlet_Type` - Type of outlet (Supermarket Type1/2/3, Grocery Store)

## Model Performance

**Algorithm:** XGBoost Regressor (100 estimators, max_depth=5, learning_rate=0.1)

**Evaluation Metrics (Test Set, 20% holdout):**
- **RMSE:** $1,047.60
- **MAE:** $729.87
- **R²:** 0.5962

**Preprocessing:**
- Numeric features: `StandardScaler`
- Categorical features: `OneHotEncoder` with unknown category handling
- Missing values: Filled with median (numeric features only)

**Data Split:**
- Training: 6,818 samples (80%)
- Test: 1,705 samples (20%)
- Random state: 42 (for reproducibility)

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train Model (Optional - model already trained)

Open and run the training notebook:

```bash
jupyter notebook notebooks/train_model.ipynb
```

This will:
- Load and preprocess data
- Train XGBoost model with proper train/test split
- Evaluate with RMSE, MAE, and R²
- Save model, preprocessor, and metrics to `model/` directory

### 3. Run Web Application

```bash
streamlit run app.py
```

The app has three tabs:
- **Predictions:** Single product prediction or batch CSV upload
- **Analytics:** Model performance metrics and visualizations
- **About:** Dataset and model information

## Dataset

**Source:** Supermarket sales dataset  
**Size:** 8,523 records  
**Columns:** 13 (including target variable and IDs)

The dataset contains sales information for various products across different supermarket outlets, including product characteristics and outlet attributes.

## Key Features

✅ **No Data Leakage** - Target variable and ID columns properly excluded from features  
✅ **Consistent Preprocessing** - Shared module ensures training/inference consistency  
✅ **Proper Evaluation** - Train/test split with multiple metrics reported  
✅ **Simple Interface** - Clean 3-tab Streamlit app without unnecessary complexity  
✅ **Reproducible** - Fixed random seeds and saved artifacts  

## Limitations

- **Historical Data Only** - Model trained on static dataset from specific time period
- **No Temporal Effects** - Does not account for seasonality or trends over time
- **No External Factors** - Missing information about promotions, competitors, weather, etc.
- **Point Estimates** - No confidence intervals or uncertainty quantification
- **Missing Values** - Simple median imputation may not be optimal
- **Feature Set** - Limited to available columns; no derived features or interactions
- **Single Model** - No ensemble or model comparison performed

## Future Work

Potential improvements for enhanced performance and functionality:

**Model Improvements:**
- Ensemble methods (stacking multiple models)
- Hyperparameter tuning with cross-validation
- Feature engineering (interactions, polynomial features)
- Time series analysis for seasonal patterns
- Confidence interval estimation

**Data Enhancements:**
- Additional external data (demographics, competitor info)
- Promotion and discount information
- Historical sales trends
- Weather and event data

**Application Features:**
- Model retraining pipeline
- A/B testing framework
- Feature importance dashboard
- Prediction explanations (SHAP values)
- API endpoint for programmatic access

**Infrastructure:**
- Automated data quality checks
- Model monitoring and drift detection
- Continuous integration/deployment
- Containerization with Docker

## Technical Stack

- **Language:** Python 3.8+
- **ML Framework:** XGBoost, scikit-learn
- **Web Framework:** Streamlit
- **Data Processing:** pandas, numpy
- **Visualization:** plotly, matplotlib, seaborn
- **Model Persistence:** joblib

## Repository

GitHub: [github.com/lesliefdo08/RetailVista](https://github.com/lesliefdo08/RetailVista)

## License

This project is for educational and demonstration purposes.

---

**Note:** This is an ML application demo showcasing a complete workflow from data preprocessing to model deployment. It is not production-ready and should not be used for actual business decisions without proper validation and monitoring.
