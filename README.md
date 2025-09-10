
# RetailVista: Supermarket Sales Prediction

**Author:** Leslie Fernando

RetailVista is a machine learning project to predict supermarket sales using open datasets and Python. This project includes data preprocessing, feature engineering, model training, evaluation, and a Streamlit web app for predictions.

## Project Structure

- `data/` — Place your downloaded dataset here (e.g., `supermarket_sales.csv`).
- `model/` — Trained model files will be saved here.
- `notebooks/` — Jupyter notebooks for EDA and modeling.
- `app.py` — Streamlit web application.
- `requirements.txt` — Python dependencies.

## Quick Start

1. Download a supermarket sales dataset (e.g., from Kaggle) and place it in `data/`.
2. Open and run the notebook in `notebooks/eda_and_model.ipynb` to preprocess data, train, and save the model.
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Main Libraries Used
- pandas, numpy, matplotlib, seaborn
- scikit-learn, xgboost
- streamlit, joblib

## Deployment
- Deploy locally or on Streamlit Community Cloud (see Streamlit docs for details).
