"""
RetailVista - Supermarket Sales Prediction Application
A simple ML application for predicting Item_Outlet_Sales using product and outlet features.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from preprocess import prepare_features, get_feature_columns

# Page configuration
st.set_page_config(
    page_title="RetailVista - Sales Prediction",
    page_icon="📊",
    layout="wide"
)

# Load model, preprocessor, and metrics
@st.cache_resource
def load_model_artifacts():
    """Load trained model, preprocessor, and evaluation metrics."""
    model = joblib.load('model/model.pkl')
    preprocessor = joblib.load('model/preprocessor.pkl')
    metrics = joblib.load('model/metrics.pkl')
    return model, preprocessor, metrics

@st.cache_data
def load_data():
    """Load the dataset."""
    return pd.read_csv('data/supermarket_sales.csv')

# Initialize
try:
    model, preprocessor, metrics = load_model_artifacts()
    df = load_data()
    numeric_features, categorical_features = get_feature_columns()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Title
st.title("📊 RetailVista")
st.markdown("Predict supermarket sales based on product and outlet characteristics")

# Tabs
tab1, tab2, tab3 = st.tabs(["Predictions", "Analytics", "About"])

# TAB 1: PREDICTIONS
with tab1:
    st.header("Sales Prediction")
    
    # Input method selection
    input_method = st.radio("Input Method:", ["Single Product", "CSV Upload"], horizontal=True)
    
    if input_method == "Single Product":
        st.subheader("Enter Product Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Product Information**")
            item_weight = st.number_input("Item Weight (kg)", min_value=0.0, max_value=50.0, value=10.0)
            item_fat_content = st.selectbox("Item Fat Content", df['Item_Fat_Content'].unique())
            item_visibility = st.number_input("Item Visibility", min_value=0.0, max_value=1.0, value=0.05, step=0.01)
            item_type = st.selectbox("Item Type", sorted(df['Item_Type'].unique()))
            item_mrp = st.number_input("Item MRP ($)", min_value=0.0, max_value=500.0, value=100.0)
        
        with col2:
            st.markdown("**Outlet Information**")
            outlet_establishment_year = st.number_input("Outlet Establishment Year", 
                                                       min_value=1980, max_value=2025, value=2000)
            outlet_size = st.selectbox("Outlet Size", df['Outlet_Size'].dropna().unique())
            outlet_location_type = st.selectbox("Outlet Location Type", df['Outlet_Location_Type'].unique())
            outlet_type = st.selectbox("Outlet Type", df['Outlet_Type'].unique())
        
        if st.button("Predict Sales", type="primary"):
            # Create input dataframe
            input_data = pd.DataFrame({
                'Item_Weight': [item_weight],
                'Item_Fat_Content': [item_fat_content],
                'Item_Visibility': [item_visibility],
                'Item_Type': [item_type],
                'Item_MRP': [item_mrp],
                'Outlet_Establishment_Year': [outlet_establishment_year],
                'Outlet_Size': [outlet_size],
                'Outlet_Location_Type': [outlet_location_type],
                'Outlet_Type': [outlet_type]
            })
            
            # Prepare and predict
            X = prepare_features(input_data)
            X_processed = preprocessor.transform(X)
            prediction = model.predict(X_processed)[0]
            
            # Display result
            st.success("Prediction Complete")
            st.metric(label="Predicted Sales", value=f"${prediction:,.2f}")
    
    else:  # CSV Upload
        st.subheader("Batch Prediction from CSV")
        st.markdown("Upload a CSV file with the following columns:")
        st.code(", ".join(numeric_features + categorical_features))
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                st.write(f"Loaded {len(batch_df)} rows")
                st.dataframe(batch_df.head())
                
                if st.button("Predict Batch", type="primary"):
                    # Prepare and predict
                    X = prepare_features(batch_df)
                    X_processed = preprocessor.transform(X)
                    predictions = model.predict(X_processed)
                    
                    # Add predictions to dataframe
                    result_df = batch_df.copy()
                    result_df['Predicted_Sales'] = predictions
                    
                    st.success("Batch Prediction Complete")
                    st.dataframe(result_df)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="Download Results as CSV",
                        data=csv,
                        file_name="predictions.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")

# TAB 2: ANALYTICS
with tab2:
    st.header("Model Analytics")
    
    # Display evaluation metrics
    st.subheader("Model Performance Metrics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Test RMSE", f"${metrics['test_rmse']:,.2f}")
    with col2:
        st.metric("Test MAE", f"${metrics['test_mae']:,.2f}")
    with col3:
        st.metric("Test R²", f"{metrics['test_r2']:.4f}")
    
    st.markdown("---")
    
    # Generate predictions on sample data for visualization
    st.subheader("Actual vs Predicted Sales")
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    X_sample = prepare_features(sample_df)
    X_sample_processed = preprocessor.transform(X_sample)
    y_pred = model.predict(X_sample_processed)
    y_actual = sample_df['Item_Outlet_Sales'].values
    
    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_actual,
        y=y_pred,
        mode='markers',
        marker=dict(color='#1f77b4', size=6, opacity=0.6),
        name='Predictions'
    ))
    
    # Add perfect prediction line
    min_val = min(y_actual.min(), y_pred.min())
    max_val = max(y_actual.max(), y_pred.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash'),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        title="Actual vs Predicted Sales (Sample)",
        xaxis_title="Actual Sales ($)",
        yaxis_title="Predicted Sales ($)",
        height=500,
        hovermode='closest'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Residuals histogram
    st.subheader("Residuals Distribution")
    residuals = y_actual - y_pred
    
    fig2 = go.Figure()
    fig2.add_trace(go.Histogram(
        x=residuals,
        nbinsx=50,
        marker_color='#1f77b4',
        name='Residuals'
    ))
    
    fig2.update_layout(
        title="Distribution of Prediction Errors",
        xaxis_title="Residual (Actual - Predicted)",
        yaxis_title="Frequency",
        height=400
    )
    
    st.plotly_chart(fig2, use_container_width=True)

# TAB 3: ABOUT
with tab3:
    st.header("About RetailVista")
    
    st.markdown("""
    ### Dataset
    
    This application uses a supermarket sales dataset containing information about products and outlets.
    
    **Dataset Size:** 8,523 records
    
    **Target Variable:** `Item_Outlet_Sales` - the sales amount for each item at a specific outlet
    
    ### Features Used
    
    The model uses the following features to make predictions:
    
    **Product Features:**
    - Item Weight (kg)
    - Item Fat Content (Low Fat / Regular)
    - Item Visibility (0-1 scale)
    - Item Type (16 categories including Dairy, Soft Drinks, Meat, etc.)
    - Item MRP (Maximum Retail Price in dollars)
    
    **Outlet Features:**
    - Outlet Establishment Year
    - Outlet Size (Small / Medium / High)
    - Outlet Location Type (Tier 1 / Tier 2 / Tier 3)
    - Outlet Type (Supermarket Type1/2/3 / Grocery Store)
    
    ### Model
    
    **Algorithm:** XGBoost Regressor
    
    **Preprocessing:**
    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder
    
    **Performance (Test Set):**
    - RMSE: ${:,.2f}
    - MAE: ${:,.2f}
    - R²: {:.4f}
    
    ### Limitations
    
    - Model trained on historical data from a specific time period
    - Performance may vary for products or outlets significantly different from training data
    - Does not account for seasonal variations or promotional effects
    - Missing values in Item_Weight are filled with median values
    - Predictions are point estimates without confidence intervals
    
    ### Future Work
    
    Potential improvements could include:
    - Time series analysis for seasonal patterns
    - Additional features (promotions, competitor data, demographics)
    - Advanced models (LightGBM, Neural Networks)
    - Confidence intervals for predictions
    - Model retraining pipeline with new data
    
    ### Technical Details
    
    - Built with Streamlit and Python
    - Model training performed offline in Jupyter notebooks
    - Shared preprocessing ensures consistency between training and inference
    - Feature engineering limited to avoid data leakage
    
    ---
    
    **Author:** Leslie Fernando  
    **Project:** RetailVista - ML Application Demo  
    **Repository:** [github.com/lesliefdo08/RetailVista](https://github.com/lesliefdo08/RetailVista)
    """.format(metrics['test_rmse'], metrics['test_mae'], metrics['test_r2']))

# Footer
st.markdown("---")
st.markdown("*RetailVista - A simple ML application for supermarket sales prediction*")
