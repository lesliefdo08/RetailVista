import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load the trained model and features
@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load('model/sales_predictor.joblib')
        if isinstance(model_data, dict):
            return model_data['model'], model_data['features'], model_data['target_name']
        else:
            # Old format - just the model
            return model_data, None, 'sales'
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

model, feature_names, target_name = load_model_data()

if model is None:
    st.error("❌ Could not load the trained model. Please train and save the model first.")
    st.stop()

st.set_page_config(page_title='RetailVista: Sales Prediction', layout='wide')
st.title("🛒 RetailVista: Sales Predictor")
st.write("**Author: Leslie Fernando**")
st.write("Upload a CSV file to predict sales values!")

# Show model info
if feature_names:
    st.sidebar.write("**Model Information:**")
    st.sidebar.write(f"Target: {target_name}")
    st.sidebar.write(f"Features ({len(feature_names)}):")
    for i, feature in enumerate(feature_names, 1):
        st.sidebar.write(f"{i}. {feature}")

# File upload
uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    # Read the uploaded file
    df = pd.read_csv(uploaded_file)
    
    st.write("### Data Preview:")
    st.write(df.head())
    st.write(f"Shape: {df.shape}")
    
    # Preprocess the data exactly like in training
    st.write("### Preprocessing...")
    df_processed = df.copy()
    
    # Remove missing values
    original_rows = len(df_processed)
    df_processed = df_processed.dropna()
    if len(df_processed) < original_rows:
        st.write(f"Removed {original_rows - len(df_processed)} rows with missing values")
    
    # Encode categorical variables
    categorical_cols = df_processed.select_dtypes(include='object').columns
    if len(categorical_cols) > 0:
        st.write(f"Encoding categorical columns: {list(categorical_cols)}")
        for col in categorical_cols:
            df_processed[col] = df_processed[col].astype('category').cat.codes
    
    # Make predictions
    if st.button("🔮 Make Predictions", type="primary"):
        try:
            # Prepare features for prediction
            if feature_names:
                # Use saved feature names
                missing_features = [f for f in feature_names if f not in df_processed.columns]
                if missing_features:
                    st.error(f"Missing required features: {missing_features}")
                    st.write("**Available columns:**", list(df_processed.columns))
                    st.write("**Required features:**", feature_names)
                    st.stop()
                
                X_pred = df_processed[feature_names]
            else:
                # Fallback: use all columns except target
                if target_name in df_processed.columns:
                    X_pred = df_processed.drop(columns=[target_name])
                else:
                    X_pred = df_processed
            
            st.write(f"Making predictions using {X_pred.shape[1]} features...")
            
            # Make predictions
            predictions = model.predict(X_pred)
            
            # Add predictions to the original dataframe
            df_with_predictions = df.copy()
            df_with_predictions['Predicted_Value'] = predictions
            
            st.write("### 🎉 Predictions Complete!")
            st.write(df_with_predictions)
            
            # Show statistics
            st.write("### 📊 Prediction Statistics:")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Mean Prediction", f"{predictions.mean():.2f}")
            with col2:
                st.metric("Max Prediction", f"{predictions.max():.2f}")
            with col3:
                st.metric("Min Prediction", f"{predictions.min():.2f}")
            
            # Visualization
            st.write("### 📈 Prediction Distribution:")
            st.line_chart(predictions)
            
            # Download button for results
            csv = df_with_predictions.to_csv(index=False)
            st.download_button(
                label="📥 Download Predictions (CSV)",
                data=csv,
                file_name='sales_predictions.csv',
                mime='text/csv'
            )
            
        except Exception as e:
            st.error(f"Error making predictions: {str(e)}")
            st.write("**Debug Info:**")
            st.write(f"- Input shape: {df_processed.shape}")
            st.write(f"- Available columns: {list(df_processed.columns)}")
            if feature_names:
                st.write(f"- Expected features: {feature_names}")

else:
    st.info("👆 Please upload a CSV file to get started!")
    
    # Show example format
    st.write("### 📋 Expected Data Format:")
    if feature_names:
        st.write("Your CSV should contain these columns:")
        example_data = {feature: [0] * 3 for feature in feature_names}
        example_df = pd.DataFrame(example_data)
        st.dataframe(example_df)
    else:
        st.write("Upload your data file and the app will guide you through the process.")
