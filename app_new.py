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

st.set_page_config(page_title='RetailVista: Smart Sales Predictor', layout='wide')

st.title("🛒 RetailVista: Smart Sales Predictor")
st.write("**Author: Leslie Fernando**")

st.markdown("""
### 🎯 **What does this app do?**
This app helps **store owners, managers, and business people** predict how much a product will sell based on:
- Product details (weight, type, brand)
- Store information (location, size, type)
- Market factors

### 💡 **Why is this useful?**
- **Inventory Planning:** Know how much stock to order
- **Pricing Strategy:** Set optimal prices for maximum sales  
- **Business Decisions:** Decide which products to focus on
- **Revenue Forecasting:** Predict monthly/yearly sales

---
""")

# Create tabs for different ways to use the app
tab1, tab2, tab3 = st.tabs(["🔮 Quick Predict", "📊 Batch Upload", "📚 Learn More"])

with tab1:
    st.header("🔮 Predict Sales for One Product")
    st.write("**No CSV needed! Just fill in the details below:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📦 Product Information")
        item_weight = st.number_input("Product Weight (kg)", min_value=0.1, max_value=50.0, value=10.0)
        item_fat_content = st.selectbox("Fat Content", ["Low Fat", "Regular"])
        item_visibility = st.slider("Product Visibility in Store", 0.0, 1.0, 0.1)
        item_type = st.selectbox("Product Category", [
            "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
            "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
            "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
            "Breads", "Starchy Foods", "Others", "Seafood"
        ])
    
    with col2:
        st.subheader("🏪 Store Information")
        outlet_size = st.selectbox("Store Size", ["Small", "Medium", "High"])
        outlet_location = st.selectbox("Store Location", ["Tier 1", "Tier 2", "Tier 3"])
        outlet_type = st.selectbox("Store Type", [
            "Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"
        ])
        establishment_year = st.slider("Store Established Year", 1985, 2020, 2000)
    
    if st.button("🚀 Predict Sales Now!", type="primary"):
        # Create prediction data
        prediction_data = {
            'Item_Weight': item_weight,
            'Item_Fat_Content': 0 if item_fat_content == "Low Fat" else 1,
            'Item_Visibility': item_visibility,
            'Item_Type': ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
                         "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                         "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                         "Breads", "Starchy Foods", "Others", "Seafood"].index(item_type),
            'Outlet_Size': ["Small", "Medium", "High"].index(outlet_size),
            'Outlet_Location_Type': ["Tier 1", "Tier 2", "Tier 3"].index(outlet_location),
            'Outlet_Type': ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"].index(outlet_type),
            'Outlet_Establishment_Year': establishment_year,
            # Add dummy values for other required features
            'Item_Identifier': 0,
            'Outlet_Identifier': 0,
            'Item_Outlet_Sales': 1000,  # Default value
            'Profit': 200  # Default value
        }
        
        try:
            # Create dataframe with correct feature order
            if feature_names:
                # Filter to only the features used in training
                filtered_data = {k: v for k, v in prediction_data.items() if k in feature_names}
                df_pred = pd.DataFrame([filtered_data])
                # Ensure all required features are present with default values
                for feature in feature_names:
                    if feature not in df_pred.columns:
                        df_pred[feature] = 0
                df_pred = df_pred[feature_names]  # Correct order
            else:
                df_pred = pd.DataFrame([prediction_data])
            
            prediction = model.predict(df_pred)[0]
            
            st.success(f"🎉 **Predicted Sales: ${prediction:.2f}**")
            
            # Give business insights
            if prediction > 2000:
                st.info("📈 **High Sales Expected!** This is a strong performer. Consider stocking more.")
            elif prediction > 1000:
                st.info("📊 **Moderate Sales Expected.** A steady seller for your inventory.")
            else:
                st.warning("📉 **Lower Sales Expected.** Consider promotions or review pricing.")
                
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            st.write("Debug info:", prediction_data)

with tab2:
    st.header("📊 Batch Predictions (Upload CSV)")
    st.write("**For businesses with lots of products - upload a CSV file:**")
    
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
        if st.button("🔮 Make Batch Predictions", type="primary"):
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
                df_with_predictions['Predicted_Sales'] = predictions
                
                st.write("### 🎉 Predictions Complete!")
                st.write(df_with_predictions)
                
                # Show statistics
                st.write("### 📊 Prediction Statistics:")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Average Sales", f"${predictions.mean():.2f}")
                with col2:
                    st.metric("Highest Sales", f"${predictions.max():.2f}")
                with col3:
                    st.metric("Lowest Sales", f"${predictions.min():.2f}")
                
                # Visualization
                st.write("### 📈 Sales Distribution:")
                st.line_chart(predictions)
                
                # Download button for results
                csv = df_with_predictions.to_csv(index=False)
                st.download_button(
                    label="📥 Download All Predictions (CSV)",
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
        st.info("👆 Upload a CSV file with your product data")
        
        # Show example format
        st.write("### 📋 CSV Format Example:")
        if feature_names:
            st.write("Your CSV should contain these columns:")
            example_data = {feature: [0] * 3 for feature in feature_names}
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)

with tab3:
    st.header("📚 Learn More About This App")
    
    st.markdown("""
    ### 🏪 **Who Can Use This?**
    - **Store Owners:** Plan inventory and pricing
    - **Product Managers:** Forecast demand for new products  
    - **Business Analysts:** Make data-driven decisions
    - **Students/Researchers:** Learn about sales prediction
    
    ### 🔍 **How It Works:**
    1. **Machine Learning Model:** Trained on real supermarket data
    2. **Input Features:** Product and store characteristics
    3. **Output:** Predicted sales amount in dollars
    4. **Accuracy:** Model learns patterns from historical sales data
    
    ### 📊 **What Affects Sales?**
    - **Product Weight:** Heavier items often sell differently
    - **Fat Content:** Health-conscious choices affect sales
    - **Store Location:** Tier 1 cities vs smaller towns
    - **Store Size:** Bigger stores = more visibility
    - **Product Category:** Some categories naturally sell more
    
    ### 💼 **Real Business Use Cases:**
    
    **Scenario 1: New Product Launch**
    - Input: New snack food details
    - Output: Expected sales forecast
    - Decision: How much to stock initially
    
    **Scenario 2: Seasonal Planning**  
    - Input: Holiday product mix
    - Output: Sales predictions for each item
    - Decision: Optimize shelf space allocation
    
    **Scenario 3: Pricing Strategy**
    - Input: Same product in different store types
    - Output: Compare expected sales
    - Decision: Set location-specific prices
    
    ### 🎯 **Tips for Best Results:**
    1. Use realistic product weights (0.1kg - 50kg)
    2. Choose appropriate store size for your location
    3. Consider your target market (Tier 1/2/3 cities)
    4. Test multiple scenarios to compare options
    
    ### ⚠️ **Important Notes:**
    - Predictions are estimates based on historical data
    - Always combine with business judgment
    - Consider market conditions and seasonality
    - Use as one input in your decision-making process
    """)

# Show model info in sidebar
if feature_names:
    st.sidebar.write("**🤖 Model Information:**")
    st.sidebar.write(f"Target: {target_name}")
    st.sidebar.write(f"Features: {len(feature_names)}")
    st.sidebar.write(f"Model: XGBoost Regressor")
    
    with st.sidebar.expander("View All Features"):
        for i, feature in enumerate(feature_names, 1):
            st.write(f"{i}. {feature}")
