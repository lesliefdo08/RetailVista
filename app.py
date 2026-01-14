"""
RetailVista - Retail Sales Prediction & Decision Support
A user-friendly tool for estimating product sales and getting business insights.
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
from preprocess import prepare_features, get_feature_columns

# Constants
USD_TO_INR = 83.0  # Approximate conversion rate for demonstration

# Page configuration
st.set_page_config(
    page_title="RetailVista - Sales Estimator",
    page_icon="🛒",
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

def generate_business_insights(input_data, prediction_usd):
    """Generate actionable business suggestions based on input parameters."""
    insights = []
    
    # Extract input values
    visibility = input_data['Item_Visibility'].iloc[0]
    mrp = input_data['Item_MRP'].iloc[0] * USD_TO_INR  # Convert to INR for comparison
    location_type = input_data['Outlet_Location_Type'].iloc[0]
    outlet_type = input_data['Outlet_Type'].iloc[0]
    item_type = input_data['Item_Type'].iloc[0]
    fat_content = input_data['Item_Fat_Content'].iloc[0]
    
    # Insight 1: Visibility-based suggestion
    if visibility < 0.05:
        insights.append({
            'icon': '👁️',
            'title': 'Improve Product Visibility',
            'suggestion': 'This product has low shelf visibility. Consider placing it at eye level or near checkout counters to increase sales by 15-25%.'
        })
    elif visibility < 0.08:
        insights.append({
            'icon': '📍',
            'title': 'Good Visibility',
            'suggestion': 'Product placement is decent. You could experiment with end-cap displays or promotional signage for additional boost.'
        })
    
    # Insight 2: Location-based suggestion
    if location_type == 'Tier 3':
        insights.append({
            'icon': '🏘️',
            'title': 'Small Town Marketing',
            'suggestion': 'In smaller towns, word-of-mouth and local festivals work well. Consider community-focused promotions and bundle offers.'
        })
    elif location_type == 'Tier 1':
        insights.append({
            'icon': '🏙️',
            'title': 'Metro Strategy',
            'suggestion': 'In metro areas, focus on convenience and premium positioning. Online ordering integration can increase reach.'
        })
    
    # Insight 3: Pricing-based suggestion
    if mrp > 200:
        insights.append({
            'icon': '💰',
            'title': 'Premium Product Strategy',
            'suggestion': 'High-priced items benefit from quality emphasis and targeted promotions. Consider loyalty discounts for repeat buyers.'
        })
    elif mrp < 50:
        insights.append({
            'icon': '🎯',
            'title': 'Value Product Opportunity',
            'suggestion': 'Low-priced items sell through volume. Multi-buy offers (e.g., "Buy 3 Get 1 Free") can significantly boost sales.'
        })
    
    # Insight 4: Store format suggestion
    if outlet_type == 'Grocery Store' and mrp > 100:
        insights.append({
            'icon': '🏪',
            'title': 'Format Consideration',
            'suggestion': 'Premium items typically perform better in larger supermarkets. Consider selective distribution or special orders.'
        })
    
    # Insight 5: Product category specific
    if item_type in ['Dairy', 'Fruits and Vegetables']:
        insights.append({
            'icon': '🥬',
            'title': 'Freshness Matters',
            'suggestion': 'Perishable items need frequent restocking and prominent "fresh today" signage to drive sales.'
        })
    elif item_type in ['Soft Drinks', 'Snack Foods']:
        insights.append({
            'icon': '🥤',
            'title': 'Impulse Purchase Zone',
            'suggestion': 'Snacks and beverages perform well near checkout. Create combo offers with complementary products.'
        })
    
    # Limit to top 3 most relevant insights
    return insights[:3]

def map_user_friendly_to_dataset(user_value, field_name):
    """Map user-friendly labels back to dataset values."""
    mappings = {
        'visibility': {
            'Low (Hard to Spot)': 0.03,
            'Medium (Noticeable)': 0.08,
            'High (Very Prominent)': 0.15
        },
        'city_type': {
            'Metro City': 'Tier 1',
            'Developing City': 'Tier 2',
            'Small Town': 'Tier 3'
        },
        'store_format': {
            'Small Grocery Store': 'Grocery Store',
            'Supermarket Type 1': 'Supermarket Type1',
            'Supermarket Type 2': 'Supermarket Type2',
            'Supermarket Type 3': 'Supermarket Type3'
        }
    }
    
    if field_name in mappings:
        return mappings[field_name].get(user_value, user_value)
    return user_value

def explain_prediction(model, preprocessor, input_data):
    """Explain why the model made this prediction (top 3 factors)."""
    # Get feature importance from model
    feature_importance = model.feature_importances_
    
    # Get feature names after preprocessing
    numeric_features, categorical_features = get_feature_columns()
    feature_names = numeric_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
    
    # Create importance dataframe
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)
    
    # Map technical features to user-friendly explanations
    feature_explanations = {
        'Item_MRP': ('Product Price', 'Higher priced items generally have higher sales value'),
        'Item_Visibility': ('Product Visibility', 'Better placement leads to more customer attention'),
        'Outlet_Type_Grocery Store': ('Store Type: Grocery', 'Small grocery stores have different sales patterns'),
        'Outlet_Type_Supermarket Type1': ('Store Type: Supermarket', 'Larger stores attract more customers'),
        'Outlet_Type_Supermarket Type2': ('Store Type: Supermarket', 'Larger stores attract more customers'),
        'Outlet_Type_Supermarket Type3': ('Store Type: Supermarket', 'Larger stores attract more customers'),
        'Outlet_Establishment_Year': ('Store Age', 'Older stores have established customer base'),
        'Item_Weight': ('Product Weight', 'Heavier items may affect handling and sales'),
        'Outlet_Size_Medium': ('Store Size: Medium', 'Medium stores balance variety and convenience'),
        'Outlet_Size_High': ('Store Size: Large', 'Larger stores offer more variety'),
        'Outlet_Location_Type_Tier 1': ('Location: Metro City', 'Urban areas have higher foot traffic'),
        'Outlet_Location_Type_Tier 2': ('Location: Developing City', 'Growing cities offer good potential'),
        'Outlet_Location_Type_Tier 3': ('Location: Small Town', 'Rural areas have different buying patterns'),
    }
    
    # Get top 3 factors
    top_factors = []
    for idx, row in importance_df.head(3).iterrows():
        feature = row['feature']
        importance = row['importance']
        
        # Get user-friendly explanation
        if feature in feature_explanations:
            friendly_name, explanation = feature_explanations[feature]
        else:
            # Handle item type and other categorical features
            if feature.startswith('Item_Type_'):
                item_type = feature.replace('Item_Type_', '')
                friendly_name = f'Product Category: {item_type}'
                explanation = f'{item_type} products have specific sales patterns'
            elif feature.startswith('Item_Fat_Content_'):
                friendly_name = 'Fat Content'
                explanation = 'Low-fat vs regular affects health-conscious buyers'
            else:
                friendly_name = feature.replace('_', ' ').title()
                explanation = 'Influences sales patterns'
        
        top_factors.append({
            'name': friendly_name,
            'explanation': explanation,
            'importance': importance
        })
    
    return top_factors

def calculate_confidence_level(input_data, df):
    """Calculate prediction confidence based on whether inputs are typical."""
    confidence_scores = []
    
    # Check numeric features
    for col in ['Item_Weight', 'Item_Visibility', 'Item_MRP']:
        if col in input_data.columns and col in df.columns:
            value = input_data[col].iloc[0]
            q1 = df[col].quantile(0.25)
            q3 = df[col].quantile(0.75)
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            
            if lower_bound <= value <= upper_bound:
                confidence_scores.append(1.0)  # Within normal range
            else:
                confidence_scores.append(0.5)  # Outlier
    
    # Average confidence
    avg_confidence = np.mean(confidence_scores) if confidence_scores else 0.8
    
    if avg_confidence >= 0.9:
        return 'High', '🟢'
    elif avg_confidence >= 0.7:
        return 'Medium', '🟡'
    else:
        return 'Low', '🟠'

def simulate_scenarios(model, preprocessor, input_data, original_prediction):
    """Simulate what-if scenarios for key improvements."""
    scenarios = []
    
    # Scenario 1: Improve visibility
    visibility_levels = {
        'Low (Hard to Spot)': 0.03,
        'Medium (Noticeable)': 0.08,
        'High (Very Prominent)': 0.15
    }
    
    current_visibility = input_data['Item_Visibility'].iloc[0]
    
    for level_name, level_value in visibility_levels.items():
        if level_value > current_visibility:
            scenario_data = input_data.copy()
            scenario_data['Item_Visibility'] = level_value
            
            X = prepare_features(scenario_data)
            X_processed = preprocessor.transform(X)
            new_prediction = model.predict(X_processed)[0]
            
            improvement_pct = ((new_prediction - original_prediction) / original_prediction) * 100
            
            scenarios.append({
                'change': f'Increase visibility to {level_name}',
                'original_sales': original_prediction * USD_TO_INR,
                'new_sales': new_prediction * USD_TO_INR,
                'improvement_pct': improvement_pct,
                'type': 'visibility'
            })
    
    # Scenario 2: Store size upgrade
    current_size = input_data['Outlet_Size'].iloc[0]
    size_order = ['Small', 'Medium', 'High']
    
    if current_size in size_order:
        current_idx = size_order.index(current_size)
        if current_idx < len(size_order) - 1:
            next_size = size_order[current_idx + 1]
            
            scenario_data = input_data.copy()
            scenario_data['Outlet_Size'] = next_size
            
            X = prepare_features(scenario_data)
            X_processed = preprocessor.transform(X)
            new_prediction = model.predict(X_processed)[0]
            
            improvement_pct = ((new_prediction - original_prediction) / original_prediction) * 100
            
            scenarios.append({
                'change': f'Upgrade store size to {next_size}',
                'original_sales': original_prediction * USD_TO_INR,
                'new_sales': new_prediction * USD_TO_INR,
                'improvement_pct': improvement_pct,
                'type': 'store_size'
            })
    
    return scenarios

def calculate_benchmarks(df, input_data, prediction_usd):
    """Calculate category benchmarks for comparison."""
    benchmarks = []
    
    # Benchmark 1: Item Type average
    item_type = input_data['Item_Type'].iloc[0]
    category_avg = df[df['Item_Type'] == item_type]['Item_Outlet_Sales'].mean()
    
    if category_avg > 0:
        diff_pct = ((prediction_usd - category_avg) / category_avg) * 100
        status = 'Above Average' if diff_pct > 0 else 'Below Average'
        
        benchmarks.append({
            'category': f'{item_type} Products',
            'avg_sales': category_avg * USD_TO_INR,
            'your_estimate': prediction_usd * USD_TO_INR,
            'difference_pct': abs(diff_pct),
            'status': status
        })
    
    # Benchmark 2: Location Type average
    location = input_data['Outlet_Location_Type'].iloc[0]
    location_avg = df[df['Outlet_Location_Type'] == location]['Item_Outlet_Sales'].mean()
    
    if location_avg > 0:
        diff_pct = ((prediction_usd - location_avg) / location_avg) * 100
        status = 'Above Average' if diff_pct > 0 else 'Below Average'
        
        location_friendly = {'Tier 1': 'Metro Cities', 'Tier 2': 'Developing Cities', 'Tier 3': 'Small Towns'}
        
        benchmarks.append({
            'category': location_friendly.get(location, location),
            'avg_sales': location_avg * USD_TO_INR,
            'your_estimate': prediction_usd * USD_TO_INR,
            'difference_pct': abs(diff_pct),
            'status': status
        })
    
    return benchmarks

# Initialize
try:
    model, preprocessor, metrics = load_model_artifacts()
    df = load_data()
    numeric_features, categorical_features = get_feature_columns()
except Exception as e:
    st.error(f"Error loading model or data: {e}")
    st.stop()

# Header with welcome message
st.title("🛒 RetailVista")
st.markdown("### Your Retail Sales Estimator & Business Advisor")

# Welcome section
with st.container():
    st.info("""
    **Welcome!** RetailVista helps you estimate how much a product might sell in your store, 
    and gives you practical tips to improve those sales.
    
    **What you'll get:**
    - An estimated monthly sales figure based on your product and store details
    - Personalized suggestions to boost sales
    - Understanding of what factors drive retail performance
    
    **Who is this for?**  
    Retail store owners, business students, market analysts, and anyone curious about retail dynamics.
    """)

# Tabs
tab1, tab2, tab3 = st.tabs(["💡 Get Sales Estimate", "📊 Model Performance", "ℹ️ About This Tool"])

# TAB 1: PREDICTIONS
with tab1:
    # Input guidance section
    with st.expander("📖 Before You Start: Understanding the Inputs", expanded=False):
        st.markdown("""
        **Product Visibility** - How easy it is to spot your product in the store  
        *Higher visibility = More people see it = Better sales*  
        Example: Products at eye level or near checkout have high visibility
        
        **Product Weight** - Physical weight of the item in kilograms  
        *Affects logistics and customer carrying convenience*  
        Example: 1 liter milk carton ≈ 1 kg
        
        **Fat Content** - For food products, whether it's low-fat or regular  
        *Health-conscious areas prefer low-fat options*
        
        **Product Category** - What type of product (Dairy, Snacks, Beverages, etc.)  
        *Different categories have different sales patterns*
        
        **Maximum Retail Price (MRP)** - The price tag on the product  
        *Balance between profit margin and affordability*  
        Example: Premium products have higher MRP but may sell less volume
        
        **Store Size** - Physical size of your retail outlet  
        *Larger stores can stock more and attract more customers*
        
        **City Type** - The kind of location where your store operates  
        *Metro cities have different buying patterns than small towns*
        
        **Store Format** - Type of retail format  
        *Grocery stores vs large supermarkets serve different needs*
        
        **Store Age** - Year when the outlet was established  
        *Older stores have established customer base*
        """)
    
    st.header("Enter Your Product & Store Details")
    
    # Input method selection
    input_method = st.radio("How would you like to provide information?", 
                           ["Single Product (Manual Entry)", "Multiple Products (Upload File)"], 
                           horizontal=True)
    
    if input_method == "Single Product (Manual Entry)":
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📦 Product Information")
            
            # User-friendly visibility input
            visibility_option = st.select_slider(
                "Product Visibility",
                options=['Low (Hard to Spot)', 'Medium (Noticeable)', 'High (Very Prominent)'],
                value='Medium (Noticeable)',
                help="How prominently is this product displayed in your store?"
            )
            visibility_value = map_user_friendly_to_dataset(visibility_option, 'visibility')
            
            item_weight = st.number_input(
                "Product Weight (kg)", 
                min_value=0.1, 
                max_value=50.0, 
                value=10.0,
                help="Physical weight of the item"
            )
            
            item_fat_content = st.selectbox(
                "Fat Content", 
                df['Item_Fat_Content'].unique(),
                help="For food items: Low Fat or Regular"
            )
            
            item_type = st.selectbox(
                "Product Category", 
                sorted(df['Item_Type'].unique()),
                help="What type of product is this?"
            )
            
            item_mrp = st.number_input(
                "Maximum Retail Price - MRP (₹)", 
                min_value=10.0, 
                max_value=500.0, 
                value=100.0,
                help="The price tag on your product"
            )
        
        with col2:
            st.markdown("#### 🏪 Store Information")
            
            outlet_establishment_year = st.number_input(
                "Store Establishment Year", 
                min_value=1980, 
                max_value=2025, 
                value=2000,
                help="When was your store first opened?"
            )
            
            # User-friendly size mapping
            outlet_size_friendly = st.selectbox(
                "Store Size",
                ['Small', 'Medium', 'High'],
                help="Physical size of your retail outlet"
            )
            outlet_size = outlet_size_friendly  # Direct mapping works here
            
            # User-friendly location mapping
            location_friendly = st.selectbox(
                "City Type",
                ['Metro City', 'Developing City', 'Small Town'],
                help="What kind of area is your store located in?"
            )
            outlet_location_type = map_user_friendly_to_dataset(location_friendly, 'city_type')
            
            # User-friendly outlet type
            outlet_type_friendly = st.selectbox(
                "Store Format",
                ['Small Grocery Store', 'Supermarket Type 1', 'Supermarket Type 2', 'Supermarket Type 3'],
                help="What type of retail format?"
            )
            outlet_type = map_user_friendly_to_dataset(outlet_type_friendly, 'store_format')
        
        st.markdown("---")
        
        if st.button("📊 Estimate Sales & Get Insights", type="primary", use_container_width=True):
            # Create input dataframe with converted MRP (INR to USD for model)
            item_mrp_usd = item_mrp / USD_TO_INR
            
            input_data = pd.DataFrame({
                'Item_Weight': [item_weight],
                'Item_Fat_Content': [item_fat_content],
                'Item_Visibility': [visibility_value],
                'Item_Type': [item_type],
                'Item_MRP': [item_mrp_usd],
                'Outlet_Establishment_Year': [outlet_establishment_year],
                'Outlet_Size': [outlet_size],
                'Outlet_Location_Type': [outlet_location_type],
                'Outlet_Type': [outlet_type]
            })
            
            # Prepare and predict
            X = prepare_features(input_data)
            X_processed = preprocessor.transform(X)
            prediction_usd = model.predict(X_processed)[0]
            prediction_inr = prediction_usd * USD_TO_INR
            
            # Calculate additional metrics
            confidence_level, confidence_icon = calculate_confidence_level(input_data, df)
            expected_range_low = prediction_inr * 0.85
            expected_range_high = prediction_inr * 1.15
            
            # Display result with context
            st.success("✅ Estimation Complete!")
            
            # BUSINESS SUMMARY CARD
            st.markdown("### 📋 Business Summary")
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="Estimated Monthly Sales",
                    value=f"₹ {prediction_inr:,.0f}"
                )
            
            with col2:
                st.metric(
                    label="Expected Range",
                    value=f"₹ {expected_range_low:,.0f} - {expected_range_high:,.0f}"
                )
            
            with col3:
                st.metric(
                    label="Confidence Level",
                    value=f"{confidence_icon} {confidence_level}"
                )
            
            with col4:
                # Determine key improvement lever
                top_factors = explain_prediction(model, preprocessor, input_data)
                if top_factors:
                    key_lever = top_factors[0]['name']
                else:
                    key_lever = "Multiple Factors"
                st.metric(
                    label="Key Improvement Lever",
                    value=key_lever
                )
            
            st.caption("*Sales converted from dataset scale for demonstration purposes*")
            
            st.markdown("---")
            
            # PREDICTION EXPLANATION SECTION
            with st.expander("🔍 Why This Sales Estimate?", expanded=False):
                st.markdown("**Top 3 factors influencing this prediction:**")
                st.markdown("")
                
                explanation_factors = explain_prediction(model, preprocessor, input_data)
                
                for i, factor in enumerate(explanation_factors, 1):
                    col_a, col_b = st.columns([1, 3])
                    with col_a:
                        st.markdown(f"**#{i}**")
                        st.progress(factor['importance'])
                    with col_b:
                        st.markdown(f"**{factor['name']}**")
                        st.caption(factor['explanation'])
                    st.markdown("")
                
                st.info("💡 These are the main factors the model considers. Focus on the top factors for maximum impact.")
            
            st.markdown("---")
            
            # SCENARIO SIMULATION SECTION
            with st.expander("🎯 What-If Analysis (Scenario Simulation)", expanded=False):
                st.markdown("**See how changes could impact your sales:**")
                st.caption("*This is a simulation using the existing model, not retraining*")
                st.markdown("")
                
                scenarios = simulate_scenarios(model, preprocessor, input_data, prediction_usd)
                
                if scenarios:
                    for scenario in scenarios:
                        st.markdown(f"**{scenario['change']}**")
                        
                        col_x, col_y, col_z = st.columns(3)
                        with col_x:
                            st.metric("Current Estimate", f"₹ {scenario['original_sales']:,.0f}")
                        with col_y:
                            st.metric("After Change", f"₹ {scenario['new_sales']:,.0f}")
                        with col_z:
                            improvement_display = f"+{scenario['improvement_pct']:.1f}%" if scenario['improvement_pct'] > 0 else f"{scenario['improvement_pct']:.1f}%"
                            st.metric("Improvement", improvement_display)
                        
                        st.markdown("")
                else:
                    st.info("Your inputs are already optimized! No immediate improvement scenarios available.")
                
                st.caption("💭 These are simulated predictions based on changing specific factors.")
            
            st.markdown("---")
            
            # CATEGORY BENCHMARKING SECTION
            with st.expander("📊 How Does This Compare? (Category Benchmarks)", expanded=False):
                st.markdown("**Compare your estimate with similar products:**")
                st.markdown("")
                
                benchmarks = calculate_benchmarks(df, input_data, prediction_usd)
                
                for benchmark in benchmarks:
                    status_color = "🟢" if benchmark['status'] == "Above Average" else "🔵"
                    st.markdown(f"**{status_color} {benchmark['category']}**")
                    
                    col_p, col_q = st.columns(2)
                    with col_p:
                        st.metric("Average Sales", f"₹ {benchmark['avg_sales']:,.0f}")
                    with col_q:
                        st.metric("Your Estimate", f"₹ {benchmark['your_estimate']:,.0f}")
                    
                    st.caption(f"{benchmark['status']} by {benchmark['difference_pct']:.1f}%")
                    st.markdown("")
                
                st.info("💡 Use these benchmarks to set realistic targets and understand your market position.")
            
            st.markdown("---")
            
            # Business insights section
            st.markdown("### 💡 How Can You Improve These Sales?")
            st.markdown("*Based on your inputs, here are practical suggestions:*")
            
            insights = generate_business_insights(input_data, prediction_usd)
            
            if insights:
                for insight in insights:
                    with st.container():
                        st.markdown(f"**{insight['icon']} {insight['title']}**")
                        st.info(insight['suggestion'])
                        st.markdown("")
            
            # Disclaimer
            st.caption("💭 *Note: These suggestions are based on general retail principles, not ML predictions. Always consider your local market conditions.*")
    
    else:  # CSV Upload
        st.subheader("📂 Upload Your Product List")
        st.markdown("Upload a CSV file with your products. Required columns:")
        st.code("Item_Weight, Item_Fat_Content, Item_Visibility, Item_Type, Item_MRP, "
                "Outlet_Establishment_Year, Outlet_Size, Outlet_Location_Type, Outlet_Type")
        
        uploaded_file = st.file_uploader("Choose a CSV file", type=['csv'])
        
        if uploaded_file is not None:
            try:
                batch_df = pd.read_csv(uploaded_file)
                
                # Convert MRP from INR to USD if needed
                if 'Item_MRP' in batch_df.columns:
                    batch_df['Item_MRP'] = batch_df['Item_MRP'] / USD_TO_INR
                
                st.write(f"✅ Loaded {len(batch_df)} products")
                st.dataframe(batch_df.head(), use_container_width=True)
                
                if st.button("📊 Estimate Sales for All Products", type="primary"):
                    # Prepare and predict
                    X = prepare_features(batch_df)
                    X_processed = preprocessor.transform(X)
                    predictions_usd = model.predict(X_processed)
                    predictions_inr = predictions_usd * USD_TO_INR
                    
                    # Add predictions to dataframe
                    result_df = batch_df.copy()
                    result_df['Item_MRP'] = result_df['Item_MRP'] * USD_TO_INR  # Convert back to INR
                    result_df['Estimated_Monthly_Sales_INR'] = predictions_inr
                    
                    st.success(f"✅ Estimated sales for {len(result_df)} products!")
                    st.dataframe(result_df, use_container_width=True)
                    
                    # Download button
                    csv = result_df.to_csv(index=False)
                    st.download_button(
                        label="⬇️ Download Results as CSV",
                        data=csv,
                        file_name="sales_estimates.csv",
                        mime="text/csv"
                    )
            except Exception as e:
                st.error(f"Error processing file: {e}")
                st.info("Please make sure your CSV has all required columns with correct names.")

# TAB 2: ANALYTICS
with tab2:
    st.header("📊 How Accurate Is This Tool?")
    
    st.markdown("""
    This tool uses a machine learning model trained on real supermarket sales data. 
    Here's how well it performs:
    """)
    
    # Display evaluation metrics with explanations
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Average Error", f"₹ {metrics['test_mae'] * USD_TO_INR:,.0f}")
        st.caption("On average, predictions are off by this amount")
    
    with col2:
        st.metric("Root Mean Square Error", f"₹ {metrics['test_rmse'] * USD_TO_INR:,.0f}")
        st.caption("Measures typical prediction accuracy")
    
    with col3:
        st.metric("R² Score", f"{metrics['test_r2']:.2%}")
        st.caption("Explains 60% of sales variation")
    
    st.markdown("---")
    
    # Generate predictions on sample data for visualization
    st.subheader("📈 Actual vs Estimated Sales")
    st.markdown("*How well predictions match reality (sample of 1,000 products)*")
    
    sample_size = min(1000, len(df))
    sample_df = df.sample(n=sample_size, random_state=42)
    
    X_sample = prepare_features(sample_df)
    X_sample_processed = preprocessor.transform(X_sample)
    y_pred_usd = model.predict(X_sample_processed)
    y_actual_usd = sample_df['Item_Outlet_Sales'].values
    
    # Convert to INR
    y_pred_inr = y_pred_usd * USD_TO_INR
    y_actual_inr = y_actual_usd * USD_TO_INR
    
    # Create scatter plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=y_actual_inr,
        y=y_pred_inr,
        mode='markers',
        marker=dict(color='#1f77b4', size=6, opacity=0.6),
        name='Predictions',
        hovertemplate='Actual: ₹%{x:,.0f}<br>Predicted: ₹%{y:,.0f}<extra></extra>'
    ))
    
    # Add perfect prediction line
    min_val = min(y_actual_inr.min(), y_pred_inr.min())
    max_val = max(y_actual_inr.max(), y_pred_inr.max())
    fig.add_trace(go.Scatter(
        x=[min_val, max_val],
        y=[min_val, max_val],
        mode='lines',
        line=dict(color='red', dash='dash', width=2),
        name='Perfect Prediction'
    ))
    
    fig.update_layout(
        xaxis_title="Actual Sales (₹)",
        yaxis_title="Estimated Sales (₹)",
        height=500,
        hovermode='closest',
        showlegend=True
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("""
    **How to read this chart:**
    - Each dot represents one product
    - Dots closer to the red line = more accurate predictions
    - The model is reasonably accurate for most products
    """)

# TAB 3: ABOUT
with tab3:
    st.header("ℹ️ About RetailVista")
    
    st.markdown("""
    ### What Is This Tool?
    
    RetailVista is a decision-support system that helps retail businesses estimate product sales 
    and discover opportunities for improvement. It combines machine learning predictions with 
    practical business insights.
    
    ### What Can It Do?
    
    **Sales Estimation:** Predicts monthly sales for a product based on its characteristics and store details  
    **Business Insights:** Provides actionable suggestions to improve sales performance  
    **Batch Processing:** Upload multiple products at once for quick analysis  
    
    ### What It Cannot Do
    
    ⚠️ This tool has limitations:
    - Does not account for seasons, holidays, or special events
    - Does not know about your competitors or local market conditions
    - Does not consider promotions, discounts, or advertising
    - Predictions are estimates, not guarantees
    - Based on historical data that may not reflect future trends
    
    ### Technical Details
    
    **Dataset:** 8,523 retail transactions from supermarkets  
    **Model:** XGBoost machine learning algorithm  
    **Features Used:** 8 key factors (product weight, price, category, store size, location, etc.)  
    **Accuracy:** Predictions are typically within ₹{:,.0f} of actual sales  
    
    ### Currency Note
    
    All sales estimates are shown in Indian Rupees (₹) for easier understanding. The underlying model 
    was trained on data in USD scale, which we convert for display purposes.
    
    ### Who Made This?
    
    **Author:** Leslie Fernando  
    **Purpose:** Educational demonstration of ML in retail  
    **Code:** [github.com/lesliefdo08/RetailVista](https://github.com/lesliefdo08/RetailVista)
    
    ### Privacy & Data
    
    - No data you enter is stored or shared
    - All predictions happen locally on this app
    - Your business information remains private
    
    ---
    
    **Disclaimer:** This tool is for educational and demonstration purposes. It should not be the sole 
    basis for business decisions. Always consult with retail experts and analyze your specific market conditions.
    """.format(metrics['test_mae'] * USD_TO_INR))

# Footer
st.markdown("---")
st.caption("🛒 RetailVista - Helping retail businesses make informed decisions • Built with Streamlit")
