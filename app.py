import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time
import matplotlib.pyplot as plt

# Utility functions to reduce code repetition
def format_currency(value):
    """Format value as rupee currency"""
    return f"₹{value:,.0f}" if value >= 1 else f"₹{value:.2f}"

def create_section_header(title):
    """Create standardized section header"""
    st.markdown(f'<h2 class="section-header">{title}</h2>', unsafe_allow_html=True)

def create_metric_card(label, value, help_text=None):
    """Create standardized metric card"""
    if isinstance(value, (int, float)):
        value = format_currency(value)
    st.metric(label, value, help=help_text)

def create_chart(data, chart_type='bar', title='', color='#667eea'):
    """Create standardized chart"""
    fig, ax = plt.subplots(figsize=(10, 6))
    if chart_type == 'bar':
        data.plot(kind='bar', ax=ax, color=color)
    elif chart_type == 'barh':
        data.plot(kind='barh', ax=ax, color=color)
    elif chart_type == 'pie':
        ax.pie(data.values, labels=data.index, autopct='%1.1f%%', colors=['#667eea', '#764ba2', '#f093fb', '#f5576c'])
    ax.set_title(title)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

# Page configuration
st.set_page_config(
    page_title='RetailVista: Smart Sales Predictor', 
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Custom CSS for mobile responsiveness and beautiful styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Poppins:wght@300;400;500;600;700;800&display=swap');
    
    /* Global Styling with Enhanced Typography */
    .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
        font-optical-sizing: auto;
        -webkit-font-smoothing: antialiased;
        -moz-osx-font-smoothing: grayscale;
    }
    
    /* Enhanced Text Classes */
    .primary-text { color: #2d3748 !important; font-weight: 600 !important; line-height: 1.6 !important; font-size: 1.05rem !important; }
    .secondary-text { color: #1a202c !important; font-weight: 500 !important; line-height: 1.5 !important; font-size: 1rem !important; }
    .accent-text { color: #5a67d8 !important; font-weight: 700 !important; font-size: 1.2rem !important; }
    .highlight-text { color: #e53e3e !important; font-weight: 700 !important; font-size: 1.1rem !important; }
    
    /* Mobile First Responsive Design */
    @media (max-width: 768px) {
        /* Mobile responsiveness */
        .hero-title { font-size: 2.2rem !important; line-height: 1.2 !important; }
        .hero-subtitle { font-size: 1.1rem !important; line-height: 1.4 !important; }
        .hero-section { padding: 2rem 1rem !important; }
        .feature-card { padding: 1.5rem !important; margin: 1rem 0 !important; }
        .scenario-card { margin: 1rem 0 !important; height: auto !important; min-height: 200px !important; padding: 1.5rem !important; }
        .stColumns > div { margin-bottom: 1rem !important; }
        .primary-text { font-size: 0.95rem !important; }
        .secondary-text { font-size: 0.9rem !important; }
        .accent-text { font-size: 1rem !important; }
    }
    
    /* Hero Section with Enhanced Typography */
    .hero-section {
        background: rgba(255, 255, 255, 0.15);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .hero-title {
        font-family: 'Poppins', sans-serif;
        font-size: 3.2rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        line-height: 1.1;
        letter-spacing: -0.02em;
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 2rem;
    }
    
    .author-tag {
        background: linear-gradient(45deg, #ff6b6b, #ffa726);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 500;
        display: inline-block;
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
    }
    
    /* Feature Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    /* Scenario Cards - Mobile Friendly */
    .scenario-card {
        background: linear-gradient(45deg, #f8f9fa, #e9ecef);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        border-left: 4px solid #667eea;
        margin: 1rem 0;
        min-height: 280px;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    /* Enhanced Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 25px;
        font-family: 'Inter', sans-serif;
        font-weight: 600;
        font-size: 1rem;
        letter-spacing: 0.5px;
        transition: all 0.3s ease;
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
    }
    
    /* Enhanced Form Styling */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #1a202c !important;
        font-family: 'Inter', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        letter-spacing: 0.3px !important;
    }
    
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 0.5rem;
        gap: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 15px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        padding: 1rem 2rem;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: rgba(255, 255, 255, 0.25);
        color: white !important;
        box-shadow: 0 8px 20px rgba(255, 255, 255, 0.2);
    }
    
    /* Enhanced Section Headers */
    .section-header {
        font-family: 'Poppins', sans-serif;
        font-size: 1.9rem;
        font-weight: 600;
        color: white !important;
        text-align: center;
        margin: 2rem 0 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        letter-spacing: -0.01em;
        line-height: 1.2;
    }
    
    /* Messages */
    .stSuccess {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    .stError {
        background: linear-gradient(45deg, #f44336, #d32f2f);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    .stInfo {
        background: linear-gradient(45deg, #2196F3, #1976D2);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    .stWarning {
        background: linear-gradient(45deg, #FF9800, #F57C00);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    /* Sidebar */
    .css-1d391kg {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.08) 100%) !important;
        backdrop-filter: blur(25px) !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: white !important;
    }
    
    .css-1d391kg .stMetric {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and features
@st.cache_resource
def load_model_data():
    try:
        model_data = joblib.load('model/sales_predictor.joblib')
        if isinstance(model_data, dict):
            return model_data['model'], model_data['features'], model_data['target_name']
        else:
            return model_data, None, 'sales'
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, None, None

# Load model
with st.spinner('Loading RetailVista...'):
    model, feature_names, target_name = load_model_data()
    time.sleep(0.5)

if model is None:
    st.error("Could not load the trained model. Please train the model first!")
    st.stop()

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">RetailVista</h1>
    <p class="hero-subtitle">Smart Sales Prediction Platform</p>
    <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;">
        AI-powered insights for better business decisions
    </p>
    <div class="author-tag">
        Crafted by Leslie Fernando
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome Section - Better readability
st.markdown("""
<div style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(10px); 
           padding: 2.5rem; border-radius: 20px; margin: 2rem 0; 
           box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);">
    <h3 style="color: #5a67d8; font-weight: 700; font-size: 1.4rem; margin-bottom: 1.5rem; text-align: center;">
        Welcome to Professional Sales Forecasting
    </h3>
    <p style="color: #2d3748; font-weight: 500; font-size: 1.1rem; line-height: 1.7; text-align: center; margin-bottom: 2rem;">
        RetailVista helps businesses make data-driven decisions with accurate sales predictions. 
        Whether you're managing inventory for a small store or analyzing product performance for a retail chain, 
        our AI-powered platform provides the insights you need to succeed.
    </p>
</div>
""", unsafe_allow_html=True)

# Key Features - Using better styling
st.markdown("""
<div style="background: linear-gradient(45deg, #f7fafc, #edf2f7); 
           padding: 2rem; border-radius: 15px; margin: 1rem 0; 
           border-left: 4px solid #5a67d8;">
    <h4 style="color: #2b6cb0; font-weight: 700; font-size: 1.2rem; margin-bottom: 1rem; text-align: center;">
        Key Features
    </h4>
    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem;">
        <div style="color: #2d3748; font-weight: 600; font-size: 1rem;">
            <strong style="color: #5a67d8;">• Intelligent Forms</strong> - No spreadsheet complexity
        </div>
        <div style="color: #2d3748; font-weight: 600; font-size: 1rem;">
            <strong style="color: #5a67d8;">• Mobile Optimized</strong> - Perfect on any device
        </div>
        <div style="color: #2d3748; font-weight: 600; font-size: 1rem;">
            <strong style="color: #5a67d8;">• Instant Analysis</strong> - Real-time predictions
        </div>
        <div style="color: #2d3748; font-weight: 600; font-size: 1rem;">
            <strong style="color: #5a67d8;">• Actionable Insights</strong> - Business recommendations
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs(["Single Product Analysis", "Batch Processing", "Dashboard", "Learn More"])

with tab1:
    create_section_header("Single Product Prediction")
    st.info("Get instant sales predictions for individual products. Simply enter your product and store details below.")
    
    col1, col2 = st.columns(2, gap="large")
    
    # Product categories and store options
    product_categories = ["Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", "Household", "Baking Goods", "Snack Foods", "Frozen Foods", "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned", "Breads", "Starchy Foods", "Others", "Seafood"]
    store_sizes = ["Small", "Medium", "High"]
    location_tiers = ["Tier 1", "Tier 2", "Tier 3"]
    store_types = ["Supermarket Type1", "Supermarket Type2", "Supermarket Type3", "Grocery Store"]
    
    with col1:
        st.subheader("Product Information")
        item_weight = st.number_input("Product Weight (kg)", min_value=0.1, max_value=50.0, value=10.0, help="Enter the product weight from 0.1kg to 50kg")
        item_fat_content = st.selectbox("Fat Content Level", ["Low Fat", "Regular"], help="Select the fat content category for your product")
        item_visibility = st.slider("Store Visibility Score", 0.0, 1.0, 0.1, help="0.0 = Low visibility, 1.0 = High visibility/prime location")
        item_type = st.selectbox("Product Category", product_categories, help="Select the product category that best matches your item")
    
    with col2:
        st.subheader("Store Details")
        outlet_size = st.selectbox("Store Size", store_sizes, help="Small = Local store, Medium = Mid-size retail, High = Large supermarket")
        outlet_location = st.selectbox("Location Tier", location_tiers, help="Tier 1 = Major city, Tier 2 = Mid-size city, Tier 3 = Small town")
        outlet_type = st.selectbox("Store Type", store_types, help="Choose the store format that matches your business")
        establishment_year = st.slider("Establishment Year", 1985, 2020, 2000, help="Year when the store was established")
    
    # Prediction button
    if st.button("Generate Sales Prediction", type="primary"):
        with st.spinner('Analyzing data and generating prediction...'):
            time.sleep(1.2)
            
            # Create prediction data
            prediction_data = {
                'Item_Weight': item_weight,
                'Item_Fat_Content': 0 if item_fat_content == "Low Fat" else 1,
                'Item_Visibility': item_visibility,
                'Item_Type': product_categories.index(item_type),
                'Outlet_Size': store_sizes.index(outlet_size),
                'Outlet_Location_Type': location_tiers.index(outlet_location),
                'Outlet_Type': store_types.index(outlet_type),
                'Outlet_Establishment_Year': establishment_year,
                'Item_Identifier': 0, 'Outlet_Identifier': 0, 'Item_Outlet_Sales': 1000, 'Profit': 200
            }
            
            try:
                if feature_names:
                    filtered_data = {k: v for k, v in prediction_data.items() if k in feature_names}
                    df_pred = pd.DataFrame([filtered_data])
                    for feature in feature_names:
                        if feature not in df_pred.columns:
                            df_pred[feature] = 0
                    df_pred = df_pred[feature_names]
                else:
                    df_pred = pd.DataFrame([prediction_data])
                
                prediction = model.predict(df_pred)[0]
                
                # Show result
                st.success(f"**Predicted Sales: {format_currency(prediction)}**")
                
                # Business insights
                if prediction > 2000:
                    st.info("**High Performance Expected** - This product shows strong sales potential. Consider increasing inventory and prime shelf placement.")
                elif prediction > 1000:
                    st.info("**Steady Performance Expected** - Reliable sales projections indicate consistent performance.")
                else:
                    st.warning("**Growth Opportunity** - Consider promotional strategies or better positioning to maximize potential.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")

with tab2:
    create_section_header("Batch Processing & Analysis")
    st.info("Upload a CSV file containing multiple products for comprehensive analysis.")
    
    uploaded_file = st.file_uploader("Upload Product Data (CSV Format)", type="csv", help="Upload a CSV file containing your product catalog data")

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("File uploaded successfully!")
            
            col1, col2, col3 = st.columns(3)
            metrics_data = [("Total Records", f"{df.shape[0]:,}"), ("Data Columns", f"{df.shape[1]:,}"), ("File Size", f"{uploaded_file.size/1024:.1f} KB")]
            for col, (label, value) in zip([col1, col2, col3], metrics_data):
                with col:
                    st.metric(label, value)
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Process All Records", type="primary"):
                with st.spinner('Running comprehensive analysis...'):
                    df_processed = df.dropna()
                    categorical_cols = df_processed.select_dtypes(include='object').columns
                    
                    for col in categorical_cols:
                        df_processed[col] = df_processed[col].astype('category').cat.codes
                    
                    try:
                        if feature_names:
                            missing_features = [f for f in feature_names if f not in df_processed.columns]
                            if missing_features:
                                st.error(f"Missing required columns: {', '.join(missing_features)}")
                                st.stop()
                            X_pred = df_processed[feature_names]
                        else:
                            X_pred = df_processed
                        
                        predictions = model.predict(X_pred)
                        st.balloons()
                        st.success("Analysis Complete!")
                        
                        df_results = df.copy()
                        df_results['Predicted_Sales'] = predictions
                        
                        st.subheader("Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Statistics
                        col1, col2, col3, col4 = st.columns(4)
                        stats_data = [("Average Sales", predictions.mean()), ("Top Performer", predictions.max()), ("Lowest", predictions.min()), ("Total Value", f"₹{predictions.sum():,.2f}")]
                        for col, (label, value) in zip([col1, col2, col3, col4], stats_data):
                            with col:
                                if isinstance(value, str):
                                    st.metric(label, value)
                                else:
                                    create_metric_card(label, value)
                        
                        st.line_chart(predictions)
                        
                        csv_export = df_results.to_csv(index=False)
                        st.download_button("Download Complete Analysis (CSV)", csv_export, 'retailvista_analysis.csv', 'text/csv')
                        
                    except Exception as e:
                        st.error(f"Processing Error: {str(e)}")
                        
        except Exception as e:
            st.error(f"File Reading Error: {str(e)}")

    else:
        st.info("Upload your CSV file to get started with batch processing.")
        
        if feature_names:
            st.subheader("Required Data Format")
            st.write("Your CSV should contain these columns:")
            
            cols = st.columns(3)
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    st.write(f"• **{feature}**")

with tab3:
    create_section_header("Analytics Dashboard")
    st.info("Explore comprehensive statistics and insights from the retail sales dataset.")
    
    @st.cache_data
    def load_sales_data():
        try:
            df = pd.read_csv('data/supermarket_sales.csv')
            return df
        except Exception as e:
            st.error(f"Error loading data: {str(e)}")
            return None
    
    df = load_sales_data()
    
    if df is not None:
        # Key Metrics Row
        st.subheader("📊 Key Performance Indicators")
        
        col1, col2, col3, col4, col5 = st.columns(5)
        
        metrics = [
            ("Total Sales", df['Item_Outlet_Sales'].sum(), "Sum of all item outlet sales"),
            ("Average Sales", df['Item_Outlet_Sales'].mean(), "Mean sales per item"),
            ("Total Profit", df['Profit'].sum(), "Sum of all profits"),
            ("Avg Profit Margin", f"{(df['Profit'] / df['Item_Outlet_Sales'] * 100).mean():.1f}%", "Average profit margin percentage"),
            ("Total Products", f"{df.shape[0]:,}", "Number of products in dataset")
        ]
        
        for col, (label, value, help_text) in zip([col1, col2, col3, col4, col5], metrics):
            with col:
                if "%" in str(value) or "," in str(value):
                    st.metric(label, value, help=help_text)
                else:
                    create_metric_card(label, value, help_text)
        
        st.markdown("---")
        
        # Sales Analysis Section
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🛍️ Sales by Product Category")
            category_sales = df.groupby('Item_Type')['Item_Outlet_Sales'].sum().sort_values(ascending=False)
            create_chart(category_sales.head(10), 'barh', 'Top 10 Categories by Sales')
            
            with st.expander("View Detailed Category Statistics"):
                category_df = pd.DataFrame({
                    'Category': category_sales.index,
                    'Total Sales': [format_currency(x) for x in category_sales.values],
                    'Products Count': df.groupby('Item_Type').size().values
                }).set_index('Category')
                st.dataframe(category_df, use_container_width=True)
        
        with col2:
            st.subheader("🏪 Performance by Store Type")
            outlet_performance = df.groupby('Outlet_Type').agg({
                'Item_Outlet_Sales': ['sum', 'mean', 'count'], 'Profit': 'sum'
            }).round(2)
            outlet_performance.columns = ['Total Sales', 'Avg Sales', 'Product Count', 'Total Profit']
            
            # Format currency columns
            for col in ['Total Sales', 'Avg Sales', 'Total Profit']:
                outlet_performance[col] = outlet_performance[col].apply(lambda x: format_currency(x))
            
            st.dataframe(outlet_performance, use_container_width=True)
            
            # Pie chart for store type distribution
            store_sales = df.groupby('Outlet_Type')['Item_Outlet_Sales'].sum()
            create_chart(store_sales, 'pie', 'Sales Distribution by Store Type')
        
        st.markdown("---")
        
        # Geographic and Age Analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🌍 Geographic Performance")
            location_stats = df.groupby('Outlet_Location_Type').agg({
                'Item_Outlet_Sales': ['sum', 'mean'], 'Profit': 'mean', 'Item_Identifier': 'count'
            }).round(2)
            location_stats.columns = ['Total Sales', 'Avg Sales', 'Avg Profit', 'Store Count']
            
            # Format currency columns
            for col in ['Total Sales', 'Avg Sales', 'Avg Profit']:
                location_stats[col] = location_stats[col].apply(lambda x: format_currency(x))
            
            st.dataframe(location_stats, use_container_width=True)
            
            location_sales = df.groupby('Outlet_Location_Type')['Item_Outlet_Sales'].mean()
            create_chart(location_sales, 'bar', 'Average Sales by Location Tier', '#764ba2')
        
        with col2:
            st.subheader("📅 Store Age Analysis")
            df['Store_Age'] = 2025 - df['Outlet_Establishment_Year']
            age_performance = df.groupby('Store_Age').agg({'Item_Outlet_Sales': 'mean', 'Profit': 'mean'}).round(2)
            
            fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))
            age_performance['Item_Outlet_Sales'].plot(ax=ax1, color='#667eea', marker='o')
            ax1.set_ylabel('Average Sales (₹)')
            ax1.set_title('Sales Performance vs Store Age')
            ax1.grid(True, alpha=0.3)
            
            age_performance['Profit'].plot(ax=ax2, color='#f5576c', marker='s')
            ax2.set_ylabel('Average Profit (₹)')
            ax2.set_xlabel('Store Age (Years)')
            ax2.set_title('Profit vs Store Age')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()
        
        st.markdown("---")
        
        # Product Insights
        st.subheader("🔍 Product Performance Insights")
        col1, col2, col3 = st.columns(3)
        
        insights = [
            ("Fat Content Impact", df.groupby('Item_Fat_Content')['Item_Outlet_Sales'].mean()),
            ("Store Size Impact", df.groupby('Outlet_Size')['Item_Outlet_Sales'].mean()),
            ("Visibility Impact", df.groupby(pd.cut(df['Item_Visibility'], bins=[0, 0.05, 0.1, 0.15, 1.0], labels=['Very Low', 'Low', 'Medium', 'High']))['Item_Outlet_Sales'].mean())
        ]
        
        for col, (title, data) in zip([col1, col2, col3], insights):
            with col:
                st.subheader(title)
                for category, sales in data.items():
                    if pd.notna(sales):
                        create_metric_card(f"{category}", sales)
        
        st.markdown("---")
        
        # Top and Bottom Performers
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🏆 Top Performing Products")
            top_products = df.nlargest(10, 'Item_Outlet_Sales')[['Item_Identifier', 'Item_Type', 'Item_Outlet_Sales', 'Outlet_Type']]
            top_products['Item_Outlet_Sales'] = top_products['Item_Outlet_Sales'].apply(lambda x: format_currency(x))
            st.dataframe(top_products.set_index('Item_Identifier'), use_container_width=True)
        
        with col2:
            st.subheader("📉 Improvement Opportunities")
            bottom_products = df.nsmallest(10, 'Item_Outlet_Sales')[['Item_Identifier', 'Item_Type', 'Item_Outlet_Sales', 'Outlet_Type']]
            bottom_products['Item_Outlet_Sales'] = bottom_products['Item_Outlet_Sales'].apply(lambda x: format_currency(x))
            st.dataframe(bottom_products.set_index('Item_Identifier'), use_container_width=True)
        
        st.markdown("---")
        
        # Export section
        st.subheader("📥 Export Analytics Data")
        col1, col2, col3 = st.columns(3)
        
        exports = [
            ("📊 Category Analysis (CSV)", df.groupby('Item_Type').agg({'Item_Outlet_Sales': ['sum', 'mean', 'count'], 'Profit': ['sum', 'mean']}).round(2), 'category_analysis.csv'),
            ("🏪 Store Performance (CSV)", df.groupby(['Outlet_Type', 'Outlet_Location_Type']).agg({'Item_Outlet_Sales': ['sum', 'mean'], 'Profit': 'mean'}).round(2), 'store_performance.csv'),
            ("📋 Complete Dataset (CSV)", df.assign(Profit_Margin=lambda x: (x['Profit'] / x['Item_Outlet_Sales'] * 100).round(2)), 'complete_analysis.csv')
        ]
        
        for col, (label, data, filename) in zip([col1, col2, col3], exports):
            with col:
                st.download_button(label, data.to_csv(), filename, 'text/csv')
    
    else:
        st.error("Unable to load the sales dataset. Please ensure the data file is available.")

with tab4:
    create_section_header("Understanding RetailVista")
    
    # Who benefits section
    st.subheader("Who Benefits from RetailVista?")
    
    benefits_data = [
        ("Retail Managers", ["Optimize inventory planning", "Reduce stockouts and overstock", "Improve demand forecasting"]),
        ("Store Owners", ["Better product selection", "Improved pricing strategies", "Maximize profitability"]),
        ("Business Analysts", ["Data-driven decision making", "Comprehensive sales insights", "Performance analytics"]),
        ("Product Managers", ["New product forecasting", "Market entry strategies", "Performance optimization"])
    ]
    
    col1, col2 = st.columns(2)
    for i, (title, items) in enumerate(benefits_data):
        with col1 if i % 2 == 0 else col2:
            st.markdown(f"**{title}**\n" + "\n".join([f"- {item}" for item in items]))
    
    # How it works and factors
    st.subheader("How RetailVista Works")
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**AI-Powered Analysis**\n- Advanced machine learning algorithms\n- Trained on comprehensive retail datasets\n- Continuous learning and improvement")
    with col2:
        st.markdown("**Multi-Factor Modeling**\n- Product characteristics analysis\n- Store attribute considerations\n- Market positioning factors")
    
    st.subheader("Sales Performance Factors")
    
    factors = [
        ("Product Attributes", ["Weight & Size: Physical product characteristics", "Category Type: Product classification and market segment", "Fat Content: Health and dietary considerations", "Visibility: In-store placement and merchandising"]),
        ("Store Characteristics", ["Store Size: Retail space and customer capacity", "Location Tier: Geographic market classification", "Store Format: Retail model and target market", "Establishment Age: Brand maturity and customer loyalty"])
    ]
    
    for title, items in factors:
        with st.expander(title):
            st.markdown("\n".join([f"- **{item.split(':')[0]}**: {item.split(':')[1]}" for item in items]))
    
    # Applications and best practices
    st.subheader("Real-World Applications")
    
    scenarios = [
        {"title": "New Product Launch", "description": "Plan market entry for innovative products by predicting demand across store formats."},
        {"title": "Seasonal Planning", "description": "Prepare for holiday seasons by forecasting seasonal product performance."},
        {"title": "Pricing Optimization", "description": "Determine optimal price points by comparing predictions across scenarios."}
    ]
    
    for scenario in scenarios:
        st.markdown(f"**{scenario['title']}**")
        st.write(scenario['description'])
        st.markdown("---")
    
    st.subheader("Best Practices")
    
    practices = [
        ("Recommended Practices", ["Use realistic and current data", "Consider seasonal trends", "Validate with business knowledge", "Test multiple scenarios", "Regular updates as conditions change"]),
        ("Important Considerations", ["Based on historical patterns", "Market disruptions may affect accuracy", "Combine with market research", "Use as one input in decisions", "Monitor performance vs predictions"])
    ]
    
    col1, col2 = st.columns(2)
    for col, (title, items) in zip([col1, col2], practices):
        with col:
            st.markdown(f"**{title}**\n" + "\n".join([f"- {item}" for item in items]))

# Enhanced sidebar
if feature_names:
    st.sidebar.markdown("## System Information")
    
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Target Variable", target_name.title())
    with col2:
        st.metric("Features Used", len(feature_names))
    
    st.sidebar.metric("AI Model", "XGBoost Regressor")
    
    with st.sidebar.expander("Feature Details"):
        for i, feature in enumerate(feature_names, 1):
            st.write(f"{i}. {feature}")

# Footer
st.markdown("""
---
**RetailVista** - Professional Sales Forecasting Platform  
*Built with expertise by Leslie Fernando*  
Powered by Advanced Machine Learning
""")
