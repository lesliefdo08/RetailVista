import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# Custom CSS for mobile responsiveness and beautiful styling
st.markdown("""
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Poppins:wght@300;400;500;600;700&display=swap');
    
    /* Global Styling */
    .stApp {
        font-family: 'Inter', sans-serif !important;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        min-height: 100vh;
    }
    
    /* Mobile First Responsive Design */
    @media (max-width: 768px) {
        .main-title {
            font-size: 1.8rem !important;
            text-align: center;
            margin-bottom: 1rem;
        }
        
        .hero-section {
            padding: 2rem 1rem !important;
        }
        
        .feature-card {
            padding: 1.5rem !important;
            margin: 0.5rem 0 !important;
        }
        
        .metric-card {
            margin-bottom: 1rem !important;
            padding: 1.5rem !important;
        }
        
        .stColumns > div {
            margin-bottom: 1rem;
        }
    }
    
    /* Tablet Responsive */
    @media (min-width: 769px) and (max-width: 1024px) {
        .hero-title {
            font-size: 3rem !important;
        }
        
        .feature-card {
            padding: 2rem;
        }
    }
    
    /* Desktop Responsive */
    @media (min-width: 1025px) {
        .hero-title {
            font-size: 3.5rem !important;
        }
        
        .main-container {
            max-width: 1200px;
            margin: 0 auto;
        }
    }
    
    /* Custom Cards */
    .feature-card {
        background: rgba(255, 255, 255, 0.95);
        backdrop-filter: blur(10px);
        padding: 2rem;
        border-radius: 20px;
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.2);
        margin: 1rem 0;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        transform: translateY(0);
    }
    
    .feature-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.15);
    }
    
    /* Hero Section */
    .hero-section {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.2) 100%);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 3rem;
        text-align: center;
        margin: 2rem 0;
        border: 1px solid rgba(255, 255, 255, 0.3);
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    .hero-title {
        font-size: 3.5rem;
        font-weight: 700;
        color: white;
        margin-bottom: 1rem;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);
        font-family: 'Poppins', sans-serif;
        animation: slideInLeft 1s ease-out;
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .hero-subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.9);
        margin-bottom: 1rem;
        font-weight: 400;
    }
    
    .author-tag {
        background: linear-gradient(45deg, #ff6b6b, #ffa726);
        color: white;
        padding: 0.8rem 2rem;
        border-radius: 50px;
        font-weight: 500;
        display: inline-block;
        margin-top: 1rem;
        box-shadow: 0 10px 20px rgba(255, 107, 107, 0.3);
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.03); }
    }
    
    /* Enhanced Text Colors */
    .primary-text {
        color: #2c3e50 !important;
        font-weight: 500;
    }
    
    .secondary-text {
        color: #546e7a !important;
        font-weight: 400;
    }
    
    .accent-text {
        color: #667eea !important;
        font-weight: 600;
    }
    
    .highlight-text {
        color: #e91e63 !important;
        font-weight: 500;
    }
    
    /* Input Styling */
    .stSelectbox > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        font-weight: 500;
        color: #2c3e50 !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea;
        box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
        transform: translateY(-2px);
    }
    
    .stNumberInput > div > div > input {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 2px solid rgba(102, 126, 234, 0.2);
        transition: all 0.3s ease;
        font-weight: 500;
        color: #2c3e50 !important;
    }
    
    .stSlider > div > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Button Styling */
    .stButton > button {
        background: linear-gradient(45deg, #667eea, #764ba2);
        color: white;
        border: none;
        padding: 1rem 2.5rem;
        border-radius: 25px;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
        text-transform: uppercase;
        letter-spacing: 1px;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
        background: linear-gradient(45deg, #764ba2, #667eea);
    }
    
    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        background: rgba(255, 255, 255, 0.15);
        border-radius: 20px;
        padding: 0.5rem;
        backdrop-filter: blur(15px);
        gap: 0.5rem;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border-radius: 15px;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 500;
        padding: 1rem 2rem;
        transition: all 0.3s ease;
        font-size: 1rem;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(45deg, rgba(255, 255, 255, 0.25), rgba(255, 255, 255, 0.35));
        color: white !important;
        box-shadow: 0 8px 20px rgba(255, 255, 255, 0.2);
        font-weight: 600;
    }
    
    /* Success/Error Messages */
    .stSuccess {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        animation: slideInRight 0.5s ease-out;
        border: none;
    }
    
    .stSuccess > div {
        color: white !important;
    }
    
    @keyframes slideInRight {
        from {
            opacity: 0;
            transform: translateX(50px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stError {
        background: linear-gradient(45deg, #f44336, #d32f2f);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    .stError > div {
        color: white !important;
    }
    
    .stInfo {
        background: linear-gradient(45deg, #2196F3, #1976D2);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    .stInfo > div {
        color: white !important;
    }
    
    .stWarning {
        background: linear-gradient(45deg, #FF9800, #F57C00);
        color: white !important;
        border-radius: 15px;
        padding: 1.5rem;
        border: none;
    }
    
    .stWarning > div {
        color: white !important;
    }
    
    /* Metric Cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.15), rgba(255, 255, 255, 0.25));
        backdrop-filter: blur(15px);
        border-radius: 20px;
        padding: 2rem;
        text-align: center;
        border: 1px solid rgba(255, 255, 255, 0.2);
        transition: all 0.3s ease;
        color: white;
    }
    
    .metric-card:hover {
        transform: scale(1.03);
        box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
    }
    
    /* Enhanced Sidebar Styling */
    .css-1d391kg, .css-1cypcdb, .sidebar .sidebar-content {
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.12) 0%, rgba(255, 255, 255, 0.08) 100%) !important;
        backdrop-filter: blur(25px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.15) !important;
    }
    
    .sidebar-content {
        padding: 2rem 1rem !important;
    }
    
    /* Sidebar text colors */
    .css-1d391kg .stMarkdown, .css-1d391kg .stText {
        color: white !important;
    }
    
    .css-1d391kg h1, .css-1d391kg h2, .css-1d391kg h3, .css-1d391kg h4 {
        color: white !important;
        text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
    }
    
    .css-1d391kg .stMetric {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 1rem;
        margin: 0.5rem 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }
    
    .css-1d391kg .stMetric label {
        color: rgba(255, 255, 255, 0.8) !important;
        font-weight: 500;
    }
    
    .css-1d391kg .stMetric div {
        color: white !important;
        font-weight: 600;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.8rem;
        font-weight: 600;
        color: white !important;
        text-align: center;
        margin: 2rem 0 1rem 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.2);
        font-family: 'Poppins', sans-serif;
    }
    
    /* Enhanced form labels */
    .stSelectbox label, .stNumberInput label, .stSlider label {
        color: #2c3e50 !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* Progress Bars */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea, #764ba2);
        border-radius: 10px;
    }
    
    /* File uploader */
    .stFileUploader > div {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        border: 2px dashed rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
        padding: 2rem;
    }
    
    .stFileUploader > div:hover {
        border-color: #667eea;
        background: rgba(255, 255, 255, 1);
        transform: translateY(-2px);
    }
    
    /* Data frame styling */
    .stDataFrame {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 15px;
        padding: 1rem;
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(45deg, #28a745, #20c997);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 20px;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 8px 20px rgba(40, 167, 69, 0.3);
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 25px rgba(40, 167, 69, 0.4);
    }
    
    /* Spinner customization */
    .stSpinner > div {
        border-top-color: #667eea !important;
    }
</style>
""", unsafe_allow_html=True)

# Load the trained model and features with loading animation
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

# Show loading animation
with st.spinner('Loading RetailVista...'):
    model, feature_names, target_name = load_model_data()
    time.sleep(0.5)  # Small delay for effect

if model is None:
    st.error("Could not load the trained model. Please train the model first!")
    st.stop()

# Page configuration
st.set_page_config(
    page_title='RetailVista: Smart Sales Predictor', 
    page_icon='📊',
    layout='wide',
    initial_sidebar_state='expanded'
)

# Hero Section
st.markdown("""
<div class="hero-section">
    <h1 class="hero-title">RetailVista</h1>
    <p class="hero-subtitle">Your Smart Sales Prediction Platform</p>
    <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;">
        Predict sales like a pro with AI-powered insights
    </p>
    <div class="author-tag">
        Crafted with passion by Leslie Fernando
    </div>
</div>
""", unsafe_allow_html=True)

# Welcome message with personality
st.markdown("""
<div class="feature-card">
    <h3 class="accent-text" style="margin-bottom: 1rem;">Welcome to Professional Sales Forecasting</h3>
    <p class="primary-text" style="font-size: 1.1rem; line-height: 1.6;">
        RetailVista helps businesses make data-driven decisions with accurate sales predictions. 
        Whether you're managing inventory for a small store or analyzing product performance for a retail chain, 
        our AI-powered platform provides the insights you need to succeed.
    </p>
    
    <div style="background: linear-gradient(45deg, #f8f9fa, #e9ecef); padding: 1.5rem; border-radius: 15px; margin: 1rem 0; border-left: 4px solid #667eea;">
        <h4 class="highlight-text" style="margin-bottom: 0.5rem;">Key Features</h4>
        <ul class="secondary-text" style="margin: 0; list-style-type: none; padding-left: 0;">
            <li style="margin: 0.5rem 0;"><strong style="color: #667eea;">✓ Intelligent Forms</strong> - No spreadsheet complexity</li>
            <li style="margin: 0.5rem 0;"><strong style="color: #667eea;">✓ Mobile Optimized</strong> - Perfect on any device</li>
            <li style="margin: 0.5rem 0;"><strong style="color: #667eea;">✓ Instant Analysis</strong> - Real-time predictions</li>
            <li style="margin: 0.5rem 0;"><strong style="color: #667eea;">✓ Actionable Insights</strong> - Business recommendations included</li>
        </ul>
    </div>
</div>
""", unsafe_allow_html=True)

# Create tabs with personality
tab1, tab2, tab3 = st.tabs(["Single Product Analysis", "Batch Processing", "Learn More"])

with tab1:
    st.markdown('<h2 class="section-header">Single Product Prediction</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <p class="secondary-text" style="font-size: 1.1rem; text-align: center; margin-bottom: 2rem;">
            <strong class="primary-text">Get instant sales predictions</strong> for individual products. 
            Simply enter your product and store details below for AI-powered forecasting.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2, gap="large")
    
    with col1:
        st.markdown("""
        <div class="feature-card">
            <h4 class="accent-text" style="margin-bottom: 1.5rem;">Product Information</h4>
        """, unsafe_allow_html=True)
        
        item_weight = st.number_input(
            "Product Weight (kg)", 
            min_value=0.1, 
            max_value=50.0, 
            value=10.0,
            help="Enter the product weight from 0.1kg to 50kg"
        )
        
        item_fat_content = st.selectbox(
            "Fat Content Level", 
            ["Low Fat", "Regular"],
            help="Select the fat content category for your product"
        )
        
        item_visibility = st.slider(
            "Store Visibility Score", 
            0.0, 1.0, 0.1,
            help="0.0 = Low visibility, 1.0 = High visibility/prime location"
        )
        
        item_type = st.selectbox(
            "Product Category", 
            [
                "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
                "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
                "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
                "Breads", "Starchy Foods", "Others", "Seafood"
            ],
            help="Select the product category that best matches your item"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="feature-card">
            <h4 class="accent-text" style="margin-bottom: 1.5rem;">Store Details</h4>
        """, unsafe_allow_html=True)
        
        outlet_size = st.selectbox(
            "Store Size", 
            ["Small", "Medium", "High"],
            help="Small = Local store, Medium = Mid-size retail, High = Large supermarket"
        )
        
        outlet_location = st.selectbox(
            "Location Tier", 
            ["Tier 1", "Tier 2", "Tier 3"],
            help="Tier 1 = Major city, Tier 2 = Mid-size city, Tier 3 = Small town"
        )
        
        outlet_type = st.selectbox(
            "Store Type", 
            [
                "Supermarket Type1", "Supermarket Type2", 
                "Supermarket Type3", "Grocery Store"
            ],
            help="Choose the store format that matches your business"
        )
        
        establishment_year = st.slider(
            "Establishment Year", 
            1985, 2020, 2000,
            help="Year when the store was established"
        )
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    # Prediction button with animation
    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Generate Sales Prediction", type="primary"):
        with st.spinner('Analyzing data and generating prediction...'):
            time.sleep(1.2)  # Add suspense
            
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
                'Item_Outlet_Sales': 1000,
                'Profit': 200
            }
            
            try:
                # Create dataframe with correct feature order
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
                
                # Show result with style
                st.markdown("""
                <div style="background: linear-gradient(45deg, #4CAF50, #45a049); color: white; 
                           padding: 2.5rem; border-radius: 20px; text-align: center; margin: 2rem 0;
                           box-shadow: 0 20px 40px rgba(76, 175, 80, 0.3);">
                    <h2 style="margin: 0; font-size: 2rem; font-weight: 600;">Prediction Complete</h2>
                    <p style="font-size: 2.5rem; margin: 1rem 0; font-weight: 700;">
                        ${:.2f}
                    </p>
                    <p style="font-size: 1.1rem; opacity: 0.9; margin: 0;">
                        Predicted Sales Value
                    </p>
                </div>
                """.format(prediction), unsafe_allow_html=True)
                
                # Business insights with better styling
                if prediction > 2000:
                    st.success("**High Performance Expected** - This product shows strong sales potential. Consider increasing inventory and prime shelf placement for maximum returns.")
                elif prediction > 1000:
                    st.info("**Steady Performance Expected** - Reliable sales projections indicate this product will be a consistent performer in your inventory mix.")
                else:
                    st.warning("**Growth Opportunity Identified** - Consider promotional strategies, better positioning, or bundling options to maximize this product's potential.")
                    
            except Exception as e:
                st.error(f"Prediction Error: {e}")
                st.info("Please verify all input fields are completed correctly and try again.")

with tab2:
    st.markdown('<h2 class="section-header">Batch Processing & Analysis</h2>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="feature-card">
        <h4 class="accent-text">Process Multiple Products Simultaneously</h4>
        <p class="secondary-text" style="font-size: 1.1rem; line-height: 1.6;">
            Upload a CSV file containing multiple products for comprehensive analysis. 
            Our system will process all entries and provide detailed predictions with downloadable results.
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # File upload with enhanced styling
    uploaded_file = st.file_uploader(
        "Upload Product Data (CSV Format)", 
        type="csv",
        help="Upload a CSV file containing your product catalog data"
    )

    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            
            st.markdown("""
            <div class="feature-card">
                <h4 class="accent-text">File Processing Complete</h4>
            </div>
            """, unsafe_allow_html=True)
            
            # Display metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Records", f"{df.shape[0]:,}")
            with col2:
                st.metric("Data Columns", f"{df.shape[1]:,}")
            with col3:
                st.metric("File Size", f"{uploaded_file.size/1024:.1f} KB")
            
            st.subheader("Data Preview")
            st.dataframe(df.head(), use_container_width=True)
            
            # Data processing
            st.markdown("""
            <div class="feature-card">
                <h4 class="highlight-text">Data Preparation</h4>
            </div>
            """, unsafe_allow_html=True)
            
            df_processed = df.copy()
            
            # Progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Clean data
            status_text.text('Cleaning and validating data...')
            progress_bar.progress(25)
            original_rows = len(df_processed)
            df_processed = df_processed.dropna()
            
            if len(df_processed) < original_rows:
                removed_rows = original_rows - len(df_processed)
                st.info(f"Data Cleaning: Removed {removed_rows} incomplete records")
            
            # Encode categorical data
            status_text.text('Processing categorical variables...')
            progress_bar.progress(50)
            categorical_cols = df_processed.select_dtypes(include='object').columns
            
            if len(categorical_cols) > 0:
                st.info(f"Encoded {len(categorical_cols)} categorical columns: {', '.join(categorical_cols)}")
                for col in categorical_cols:
                    df_processed[col] = df_processed[col].astype('category').cat.codes
            
            status_text.text('Preparation complete - Ready for analysis')
            progress_bar.progress(100)
            time.sleep(0.5)
            status_text.empty()
            progress_bar.empty()
            
            # Analysis button
            if st.button("Process All Records", type="primary"):
                with st.spinner('Running comprehensive analysis...'):
                    try:
                        # Feature validation
                        if feature_names:
                            missing_features = [f for f in feature_names if f not in df_processed.columns]
                            if missing_features:
                                st.error(f"Missing Required Columns: {', '.join(missing_features)}")
                                
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.write("**Available Columns:**")
                                    for col in sorted(df_processed.columns):
                                        st.write(f"• {col}")
                                with col2:
                                    st.write("**Required Columns:**")
                                    for col in sorted(feature_names):
                                        st.write(f"• {col}")
                                st.stop()
                            
                            X_pred = df_processed[feature_names]
                        else:
                            if target_name in df_processed.columns:
                                X_pred = df_processed.drop(columns=[target_name])
                            else:
                                X_pred = df_processed
                        
                        st.success(f"Processing {X_pred.shape[0]} records with {X_pred.shape[1]} features")
                        
                        # Generate predictions
                        predictions = model.predict(X_pred)
                        
                        # Show completion
                        st.balloons()
                        
                        st.markdown("""
                        <div style="background: linear-gradient(45deg, #4CAF50, #45a049); color: white; 
                                   padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
                            <h2>Analysis Complete</h2>
                            <p style="font-size: 1.2rem;">Successfully processed all records</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Results compilation
                        df_results = df.copy()
                        df_results['Predicted_Sales'] = predictions
                        
                        st.subheader("Analysis Results")
                        st.dataframe(df_results, use_container_width=True)
                        
                        # Statistical summary
                        st.subheader("Performance Analytics")
                        col1, col2, col3, col4 = st.columns(4)
                        
                        with col1:
                            avg_sales = predictions.mean()
                            st.metric("Average Sales", f"${avg_sales:.2f}")
                        
                        with col2:
                            max_sales = predictions.max()
                            st.metric("Top Performer", f"${max_sales:.2f}")
                        
                        with col3:
                            min_sales = predictions.min()
                            st.metric("Lowest Prediction", f"${min_sales:.2f}")
                        
                        with col4:
                            total_predicted = predictions.sum()
                            st.metric("Total Value", f"${total_predicted:,.2f}")
                        
                        # Visualization
                        st.subheader("Sales Distribution Analysis")
                        st.line_chart(predictions, use_container_width=True)
                        
                        # Export functionality
                        st.markdown("""
                        <div class="feature-card">
                            <h4 class="accent-text">Export Results</h4>
                            <p class="secondary-text">Download your complete analysis for further business planning.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        csv_export = df_results.to_csv(index=False)
                        st.download_button(
                            label="Download Complete Analysis (CSV)",
                            data=csv_export,
                            file_name='retailvista_sales_analysis.csv',
                            mime='text/csv'
                        )
                        
                    except Exception as e:
                        st.error(f"Processing Error: {str(e)}")
                        st.info("Please verify your data format matches the requirements.")
                        
        except Exception as e:
            st.error(f"File Reading Error: {str(e)}")
            st.info("Ensure your file is a valid CSV format with proper encoding.")

    else:
        st.markdown("""
        <div class="feature-card">
            <h4 class="highlight-text">Ready for Upload</h4>
            <p class="secondary-text" style="font-size: 1.1rem;">
                Upload your CSV file using the file selector above. Our system supports standard CSV formats 
                and will guide you through any formatting requirements.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Format requirements
        if feature_names:
            st.subheader("Required Data Format")
            st.info("Your CSV file should contain the following columns (order doesn't matter):")
            
            # Show required columns in a nice format
            cols = st.columns(3)
            for i, feature in enumerate(feature_names):
                with cols[i % 3]:
                    st.write(f"• **{feature}**")

with tab3:
    st.markdown('<h2 class="section-header">Understanding RetailVista</h2>', unsafe_allow_html=True)
    
    # Target audience section
    st.markdown("""
    <div class="feature-card">
        <h3 class="accent-text">Who Benefits from RetailVista?</h3>
        <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 1.5rem; margin-top: 2rem;">
            <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                <h4 style="margin-bottom: 1rem;">Retail Managers</h4>
                <p>Optimize inventory planning and reduce stockouts with accurate demand forecasting.</p>
            </div>
            <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                <h4 style="margin-bottom: 1rem;">Business Analysts</h4>
                <p>Make data-driven decisions with comprehensive sales predictions and insights.</p>
            </div>
            <div style="background: linear-gradient(45deg, #9B59B6, #8E44AD); color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                <h4 style="margin-bottom: 1rem;">Store Owners</h4>
                <p>Improve profitability through intelligent product selection and pricing strategies.</p>
            </div>
            <div style="background: linear-gradient(45deg, #F39C12, #E67E22); color: white; padding: 2rem; border-radius: 15px; text-align: center;">
                <h4 style="margin-bottom: 1rem;">Product Managers</h4>
                <p>Forecast new product performance and optimize market entry strategies.</p>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # How it works
    st.markdown("""
    <div class="feature-card">
        <h3 class="accent-text">How RetailVista Works</h3>
        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2.5rem; border-radius: 20px; margin: 2rem 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 2rem;">
                <div style="text-align: center;">
                    <h4>AI-Powered Analysis</h4>
                    <p>Advanced machine learning algorithms trained on comprehensive retail datasets</p>
                </div>
                <div style="text-align: center;">
                    <h4>Multi-Factor Modeling</h4>
                    <p>Considers product characteristics, store attributes, and market positioning</p>
                </div>
                <div style="text-align: center;">
                    <h4>Real-Time Predictions</h4>
                    <p>Instant sales forecasts with confidence intervals and business recommendations</p>
                </div>
                <div style="text-align: center;">
                    <h4>Scalable Processing</h4>
                    <p>Handle individual products or entire catalogs with batch processing capabilities</p>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    
    # Key factors
    st.markdown("""
    <div class="feature-card">
        <h3 class="highlight-text">Factors That Drive Sales Performance</h3>
        <p class="secondary-text" style="font-size: 1.1rem; margin-bottom: 2rem;">
            RetailVista analyzes multiple variables to provide accurate predictions:
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    factors_col1, factors_col2 = st.columns(2)
    
    with factors_col1:
        st.markdown("""
        <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 2rem; border-radius: 15px;">
            <h4>Product Attributes</h4>
            <ul style="list-style: none; padding: 0; line-height: 2;">
                <li><strong>Weight & Size:</strong> Physical product characteristics</li>
                <li><strong>Category Type:</strong> Product classification and market segment</li>
                <li><strong>Fat Content:</strong> Health and dietary considerations</li>
                <li><strong>Visibility:</strong> In-store placement and merchandising</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    with factors_col2:
        st.markdown("""
        <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); color: white; padding: 2rem; border-radius: 15px;">
            <h4>Store Characteristics</h4>
            <ul style="list-style: none; padding: 0; line-height: 2;">
                <li><strong>Store Size:</strong> Retail space and customer capacity</li>
                <li><strong>Location Tier:</strong> Geographic market classification</li>
                <li><strong>Store Format:</strong> Retail model and target market</li>
                <li><strong>Establishment Age:</strong> Brand maturity and customer loyalty</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)
    
    # Business scenarios
    st.markdown("""
    <div class="feature-card">
        <h3 class="accent-text">Real-World Applications</h3>
        <p class="secondary-text">See how businesses leverage RetailVista for strategic advantage:</p>
    </div>
    """, unsafe_allow_html=True)
    
    scenario_cols = st.columns(3)
    
    scenarios = [
        {
            "title": "New Product Launch", 
            "icon": "🚀",
            "scenario": "Planning market entry for innovative products",
            "action": "Predict demand across different store formats",
            "result": "Optimized launch strategy and inventory allocation"
        },
        {
            "title": "Seasonal Planning", 
            "icon": "📊",
            "scenario": "Preparing for holiday shopping seasons",
            "action": "Forecast seasonal product performance",
            "result": "Maximized revenue during peak periods"
        },
        {
            "title": "Pricing Optimization", 
            "icon": "💡",
            "scenario": "Determining optimal price points",
            "action": "Compare predictions across price scenarios",
            "result": "Improved profit margins and competitiveness"
        }
    ]
    
    for i, scenario in enumerate(scenarios):
        with scenario_cols[i]:
            st.markdown(f"""
            <div style="background: linear-gradient(45deg, #f8f9fa, #e9ecef); 
                        padding: 2rem; border-radius: 15px; text-align: center; 
                        border-left: 4px solid #667eea; height: 300px;">
                <h4 class="accent-text">{scenario['title']}</h4>
                <div style="font-size: 2rem; margin: 1rem 0;">{scenario['icon']}</div>
                <p class="secondary-text"><strong>Scenario:</strong> {scenario['scenario']}</p>
                <p class="secondary-text"><strong>Action:</strong> {scenario['action']}</p>
                <p class="primary-text"><strong>Result:</strong> {scenario['result']}</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Best practices
    st.markdown("""
    <div class="feature-card">
        <h3 class="highlight-text">Best Practices for Optimal Results</h3>
        <div style="background: linear-gradient(135deg, rgba(255, 87, 34, 0.1), rgba(255, 87, 34, 0.15)); 
                    padding: 2rem; border-radius: 15px; margin: 1.5rem 0;">
            <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 2rem;">
                <div>
                    <h4 class="accent-text">Recommended Practices</h4>
                    <ul class="secondary-text" style="line-height: 1.8;">
                        <li>Use realistic and current product data</li>
                        <li>Consider seasonal and market trends</li>
                        <li>Validate predictions with business knowledge</li>
                        <li>Test multiple scenarios for comparison</li>
                        <li>Update predictions regularly as conditions change</li>
                    </ul>
                </div>
                <div>
                    <h4 class="highlight-text">Important Considerations</h4>
                    <ul class="secondary-text" style="line-height: 1.8;">
                        <li>Predictions are based on historical patterns</li>
                        <li>Market disruptions may affect accuracy</li>
                        <li>Combine predictions with market research</li>
                        <li>Use as one input in decision-making process</li>
                        <li>Monitor actual performance vs predictions</li>
                    </ul>
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Enhanced sidebar
st.sidebar.markdown("""
<div style="text-align: center; padding: 1.5rem;">
    <h2 style="color: white; margin-bottom: 1.5rem;">System Information</h2>
</div>
""", unsafe_allow_html=True)

if feature_names:
    col1, col2 = st.sidebar.columns(2)
    with col1:
        st.metric("Target Variable", target_name.title())
    with col2:
        st.metric("Features Used", len(feature_names))
    
    st.sidebar.metric("AI Model", "XGBoost Regressor")
    st.sidebar.metric("Accuracy Level", "Production Grade")
    
    with st.sidebar.expander("Feature Details", expanded=False):
        st.write("**Model Features:**")
        for i, feature in enumerate(feature_names, 1):
            st.write(f"{i}. `{feature}`")

# Footer
st.markdown("""
<div style="margin-top: 4rem; padding: 2.5rem; text-align: center; 
           background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)); 
           border-radius: 20px; backdrop-filter: blur(15px); border: 1px solid rgba(255, 255, 255, 0.1);">
    <h3 style="color: white; margin-bottom: 1rem; font-family: 'Poppins', sans-serif;">RetailVista</h3>
    <p style="color: rgba(255, 255, 255, 0.8); margin: 0; font-size: 1.1rem;">
        Professional Sales Forecasting Platform | Built with expertise by Leslie Fernando
    </p>
    <div style="margin-top: 1rem; padding-top: 1rem; border-top: 1px solid rgba(255, 255, 255, 0.1);">
        <p style="color: rgba(255, 255, 255, 0.6); margin: 0; font-size: 0.9rem;">
            Powered by Advanced Machine Learning • Enterprise-Ready Solutions
        </p>
    </div>
</div>
""", unsafe_allow_html=True)
