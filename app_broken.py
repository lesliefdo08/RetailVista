import streamlit as st
import pandas as pd
import joblib
import numpy as np
import time

# Custom CSS for mobile responsiveness and beautiful styling
st.markdown("""
<style>
 /* Import Google Fonts */
 @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');
 
 /* Global Styling */
 .stApp {
 font-family: 'Poppins', sans-serif !important;
 background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
 min-height: 100vh;
 }
 
 /* Mobile First Responsive Design */
 @media (max-width: 768px) {
 .main-title {
 font-size: 2rem !important;
 text-align: center;
 margin-bottom: 1rem;
 }
 
 .stColumns {
 flex-direction: column !important;
 }
 
 .metric-card {
 margin-bottom: 1rem !important;
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
 transform: translateY(-10px);
 box-shadow: 0 30px 60px rgba(0, 0, 0, 0.15);
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
 50% { transform: scale(1.05); }
 }
 
 /* Input Styling */
 .stSelectbox > div > div {
 background: rgba(255, 255, 255, 0.9);
 border-radius: 15px;
 border: 2px solid rgba(102, 126, 234, 0.3);
 transition: all 0.3s ease;
 }
 
 .stSelectbox > div > div:focus-within {
 border-color: #667eea;
 box-shadow: 0 0 20px rgba(102, 126, 234, 0.2);
 transform: translateY(-2px);
 }
 
 .stNumberInput > div > div {
 background: rgba(255, 255, 255, 0.9);
 border-radius: 15px;
 border: 2px solid rgba(102, 126, 234, 0.3);
 transition: all 0.3s ease;
 }
 
 .stSlider > div > div {
 background: rgba(255, 255, 255, 0.9);
 border-radius: 15px;
 padding: 1rem;
 }
 
 /* Button Styling */
 .stButton > button {
 background: linear-gradient(45deg, #667eea, #764ba2);
 color: white;
 border: none;
 padding: 1rem 2rem;
 border-radius: 25px;
 font-weight: 600;
 font-size: 1.1rem;
 transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
 box-shadow: 0 10px 25px rgba(102, 126, 234, 0.3);
 text-transform: uppercase;
 letter-spacing: 1px;
 }
 
 .stButton > button:hover {
 transform: translateY(-3px);
 box-shadow: 0 15px 35px rgba(102, 126, 234, 0.4);
 background: linear-gradient(45deg, #764ba2, #667eea);
 }
 
 /* Tabs Styling */
 .stTabs [data-baseweb="tab-list"] {
 background: rgba(255, 255, 255, 0.1);
 border-radius: 20px;
 padding: 0.5rem;
 backdrop-filter: blur(10px);
 gap: 0.5rem;
 }
 
 .stTabs [data-baseweb="tab"] {
 background: transparent;
 border-radius: 15px;
 color: white;
 font-weight: 500;
 padding: 1rem 2rem;
 transition: all 0.3s ease;
 }
 
 .stTabs [aria-selected="true"] {
 background: linear-gradient(45deg, rgba(255, 255, 255, 0.2), rgba(255, 255, 255, 0.3));
 color: white;
 box-shadow: 0 5px 15px rgba(255, 255, 255, 0.2);
 }
 
 /* Success/Error Messages */
 .stSuccess {
 background: linear-gradient(45deg, #4CAF50, #45a049);
 color: white;
 border-radius: 15px;
 padding: 1.5rem;
 animation: slideInRight 0.5s ease-out;
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
 color: white;
 border-radius: 15px;
 padding: 1.5rem;
 }
 
 .stInfo {
 background: linear-gradient(45deg, #2196F3, #1976D2);
 color: white;
 border-radius: 15px;
 padding: 1.5rem;
 }
 
 /* Metric Cards */
 .metric-card {
 background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.2));
 backdrop-filter: blur(10px);
 border-radius: 20px;
 padding: 2rem;
 text-align: center;
 border: 1px solid rgba(255, 255, 255, 0.2);
 transition: all 0.3s ease;
 }
 
 .metric-card:hover {
 transform: scale(1.05);
 box-shadow: 0 20px 40px rgba(0, 0, 0, 0.1);
 }
 
 /* Sidebar Styling */
 .css-1d391kg {
 background: linear-gradient(180deg, rgba(255, 255, 255, 0.1) 0%, rgba(255, 255, 255, 0.05) 100%);
 backdrop-filter: blur(20px);
 }
 
 /* Custom Section Headers */
 .section-header {
 font-size: 1.8rem;
 font-weight: 600;
 color: white;
 text-align: center;
 margin: 2rem 0 1rem 0;
 text-shadow: 1px 1px 2px rgba(0, 0, 0, 0.2);
 }
 
 /* Loading Animation */
 .loading-animation {
 display: flex;
 justify-content: center;
 align-items: center;
 height: 100px;
 }
 
 .spinner {
 width: 50px;
 height: 50px;
 border: 4px solid rgba(255, 255, 255, 0.3);
 border-top: 4px solid white;
 border-radius: 50%;
 animation: spin 1s linear infinite;
 }
 
 @keyframes spin {
 0% { transform: rotate(0deg); }
 100% { transform: rotate(360deg); }
 }
 
 /* Progress Bars */
 .stProgress > div > div {
 background: linear-gradient(90deg, #667eea, #764ba2);
 border-radius: 10px;
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
 st.error(f"Oops! Having trouble loading the model: {e}")
 return None, None, None

# Show loading animation
with st.spinner('Loading RetailVista magic...'):
 model, feature_names, target_name = load_model_data()
 time.sleep(1) # Small delay for effect

if model is None:
 st.error("Hmm, seems like our AI brain isn't ready yet. Please train the model first!")
 st.stop()

# Page configuration
st.set_page_config(
 page_title='RetailVista: Your Smart Sales Crystal Ball', 
 page_icon='',
 layout='wide',
 initial_sidebar_state='expanded'
)

# Hero Section
st.markdown("""
<div class="hero-section">
 <h1 class="hero-title">RetailVista</h1>
 <p class="hero-subtitle">Your Personal Sales Crystal Ball</p>
 <p style="color: rgba(255, 255, 255, 0.8); font-size: 1.1rem; margin-bottom: 2rem;">
 Predict sales like a pro, no PhD in data science required!
 </p>
 <div class="author-tag">
 Crafted with love by Leslie Fernando
 </div>
</div>
""", unsafe_allow_html=True)

# Welcome message with personality
st.markdown("""
<div class="feature-card">
 <h3 style="color: #667eea; margin-bottom: 1rem;">Hey there, future retail genius!</h3>
 <p style="font-size: 1.1rem; line-height: 1.6; color: #555;">
 Welcome to RetailVista! I'm here to help you predict sales like a seasoned business expert. 
 Whether you're running a corner store or managing a supermarket chain, I've got your back! 
 Just tell me about your product and store, and I'll give you insights that would make even 
 the most experienced retailers jealous.
 </p>
 
 <div style="background: linear-gradient(45deg, #ff9a9e, #fecfef); padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
 <h4 style="color: #d63384; margin-bottom: 0.5rem;">What makes this special?</h4>
 <ul style="color: #6f4e7c; margin: 0;">
 <li><strong>No complicated spreadsheets</strong> - Just friendly forms!</li>
 <li><strong>Mobile-friendly</strong> - Works perfectly on your phone</li>
 <li><strong>Instant predictions</strong> - Get results in seconds</li>
 <li><strong>Business insights</strong> - Not just numbers, but actionable advice</li>
 </ul>
 </div>
</div>
""", unsafe_allow_html=True)

# Create tabs with personality
tab1, tab2, tab3 = st.tabs(["Quick Magic", "Batch Wizard", "Learn & Explore"])

with tab1:
 st.markdown('<h2 class="section-header">Single Product Prediction</h2>', unsafe_allow_html=True)
 
 st.markdown("""
 <div class="feature-card">
 <p style="font-size: 1.1rem; text-align: center; color: #666; margin-bottom: 2rem;">
 <strong>Step right up!</strong> Let's predict your product's sales potential. 
 Just fill in the details below and watch the magic happen!
 </p>
 </div>
 """, unsafe_allow_html=True)
 
 col1, col2 = st.columns(2)
 
 with col1:
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #667eea; margin-bottom: 1.5rem;">Tell me about your product</h4>
 """, unsafe_allow_html=True)
 
 item_weight = st.number_input(
 "How much does it weigh? (kg)", 
 min_value=0.1, 
 max_value=50.0, 
 value=10.0,
 help="From a light snack (0.1kg) to a hefty family pack (50kg)"
 )
 
 item_fat_content = st.selectbox(
 "What's the fat situation?", 
 ["Low Fat", "Regular"],
 help="Health-conscious shoppers love to know!"
 )
 
 item_visibility = st.slider(
 "How visible is it in store?", 
 0.0, 1.0, 0.1,
 help="0.0 = Hidden in the back, 1.0 = Prime real estate!"
 )
 
 item_type = st.selectbox(
 "What category does it belong to?", 
 [
 "Dairy", "Soft Drinks", "Meat", "Fruits and Vegetables", 
 "Household", "Baking Goods", "Snack Foods", "Frozen Foods",
 "Breakfast", "Health and Hygiene", "Hard Drinks", "Canned",
 "Breads", "Starchy Foods", "Others", "Seafood"
 ],
 help="Choose the category that best fits your product"
 )
 
 st.markdown("</div>", unsafe_allow_html=True)
 
 with col2:
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #764ba2; margin-bottom: 1.5rem;">What's your store like?</h4>
 """, unsafe_allow_html=True)
 
 outlet_size = st.selectbox(
 "How big is your store?", 
 ["Small", "Medium", "High"],
 help="Small = Cozy corner shop, High = Massive supermarket"
 )
 
 outlet_location = st.selectbox(
 "Where are you located?", 
 ["Tier 1", "Tier 2", "Tier 3"],
 help="Tier 1 = Big city, Tier 2 = Mid-size town, Tier 3 = Smaller community"
 )
 
 outlet_type = st.selectbox(
 "What type of store?", 
 [
 "Supermarket Type1", "Supermarket Type2", 
 "Supermarket Type3", "Grocery Store"
 ],
 help="Different store types have different customer behaviors"
 )
 
 establishment_year = st.slider(
 "When was your store established?", 
 1985, 2020, 2000,
 help="Older stores often have more loyal customers"
 )
 
 st.markdown("</div>", unsafe_allow_html=True)
 
 # Prediction button with animation
 if st.button("Predict My Sales!", type="primary"):
 with st.spinner('Consulting the sales spirits...'):
 time.sleep(1.5) # Add suspense!
 
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
 padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;
 box-shadow: 0 20px 40px rgba(76, 175, 80, 0.3); animation: slideInUp 0.5s ease-out;">
 <h2 style="margin: 0; font-size: 2.5rem;"> Ta-da!</h2>
 <p style="font-size: 2rem; margin: 1rem 0; font-weight: bold;">
 Predicted Sales: ${:.2f}
 </p>
 <p style="font-size: 1.2rem; opacity: 0.9; margin: 0;">
 Your sales crystal ball has spoken! 
 </p>
 </div>
 """.format(prediction), unsafe_allow_html=True)
 
 # Personalized insights with emojis
 if prediction > 2000:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); color: white; 
 padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
 <h4> Woohoo! High Sales Expected!</h4>
 <p>This product is going to be a rockstar! Consider stocking up because customers are going to love it. 
 You might even want to give it prime shelf space!</p>
 </div>
 """, unsafe_allow_html=True)
 elif prediction > 1000:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); color: white; 
 padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
 <h4> Steady Eddie - Reliable Sales!</h4>
 <p>This is your dependable performer! Not flashy, but consistent. Perfect for maintaining 
 steady cash flow and happy customers. A solid choice for your inventory!</p>
 </div>
 """, unsafe_allow_html=True)
 else:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #FFB74D, #FFA726); color: white; 
 padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
 <h4> Time for Strategy - Growth Potential!</h4>
 <p>Don't worry! Lower predictions just mean opportunities. Consider a promotion, 
 better placement, or maybe bundle it with a popular item. Sometimes the best diamonds need a little polishing!</p>
 </div>
 """, unsafe_allow_html=True)
 
 except Exception as e:
 st.error(f"Oops! Something went wrong: {e}")
 st.info(" Don't worry, this happens sometimes. Try refreshing the page or check if all fields are filled correctly!")

with tab2:
 st.markdown('<h2 class="section-header"> Batch Processing Wizard</h2>', unsafe_allow_html=True)
 
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #667eea;">🧙‍ Welcome to the Batch Processing Wizard!</h4>
 <p style="font-size: 1.1rem; color: #666; line-height: 1.6;">
 Got a whole catalog of products? No problem! Upload your CSV file and I'll work my magic 
 on all of them at once. It's like having a team of data scientists working for you! 
 </p>
 </div>
 """, unsafe_allow_html=True)
 
 # File upload with style
 uploaded_file = st.file_uploader(
 " Choose your CSV file", 
 type="csv",
 help="Upload a CSV file with your product data. Don't worry, I'll guide you through any issues!"
 )

 if uploaded_file is not None:
 try:
 df = pd.read_csv(uploaded_file)
 
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #4CAF50;"> File uploaded successfully! Here's what I see:</h4>
 </div>
 """, unsafe_allow_html=True)
 
 col1, col2, col3 = st.columns(3)
 with col1:
 st.metric(" Total Rows", f"{df.shape[0]:,}")
 with col2:
 st.metric(" Total Columns", f"{df.shape[1]:,}")
 with col3:
 st.metric(" File Size", f"{uploaded_file.size:,} bytes")
 
 st.write("### Data Preview (First 5 rows):")
 st.dataframe(df.head(), use_container_width=True)
 
 # Preprocessing section
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #FF9800;"> Preparing your data...</h4>
 </div>
 """, unsafe_allow_html=True)
 
 df_processed = df.copy()
 
 # Progress bar for processing
 progress_bar = st.progress(0)
 status_text = st.empty()
 
 # Remove missing values
 status_text.text('🧹 Cleaning up missing values...')
 progress_bar.progress(25)
 original_rows = len(df_processed)
 df_processed = df_processed.dropna()
 
 if len(df_processed) < original_rows:
 removed_rows = original_rows - len(df_processed)
 st.info(f"🧹 Cleaned up {removed_rows} rows with missing data. Your data is now squeaky clean!")
 
 # Encode categorical variables
 status_text.text(' Converting text to numbers...')
 progress_bar.progress(50)
 categorical_cols = df_processed.select_dtypes(include='object').columns
 
 if len(categorical_cols) > 0:
 st.info(f" Converting these text columns to numbers: {', '.join(categorical_cols)}")
 for col in categorical_cols:
 df_processed[col] = df_processed[col].astype('category').cat.codes
 
 status_text.text(' Data preparation complete!')
 progress_bar.progress(100)
 time.sleep(1)
 status_text.empty()
 progress_bar.empty()
 
 # Prediction button
 if st.button(" Make All Predictions!", type="primary"):
 with st.spinner(' Working my magic on all your products...'):
 try:
 # Feature preparation
 if feature_names:
 missing_features = [f for f in feature_names if f not in df_processed.columns]
 if missing_features:
 st.error(f" Oops! I need these columns: {', '.join(missing_features)}")
 
 # Show helpful comparison
 col1, col2 = st.columns(2)
 with col1:
 st.write("** What you have:**")
 st.write(list(df_processed.columns))
 with col2:
 st.write("** What I need:**")
 st.write(feature_names)
 st.stop()
 
 X_pred = df_processed[feature_names]
 else:
 if target_name in df_processed.columns:
 X_pred = df_processed.drop(columns=[target_name])
 else:
 X_pred = df_processed
 
 st.info(f" Making predictions using {X_pred.shape[1]} features...")
 
 # Make predictions with progress
 predictions = model.predict(X_pred)
 
 # Show success
 st.balloons() # Celebration!
 
 st.markdown("""
 <div style="background: linear-gradient(45deg, #4CAF50, #45a049); color: white; 
 padding: 2rem; border-radius: 20px; text-align: center; margin: 2rem 0;">
 <h2> Predictions Complete!</h2>
 <p style="font-size: 1.3rem;">All your products have been analyzed! </p>
 </div>
 """, unsafe_allow_html=True)
 
 # Add predictions to original data
 df_with_predictions = df.copy()
 df_with_predictions['_Predicted_Sales'] = predictions
 
 st.write("### Your Results:")
 st.dataframe(df_with_predictions, use_container_width=True)
 
 # Beautiful statistics
 st.write("### Sales Insights:")
 col1, col2, col3, col4 = st.columns(4)
 
 with col1:
 avg_sales = predictions.mean()
 st.markdown(f"""
 <div class="metric-card">
 <h3 style="color: #4CAF50; margin: 0;"></h3>
 <h4 style="margin: 0.5rem 0;">${avg_sales:.2f}</h4>
 <p style="margin: 0; color: #666;">Average Sales</p>
 </div>
 """, unsafe_allow_html=True)
 
 with col2:
 max_sales = predictions.max()
 st.markdown(f"""
 <div class="metric-card">
 <h3 style="color: #FF6B6B; margin: 0;"></h3>
 <h4 style="margin: 0.5rem 0;">${max_sales:.2f}</h4>
 <p style="margin: 0; color: #666;">Top Performer</p>
 </div>
 """, unsafe_allow_html=True)
 
 with col3:
 min_sales = predictions.min()
 st.markdown(f"""
 <div class="metric-card">
 <h3 style="color: #FFA726; margin: 0;"></h3>
 <h4 style="margin: 0.5rem 0;">${min_sales:.2f}</h4>
 <p style="margin: 0; color: #666;">Needs Attention</p>
 </div>
 """, unsafe_allow_html=True)
 
 with col4:
 total_predicted = predictions.sum()
 st.markdown(f"""
 <div class="metric-card">
 <h3 style="color: #9C27B0; margin: 0;"></h3>
 <h4 style="margin: 0.5rem 0;">${total_predicted:.2f}</h4>
 <p style="margin: 0; color: #666;">Total Predicted</p>
 </div>
 """, unsafe_allow_html=True)
 
 # Visualization
 st.write("### Sales Distribution:")
 st.line_chart(predictions, use_container_width=True)
 
 # Download section
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #667eea;"> Take Your Results With You!</h4>
 <p>Download your predictions and start making informed business decisions!</p>
 </div>
 """, unsafe_allow_html=True)
 
 csv = df_with_predictions.to_csv(index=False)
 st.download_button(
 label=" Download All Predictions (CSV)",
 data=csv,
 file_name='retailvista_sales_predictions.csv',
 mime='text/csv'
 )
 
 except Exception as e:
 st.error(f" Oops! Something went wrong: {str(e)}")
 st.info(" Try checking your data format or refresh the page!")
 
 except Exception as e:
 st.error(f" Trouble reading your file: {str(e)}")
 st.info(" Make sure it's a valid CSV file with the right format!")

 else:
 st.markdown("""
 <div class="feature-card">
 <h4 style="color: #FF9800;"> Ready to upload your CSV?</h4>
 <p style="font-size: 1.1rem; color: #666;">
 Just drag and drop your file above, or click to browse. I'll take care of the rest! 
 Don't worry about the format - I'll help you fix any issues. 
 </p>
 </div>
 """, unsafe_allow_html=True)
 
 # Show example format
 if feature_names:
 st.write("### CSV Format Example:")
 st.info(" Your CSV should contain these columns (don't worry about the exact order):")
 
 example_data = {feature: ["Example values..."] for feature in feature_names[:5]} # Show first 5
 example_df = pd.DataFrame(example_data)
 st.dataframe(example_df, use_container_width=True)
 
 if len(feature_names) > 5:
 with st.expander(f" See all {len(feature_names)} required columns"):
 cols = st.columns(3)
 for i, feature in enumerate(feature_names):
 with cols[i % 3]:
 st.write(f" {feature}")

with tab3:
 st.markdown('<h2 class="section-header"> Learn & Explore</h2>', unsafe_allow_html=True)
 
 # Educational content with personality
 st.markdown("""
 <div class="feature-card">
 <h3 style="color: #667eea;"> Who's This Magic For?</h3>
 <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1rem; margin-top: 1rem;">
 <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); color: white; padding: 1.5rem; border-radius: 15px;">
 <h4> Store Owners</h4>
 <p>Plan inventory like a pro and never run out of bestsellers!</p>
 </div>
 <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); color: white; padding: 1.5rem; border-radius: 15px;">
 <h4> Product Managers</h4>
 <p>Forecast demand for new products before they hit the shelves!</p>
 </div>
 <div style="background: linear-gradient(45deg, #9B59B6, #8E44AD); color: white; padding: 1.5rem; border-radius: 15px;">
 <h4> Business Analysts</h4>
 <p>Make data-driven decisions that actually make sense!</p>
 </div>
 <div style="background: linear-gradient(45deg, #F39C12, #E67E22); color: white; padding: 1.5rem; border-radius: 15px;">
 <h4> Students & Researchers</h4>
 <p>Learn about sales prediction in a fun, interactive way!</p>
 </div>
 </div>
 </div>
 """, unsafe_allow_html=True)
 
 st.markdown("""
 <div class="feature-card">
 <h3 style="color: #764ba2;"> How Does This Magic Work?</h3>
 <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 2rem; border-radius: 15px; margin: 1rem 0;">
 <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 2rem;">
 <div>
 <h4>🧠 Smart AI Brain</h4>
 <p>Trained on real supermarket data to understand sales patterns</p>
 </div>
 <div>
 <h4> Input Features</h4>
 <p>Product and store characteristics that actually matter</p>
 </div>
 <div>
 <h4> Dollar Predictions</h4>
 <p>Get realistic sales forecasts in actual dollars</p>
 </div>
 <div>
 <h4> Learning System</h4>
 <p>Gets smarter with every prediction made</p>
 </div>
 </div>
 </div>
 </div>
 """, unsafe_allow_html=True)
 
 # What affects sales section
 st.markdown("""
 <div class="feature-card">
 <h3 style="color: #E91E63;"> What Makes Sales Tick?</h3>
 <p style="font-size: 1.1rem; color: #666; margin-bottom: 2rem;">
 Ever wondered why some products fly off the shelves while others collect dust? Here's the inside scoop! ‍
 </p>
 </div>
 """, unsafe_allow_html=True)
 
 factors_col1, factors_col2 = st.columns(2)
 
 with factors_col1:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #667eea, #764ba2); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
 <h4> Product Factors</h4>
 <ul style="list-style: none; padding: 0;">
 <li> <strong>Weight:</strong> Heavy ≠ Always Better</li>
 <li>🥑 <strong>Fat Content:</strong> Health trends matter</li>
 <li> <strong>Visibility:</strong> Location, location, location!</li>
 <li> <strong>Category:</strong> Some types just sell more</li>
 </ul>
 </div>
 """, unsafe_allow_html=True)
 
 with factors_col2:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #FF6B6B, #FF8E8E); color: white; padding: 1.5rem; border-radius: 15px; margin: 1rem 0;">
 <h4> Store Factors</h4>
 <ul style="list-style: none; padding: 0;">
 <li> <strong>Size:</strong> Bigger stores = more options</li>
 <li> <strong>Location:</strong> City vs. town dynamics</li>
 <li> <strong>Store Type:</strong> Supermarket vs. grocery</li>
 <li> <strong>Age:</strong> Established stores have loyal customers</li>
 </ul>
 </div>
 """, unsafe_allow_html=True)
 
 # Real scenarios
 st.markdown("""
 <div class="feature-card">
 <h3 style="color: #4CAF50;"> Real-World Success Stories</h3>
 <p style="color: #666; margin-bottom: 2rem;">Here are some ways smart retailers use RetailVista:</p>
 </div>
 """, unsafe_allow_html=True)
 
 scenario_col1, scenario_col2, scenario_col3 = st.columns(3)
 
 with scenario_col1:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #FFB74D, #FFA726); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
 <h4> New Product Launch</h4>
 <div style="font-size: 3rem; margin: 1rem 0;"></div>
 <p><strong>Scenario:</strong> Launching new energy drinks</p>
 <p><strong>Action:</strong> Predict demand before ordering</p>
 <p><strong>Result:</strong> Perfect inventory, no waste!</p>
 </div>
 """, unsafe_allow_html=True)
 
 with scenario_col2:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #4ECDC4, #44A08D); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
 <h4> Holiday Planning</h4>
 <div style="font-size: 3rem; margin: 1rem 0;"></div>
 <p><strong>Scenario:</strong> Christmas shopping season</p>
 <p><strong>Action:</strong> Predict holiday product mix</p>
 <p><strong>Result:</strong> Maximize seasonal profits!</p>
 </div>
 """, unsafe_allow_html=True)
 
 with scenario_col3:
 st.markdown("""
 <div style="background: linear-gradient(45deg, #9B59B6, #8E44AD); color: white; padding: 2rem; border-radius: 15px; text-align: center; margin: 1rem 0;">
 <h4> Smart Pricing</h4>
 <div style="font-size: 3rem; margin: 1rem 0;"></div>
 <p><strong>Scenario:</strong> Testing different price points</p>
 <p><strong>Action:</strong> Compare predicted sales</p>
 <p><strong>Result:</strong> Find the sweet spot price!</p>
 </div>
 """, unsafe_allow_html=True)
 
 # Tips section
 st.markdown("""
 <div class="feature-card">
 <h3 style="color: #FF5722;"> Pro Tips for Best Results</h3>
 <div style="background: linear-gradient(135deg, rgba(255, 87, 34, 0.1), rgba(255, 87, 34, 0.2)); padding: 2rem; border-radius: 15px; margin: 1rem 0;">
 <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(250px, 1fr)); gap: 1.5rem;">
 <div>
 <h4 style="color: #FF5722;"> Do This:</h4>
 <ul style="color: #666;">
 <li>Use realistic product weights</li>
 <li>Choose appropriate store size</li>
 <li>Consider your actual location</li>
 <li>Test multiple scenarios</li>
 </ul>
 </div>
 <div>
 <h4 style="color: #F44336;"> Avoid This:</h4>
 <ul style="color: #666;">
 <li>Extreme or unrealistic values</li>
 <li>Wrong store type selection</li>
 <li>Ignoring market conditions</li>
 <li>Making decisions on one prediction</li>
 </ul>
 </div>
 </div>
 </div>
 </div>
 """, unsafe_allow_html=True)
 
 # Important notes with friendly tone
 st.markdown("""
 <div class="feature-card">
 <h3 style="color: #607D8B;"> A Friendly Reminder</h3>
 <div style="background: linear-gradient(45deg, #90A4AE, #78909C); color: white; padding: 2rem; border-radius: 15px;">
 <p style="font-size: 1.1rem; margin-bottom: 1rem;">
 🤖 <strong>I'm pretty smart, but I'm not psychic!</strong>
 </p>
 <ul style="font-size: 1rem; line-height: 1.6;">
 <li>My predictions are based on historical patterns</li>
 <li>Always combine my insights with your business knowledge</li>
 <li>Market conditions and trends can change everything</li>
 <li>Use me as your advisor, not your only decision-maker</li>
 </ul>
 <p style="margin-top: 1rem; font-style: italic;">
 Think of me as your smart assistant who never gets tired and loves crunching numbers! 
 </p>
 </div>
 </div>
 """, unsafe_allow_html=True)

# Enhanced sidebar with personality
st.sidebar.markdown("""
<div style="background: linear-gradient(180deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)); 
 padding: 2rem; border-radius: 20px; margin: 1rem 0;">
 <h3 style="color: white; text-align: center; margin-bottom: 1.5rem;">🤖 AI Brain Info</h3>
</div>
""", unsafe_allow_html=True)

if feature_names:
 col1, col2 = st.sidebar.columns(2)
 with col1:
 st.metric(" Target", target_name.title())
 with col2:
 st.metric(" Features", len(feature_names))
 
 st.sidebar.metric("🧠 AI Model", "XGBoost Pro")
 
 with st.sidebar.expander(" See All Features"):
 for i, feature in enumerate(feature_names, 1):
 st.write(f"{i}. `{feature}`")

# Footer with personal touch
st.markdown("""
<div style="margin-top: 4rem; padding: 2rem; text-align: center; 
 background: linear-gradient(135deg, rgba(255, 255, 255, 0.1), rgba(255, 255, 255, 0.05)); 
 border-radius: 20px; backdrop-filter: blur(10px);">
 <h4 style="color: white; margin-bottom: 1rem;">Made with and lots of by Leslie Fernando</h4>
 <p style="color: rgba(255, 255, 255, 0.8); margin: 0;">
 RetailVista - Making sales prediction as easy as ordering pizza! 
 </p>
</div>
""", unsafe_allow_html=True)
