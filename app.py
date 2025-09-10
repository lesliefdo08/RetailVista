import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title='RetailVista: Sales Prediction', layout='wide')
st.title('RetailVista: Supermarket Sales Prediction')
st.markdown('**Author: Leslie Fernando**')

st.write('Upload a CSV file with sales data to get predictions. The model was trained on supermarket sales data.')

model_path = os.path.join('model', 'sales_predictor.joblib')

@st.cache_resource
def load_model():
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        return None

model = load_model()

uploaded_file = st.file_uploader('Choose a CSV file for prediction', type='csv')

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write('Input Data:', data.head())
    if model is not None:
        # Select features used in training (update as per your notebook)
        features = [col for col in data.columns if col not in ['Total', 'Date']]
        if 'Date' in data.columns:
            data['Date'] = pd.to_datetime(data['Date'])
            data['DayOfWeek'] = data['Date'].dt.dayofweek
            data['Month'] = data['Date'].dt.month
            features += ['DayOfWeek', 'Month']
        X_pred = data[features].select_dtypes(include=['number'])
        predictions = model.predict(X_pred)
        data['Predicted_Sales'] = predictions
        st.write('Predictions:', data[['Predicted_Sales']].head())
        st.line_chart(data['Predicted_Sales'])
        st.download_button('Download Predictions as CSV', data.to_csv(index=False), file_name='predictions.csv')
    else:
        st.warning('Trained model not found. Please train and save the model first.')
else:
    st.info('Awaiting CSV file upload.')
