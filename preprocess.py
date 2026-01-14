"""
Shared preprocessing module for RetailVista.
Ensures consistent preprocessing between training and inference.
"""

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder


def get_feature_columns():
    """
    Define the features used in the model.
    Excludes target variable and ID columns.
    """
    numeric_features = [
        'Item_Weight',
        'Item_Visibility',
        'Item_MRP',
        'Outlet_Establishment_Year'
    ]
    
    categorical_features = [
        'Item_Fat_Content',
        'Item_Type',
        'Outlet_Size',
        'Outlet_Location_Type',
        'Outlet_Type'
    ]
    
    return numeric_features, categorical_features


def create_preprocessor():
    """
    Create a ColumnTransformer for preprocessing.
    - Numeric features: StandardScaler
    - Categorical features: OneHotEncoder with handle_unknown="ignore"
    """
    numeric_features, categorical_features = get_feature_columns()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), categorical_features)
        ],
        remainder='drop'
    )
    
    return preprocessor


def prepare_features(df):
    """
    Prepare features from raw dataframe.
    Returns only the columns needed for the model.
    """
    numeric_features, categorical_features = get_feature_columns()
    all_features = numeric_features + categorical_features
    
    # Select only required columns
    df_features = df[all_features].copy()
    
    # Handle missing values
    df_features = df_features.fillna(df_features.median(numeric_only=True))
    
    return df_features
