"""
GI Craft Fair Data Preprocessing Module
=====================================
Handles data loading, cleaning, and basic transformations
"""

import pandas as pd
import numpy as np
import yaml
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    """Handles all data preprocessing operations"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
    def load_data(self, data_path: str = None) -> pd.DataFrame:
        """Load raw data from CSV"""
        if data_path is None:
            data_path = self.config['data']['raw_data_path']
        
        df = pd.read_csv(data_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def create_target_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create target variables for different analyses"""
        df = df.copy()
        
        # Performance classification (for ensemble models)
        income_threshold_high = df['Income_Numeric'].quantile(0.75)
        income_threshold_low = df['Income_Numeric'].quantile(0.25)
        
        def categorize_performance(income):
            if income >= income_threshold_high:
                return 2  # High performer
            elif income >= income_threshold_low:
                return 1  # Medium performer
            else:
                return 0  # Low performer
        
        df['Performance_Category'] = df['Income_Numeric'].apply(categorize_performance)
        
        # Survival analysis variables
        df['Vendor_Age'] = df['Year'] - df.groupby('Stall_ID')['Year'].transform('min')
        df['Still_Active'] = df.groupby('Stall_ID')['Year'].transform('max') == 2025
        df['Years_Active'] = df.groupby('Stall_ID')['Year'].transform('nunique')
        df['Churned'] = (~df['Still_Active']).astype(int)
        
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values with appropriate strategies"""
        df = df.copy()
        
        # Numerical columns - fill with median
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].median(), inplace=True)
        
        # Categorical columns - fill with mode
        cat_cols = df.select_dtypes(include=['object']).columns
        for col in cat_cols:
            if df[col].isnull().any():
                df[col].fillna(df[col].mode()[0], inplace=True)
        
        return df
    
    def encode_categorical_variables(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode categorical variables"""
        df = df.copy()
        
        # Label encoding for ordinal variables
        ordinal_mappings = {
            'Monthly_Income_Bracket': {'<50K': 0, '50K-100K': 1, '100K-200K': 2, '200K+': 3},
            'Govt_Interaction_Rating': {'Poor': 0, 'Fair': 1, 'Good': 2, 'Excellent': 3}
        }
        
        for col, mapping in ordinal_mappings.items():
            if col in df.columns:
                df[f'{col}_Encoded'] = df[col].map(mapping)
        
        # One-hot encoding for nominal variables
        nominal_cols = ['State', 'GI_Product', 'State_Region']
        for col in nominal_cols:
            if col in df.columns:
                dummies = pd.get_dummies(df[col], prefix=col)
                df = pd.concat([df, dummies], axis=1)
        
        return df
    
    def detect_outliers(self, df: pd.DataFrame, method: str = 'iqr') -> pd.DataFrame:
        """Detect and handle outliers"""
        df = df.copy()
        
        numerical_cols = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score']
        
        if method == 'iqr':
            for col in numerical_cols:
                if col in df.columns:
                    Q1 = df[col].quantile(0.25)
                    Q3 = df[col].quantile(0.75)
                    IQR = Q3 - Q1
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                    
                    # Cap outliers instead of removing
                    df[col] = df[col].clip(lower_bound, upper_bound)
        
        return df
    
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create time-based features"""
        df = df.copy()
        
        # Year-based features
        df['Years_Since_Start'] = df['Year'] - 2015
        df['Is_Recent_Year'] = (df['Year'] >= 2020).astype(int)
        df['COVID_Period'] = ((df['Year'] >= 2020) & (df['Year'] <= 2021)).astype(int)
        df['Post_Digital_Era'] = (df['Year'] >= 2021).astype(int)
        
        return df
    
    def preprocess_pipeline(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
        """Complete preprocessing pipeline"""
        print("Starting data preprocessing...")
        
        # Store original info
        original_shape = df.shape
        preprocessing_info = {
            'original_shape': original_shape,
            'missing_values_before': df.isnull().sum().sum()
        }
        
        # Apply all preprocessing steps
        df = self.create_target_variables(df)
        df = self.handle_missing_values(df)
        df = self.encode_categorical_variables(df)
        df = self.detect_outliers(df)
        df = self.create_time_features(df)
        
        # Final info
        preprocessing_info.update({
            'final_shape': df.shape,
            'missing_values_after': df.isnull().sum().sum(),
            'new_features_created': df.shape[1] - original_shape[1]
        })
        
        print(f"Preprocessing complete:")
        print(f"- Original shape: {original_shape}")
        print(f"- Final shape: {df.shape}")
        print(f"- New features created: {preprocessing_info['new_features_created']}")
        
        return df, preprocessing_info
    
    def save_processed_data(self, df: pd.DataFrame, filename: str = "processed_data.csv"):
        """Save processed data"""
        import os
        
        # Create directory if it doesn't exist
        processed_dir = self.config['data']['processed_data_path']
        os.makedirs(processed_dir, exist_ok=True)
        
        save_path = f"{processed_dir}{filename}"
        df.to_csv(save_path, index=False)
        print(f"Processed data saved to: {save_path}")

def main():
    """Run preprocessing pipeline"""
    preprocessor = DataPreprocessor()
    
    # Load data (you'll need to provide the actual CSV file)
    df = preprocessor.load_data()
    
    # Run preprocessing
    processed_df, info = preprocessor.preprocess_pipeline(df)
    
    # Save processed data
    preprocessor.save_processed_data(processed_df)
    
    return processed_df, info

if __name__ == "__main__":
    processed_data, preprocessing_info = main()
    print("Data preprocessing completed successfully!")
