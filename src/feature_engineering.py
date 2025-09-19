"""
Advanced Feature Engineering Pipeline
===================================
Creates 200+ features with automated selection
"""

import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from boruta import BorutaPy
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """Advanced feature engineering with automated selection"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        self.scaler = StandardScaler()
        self.feature_names = []
        
    def create_lag_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create lag features for time series data"""
        df = df.copy()
        df = df.sort_values(['Stall_ID', 'Year'])
        
        lag_periods = self.config['feature_engineering']['lag_periods']
        numerical_cols = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score']
        
        for col in numerical_cols:
            if col in df.columns:
                for lag in lag_periods:
                    lag_col = f'{col}_lag_{lag}'
                    df[lag_col] = df.groupby('Stall_ID')[col].shift(lag)
                    self.feature_names.append(lag_col)
        
        print(f"Created {len(lag_periods) * len(numerical_cols)} lag features")
        return df
    
    def create_rolling_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create rolling window statistical features"""
        df = df.copy()
        df = df.sort_values(['Stall_ID', 'Year'])
        
        windows = self.config['feature_engineering']['rolling_windows']
        numerical_cols = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score']
        
        for col in numerical_cols:
            if col in df.columns:
                for window in windows:
                    # Rolling statistics
                    df[f'{col}_rolling_mean_{window}'] = df.groupby('Stall_ID')[col].rolling(window).mean().reset_index(0, drop=True)
                    df[f'{col}_rolling_std_{window}'] = df.groupby('Stall_ID')[col].rolling(window).std().reset_index(0, drop=True)
                    df[f'{col}_rolling_min_{window}'] = df.groupby('Stall_ID')[col].rolling(window).min().reset_index(0, drop=True)
                    df[f'{col}_rolling_max_{window}'] = df.groupby('Stall_ID')[col].rolling(window).max().reset_index(0, drop=True)
                    
                    # Add to feature names
                    for stat in ['mean', 'std', 'min', 'max']:
                        self.feature_names.append(f'{col}_rolling_{stat}_{window}')
        
        print(f"Created {len(windows) * len(numerical_cols) * 4} rolling features")
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between important variables"""
        df = df.copy()
        
        # Key numerical features for interactions
        key_features = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score', 
                       'Years_Since_Start', 'Years_Active']
        
        # Create pairwise interactions
        for i, feat1 in enumerate(key_features):
            for feat2 in key_features[i+1:]:
                if feat1 in df.columns and feat2 in df.columns:
                    # Multiplication interaction
                    interaction_name = f'{feat1}_x_{feat2}'
                    df[interaction_name] = df[feat1] * df[feat2]
                    self.feature_names.append(interaction_name)
                    
                    # Ratio interaction (avoid division by zero)
                    if df[feat2].min() > 0:
                        ratio_name = f'{feat1}_div_{feat2}'
                        df[ratio_name] = df[feat1] / (df[feat2] + 1e-8)
                        self.feature_names.append(ratio_name)
        
        print(f"Created {len(self.feature_names[-20:])} interaction features")
        return df
    
    def create_aggregation_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregation features by groups"""
        df = df.copy()
        
        # State-level aggregations
        state_aggs = df.groupby('State').agg({
            'Income_Numeric': ['mean', 'median', 'std', 'min', 'max'],
            'Digital_Adoption_Index': ['mean', 'std'],
            'Govt_Interaction_Score': ['mean', 'std']
        }).round(4)
        
        # Flatten column names
        state_aggs.columns = [f'State_{col[0]}_{col[1]}' for col in state_aggs.columns]
        state_aggs = state_aggs.add_prefix('State_')
        state_aggs.reset_index(inplace=True)
        
        # Merge back to main dataframe
        df = df.merge(state_aggs, on='State', how='left')
        
        # GI Product aggregations
        product_aggs = df.groupby('GI_Product').agg({
            'Income_Numeric': ['mean', 'std', 'count'],
            'Digital_Adoption_Index': ['mean', 'std']
        }).round(4)
        
        product_aggs.columns = [f'Product_{col[0]}_{col[1]}' for col in product_aggs.columns]
        product_aggs.reset_index(inplace=True)
        
        df = df.merge(product_aggs, on='GI_Product', how='left')
        
        # Year-level aggregations
        year_aggs = df.groupby('Year').agg({
            'Income_Numeric': ['mean', 'std'],
            'Digital_Adoption_Index': ['mean', 'std']
        }).round(4)
        
        year_aggs.columns = [f'Year_{col[0]}_{col[1]}' for col in year_aggs.columns]
        year_aggs.reset_index(inplace=True)
        
        df = df.merge(year_aggs, on='Year', how='left')
        
        print("Created aggregation features by State, Product, and Year")
        return df
    
    def create_ranking_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create ranking and percentile features"""
        df = df.copy()
        
        # Overall rankings
        df['Income_Rank_Overall'] = df['Income_Numeric'].rank(pct=True)
        df['Digital_Rank_Overall'] = df['Digital_Adoption_Index'].rank(pct=True)
        
        # State-wise rankings
        df['Income_Rank_State'] = df.groupby('State')['Income_Numeric'].rank(pct=True)
        df['Digital_Rank_State'] = df.groupby('State')['Digital_Adoption_Index'].rank(pct=True)
        
        # Product-wise rankings
        df['Income_Rank_Product'] = df.groupby('GI_Product')['Income_Numeric'].rank(pct=True)
        df['Digital_Rank_Product'] = df.groupby('GI_Product')['Digital_Adoption_Index'].rank(pct=True)
        
        # Year-wise rankings
        df['Income_Rank_Year'] = df.groupby('Year')['Income_Numeric'].rank(pct=True)
        
        ranking_features = ['Income_Rank_Overall', 'Digital_Rank_Overall', 'Income_Rank_State', 
                           'Digital_Rank_State', 'Income_Rank_Product', 'Digital_Rank_Product', 
                           'Income_Rank_Year']
        self.feature_names.extend(ranking_features)
        
        print(f"Created {len(ranking_features)} ranking features")
        return df
    
    def create_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create change/growth rate features"""
        df = df.copy()
        df = df.sort_values(['Stall_ID', 'Year'])
        
        # Year-over-year changes
        change_cols = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score']
        
        for col in change_cols:
            if col in df.columns:
                # Absolute change
                df[f'{col}_change'] = df.groupby('Stall_ID')[col].diff()
                
                # Percentage change
                df[f'{col}_pct_change'] = df.groupby('Stall_ID')[col].pct_change()
                
                # Cumulative change from first year
                df[f'{col}_cumulative_change'] = df.groupby('Stall_ID')[col].apply(
                    lambda x: x - x.iloc[0] if len(x) > 0 else 0
                )
                
                self.feature_names.extend([f'{col}_change', f'{col}_pct_change', f'{col}_cumulative_change'])
        
        print(f"Created {len(change_cols) * 3} change features")
        return df
    
    def create_polynomial_features(self, df: pd.DataFrame, degree: int = 2) -> pd.DataFrame:
        """Create polynomial features for key variables"""
        df = df.copy()
        
        # Select key features for polynomial expansion
        poly_features = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score']
        poly_features = [f for f in poly_features if f in df.columns]
        
        if len(poly_features) > 0:
            poly = PolynomialFeatures(degree=degree, include_bias=False, interaction_only=False)
            
            # Fit and transform
            poly_data = poly.fit_transform(df[poly_features])
            poly_feature_names = poly.get_feature_names_out(poly_features)
            
            # Create DataFrame with polynomial features
            poly_df = pd.DataFrame(poly_data, columns=poly_feature_names, index=df.index)
            
            # Remove original features (to avoid duplication)
            new_features = [col for col in poly_df.columns if col not in poly_features]
            
            # Add to main dataframe
            for col in new_features:
                df[col] = poly_df[col]
                self.feature_names.append(col)
            
            print(f"Created {len(new_features)} polynomial features")
        
        return df
    
    def select_features_boruta(self, X: pd.DataFrame, y: pd.Series, max_features: int = 100) -> list:
        """Feature selection using Boruta algorithm"""
        print("Running Boruta feature selection...")
        
        # Initialize Boruta
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        boruta_selector = BorutaPy(rf, n_estimators='auto', random_state=42, max_iter=50)
        
        # Fit Boruta
        boruta_selector.fit(X.values, y.values)
        
        # Get selected features
        selected_features = X.columns[boruta_selector.support_].tolist()
        
        # If too many features, take top ones by ranking
        if len(selected_features) > max_features:
            feature_ranking = boruta_selector.ranking_
            feature_importance = list(zip(X.columns, feature_ranking))
            feature_importance.sort(key=lambda x: x[1])
            selected_features = [feat[0] for feat in feature_importance[:max_features]]
        
        print(f"Boruta selected {len(selected_features)} features")
        return selected_features
    
    def select_features_statistical(self, X: pd.DataFrame, y: pd.Series, k: int = 50) -> list:
        """Statistical feature selection using mutual information"""
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        selector.fit(X, y)
        
        selected_features = X.columns[selector.get_support()].tolist()
        print(f"Statistical selection chose {len(selected_features)} features")
        return selected_features
    
    def feature_engineering_pipeline(self, df: pd.DataFrame, target_col: str = 'Performance_Category') -> pd.DataFrame:
        """Complete feature engineering pipeline"""
        print("Starting advanced feature engineering...")
        
        original_features = df.shape[1]
        
        # Apply all feature engineering steps
        df = self.create_lag_features(df)
        df = self.create_rolling_features(df)
        df = self.create_interaction_features(df)
        df = self.create_aggregation_features(df)
        df = self.create_ranking_features(df)
        df = self.create_change_features(df)
        df = self.create_polynomial_features(df)
        
        # Handle any remaining NaN values
        df = df.fillna(df.median(numeric_only=True))
        df = df.fillna(0)  # For any remaining NaNs
        
        print(f"Feature engineering complete:")
        print(f"- Original features: {original_features}")
        print(f"- Final features: {df.shape[1]}")
        print(f"- New features created: {df.shape[1] - original_features}")
        
        return df
    
    def save_feature_names(self, filename: str = "feature_names.txt"):
        """Save feature names to file"""
        import os
        
        # Create results directory if it doesn't exist
        os.makedirs("results", exist_ok=True)
        
        with open(f"results/{filename}", 'w') as f:
            for feature in self.feature_names:
                f.write(f"{feature}\n")
        print(f"Feature names saved to results/{filename}")

def main():
    """Run feature engineering pipeline"""
    import os
    
    # Create necessary directories
    os.makedirs("data/processed", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # Load processed data
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
    except FileNotFoundError:
        print("Error: processed_data.csv not found. Please run data preprocessing first.")
        return None
    
    # Initialize feature engineer
    feature_engineer = AdvancedFeatureEngineer()
    
    # Run feature engineering
    engineered_df = feature_engineer.feature_engineering_pipeline(df)
    
    # Save engineered data
    engineered_df.to_csv("data/processed/engineered_features.csv", index=False)
    feature_engineer.save_feature_names()
    
    print(f"Final dataset shape: {engineered_df.shape}")
    return engineered_df

if __name__ == "__main__":
    engineered_data = main()
    print("Feature engineering completed successfully!")
