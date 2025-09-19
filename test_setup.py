"""
Test Setup Script
================
Quick test to verify directory creation and data preprocessing
"""

import os
import sys

# Add src to path
sys.path.append('src')

def main():
    print("Testing project setup...")
    
    # Step 1: Create directories
    directories = [
        'data/raw',
        'data/processed',
        'results/plots',
        'models',
        'config'
    ]
    
    print("Creating directories...")
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")
    
    # Step 2: Test data preprocessing
    print("\nTesting data preprocessing...")
    
    try:
        from data_preprocessing import DataPreprocessor
        import pandas as pd
        
        # Check if data file exists
        if not os.path.exists('data/raw/gi_craft_data.csv'):
            print("No data file found. Please upload your CSV file to data/raw/gi_craft_data.csv")
            return False
        
        # Initialize preprocessor
        preprocessor = DataPreprocessor()
        
        # Load data
        df = preprocessor.load_data()
        
        # Run preprocessing
        processed_df, info = preprocessor.preprocess_pipeline(df)
        
        # Save processed data (this should work now)
        preprocessor.save_processed_data(processed_df)
        
        print("✓ Data preprocessing completed successfully!")
        print(f"  Final shape: {processed_df.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in data preprocessing: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 Setup test passed!")
    else:
        print("\n❌ Setup test failed!")
