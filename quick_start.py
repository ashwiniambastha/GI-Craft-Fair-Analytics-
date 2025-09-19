"""
Quick Start Script - Install Packages and Run Analysis
======================================================
This script will install required packages and then run the analysis
"""

import subprocess
import sys
import os

def install_package(package):
    """Install a single package"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package], 
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"✓ {package} installed")
        return True
    except:
        print(f"✗ Failed to install {package}")
        return False

def main():
    print("GI Craft Fair Analytics - Quick Start")
    print("=" * 50)
    
    # Essential packages only
    packages = [
        "scikit-learn",
        "pandas", 
        "numpy",
        "matplotlib",
        "seaborn",
        "xgboost", 
        "lightgbm",
        "catboost",
        "lifelines",
        "plotly",
        "streamlit",
        "pyyaml",
        "joblib"
    ]
    
    print("📦 Installing required packages...")
    
    failed_packages = []
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    if failed_packages:
        print(f"\n❌ Failed to install: {failed_packages}")
        print("Please install them manually:")
        for pkg in failed_packages:
            print(f"  pip install {pkg}")
        return False
    
    print("\n✅ All packages installed successfully!")
    
    # Create directories
    print("\n📁 Creating directories...")
    directories = ['data/raw', 'data/processed', 'results/plots', 'models', 'config']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
        print(f"✓ {directory}")
    
    # Create sample data
    print("\n📊 Creating sample dataset...")
    create_sample_data()
    
    # Run basic analysis
    print("\n🔬 Running data preprocessing...")
    run_basic_analysis()
    
    print("\n🎉 Quick start completed!")
    print("\nNext steps:")
    print("1. Check results in the results/ folder")
    print("2. Run: streamlit run dashboard/streamlit_app.py")
    print("3. For full analysis: python run_complete_analysis.py")

def create_sample_data():
    """Create sample dataset"""
    import pandas as pd
    import numpy as np
    
    np.random.seed(42)
    
    # Sample data creation
    vendors = []
    for vendor_id in range(1, 501):  # 500 vendors
        state = np.random.choice(['Bihar', 'UP', 'WB', 'Rajasthan', 'Karnataka'])
        product = np.random.choice(['Zardozi', 'Pottery', 'Textiles', 'Handicrafts'])
        
        for year in range(2020, 2024):  # 4 years of data
            income = np.random.normal(50000, 15000)
            vendors.append({
                'Stall_ID': vendor_id,
                'State': state,
                'GI_Product': product,
                'Is_Registered_GI': 1,
                'Monthly_Income_Bracket': '<50K' if income < 50000 else '50K+',
                'Has_Govt_ID': np.random.choice([0, 1]),
                'Ecommerce_Active': np.random.choice([0, 1]),
                'Govt_Interaction_Rating': np.random.choice(['Good', 'Fair', 'Excellent']),
                'Fair_Beneficial': 1,
                'Wants_More_Events': 1,
                'Year': year,
                'Income_Numeric': max(10000, int(income)),
                'Govt_Interaction_Score': np.random.randint(1, 5),
                'State_Region': 'North' if state in ['Bihar', 'UP'] else 'East',
                'Years_Registered': year - 2019,
                'Digital_Adoption_Index': np.random.random()
            })
    
    df = pd.DataFrame(vendors)
    df.to_csv('data/raw/gi_craft_data.csv', index=False)
    print(f"✓ Created dataset with {len(df)} records")

def run_basic_analysis():
    """Run basic data analysis"""
    try:
        import pandas as pd
        import numpy as np
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        import matplotlib.pyplot as plt
        import os
        
        # Load data
        df = pd.read_csv('data/raw/gi_craft_data.csv')
        print(f"✓ Loaded {len(df)} records")
        
        # Basic preprocessing
        df['Performance_Category'] = pd.cut(df['Income_Numeric'], 
                                          bins=3, labels=[0, 1, 2])
        
        # Encode categorical variables
        le_state = LabelEncoder()
        le_product = LabelEncoder()
        
        df['State_Encoded'] = le_state.fit_transform(df['State'])
        df['Product_Encoded'] = le_product.fit_transform(df['GI_Product'])
        
        # Select features
        features = ['State_Encoded', 'Product_Encoded', 'Has_Govt_ID', 
                   'Ecommerce_Active', 'Govt_Interaction_Score', 
                   'Years_Registered', 'Digital_Adoption_Index']
        
        X = df[features].fillna(0)
        y = df['Performance_Category'].fillna(1)
        
        # Train model
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        
        accuracy = model.score(X_test, y_test)
        print(f"✓ Model accuracy: {accuracy:.2%}")
        
        # Create basic visualization
        feature_importance = pd.DataFrame({
            'feature': features,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        plt.figure(figsize=(10, 6))
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.title('Feature Importance')
        plt.tight_layout()
        plt.savefig('results/plots/feature_importance.png', dpi=150, bbox_inches='tight')
        plt.close()
        
        print("✓ Basic analysis completed")
        
        # Save processed data
        df.to_csv('data/processed/processed_data.csv', index=False)
        
        # Save results summary
        summary = {
            'total_vendors': df['Stall_ID'].nunique(),
            'total_records': len(df),
            'model_accuracy': f"{accuracy:.2%}",
            'avg_income': f"₹{df['Income_Numeric'].mean():,.0f}",
            'top_state': df['State'].mode()[0],
            'digital_adoption_rate': f"{df['Digital_Adoption_Index'].mean():.1%}"
        }
        
        print("\n📊 Analysis Summary:")
        for key, value in summary.items():
            print(f"  {key.replace('_', ' ').title()}: {value}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error in analysis: {e}")
        return False

if __name__ == "__main__":
    main()
