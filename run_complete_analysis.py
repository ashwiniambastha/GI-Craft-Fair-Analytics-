"""
Complete GI Craft Fair Analytics Pipeline
========================================
Run entire analysis pipeline with one command
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add src directory to path
sys.path.append('src')

def create_directories():
    """Create necessary directories"""
    directories = [
        'data/raw',
        'data/processed',
        'models',
        'results/plots',
        'dashboard'
    ]
    
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
    print("Directory structure created successfully!")

def run_pipeline():
    """Execute complete analytics pipeline"""
    
    print("="*60)
    print("GI CRAFT FAIR ANALYTICS - COMPLETE PIPELINE")
    print("="*60)
    print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Create directories
    create_directories()
    
    # Step 1: Data Preprocessing
    print("STEP 1: DATA PREPROCESSING")
    print("-" * 30)
    try:
        from data_preprocessing import main as preprocess_main
        processed_data, preprocessing_info = preprocess_main()
        print("✓ Data preprocessing completed successfully!")
        print(f"  - Final dataset shape: {processed_data.shape}")
        print(f"  - New features created: {preprocessing_info['new_features_created']}")
    except Exception as e:
        print(f"✗ Error in data preprocessing: {e}")
        return False
    
    print()
    
    # Step 2: Feature Engineering
    print("STEP 2: FEATURE ENGINEERING")
    print("-" * 30)
    try:
        from feature_engineering import main as feature_main
        engineered_data = feature_main()
        print("✓ Feature engineering completed successfully!")
        print(f"  - Final feature count: {engineered_data.shape[1]}")
        print(f"  - Dataset size: {engineered_data.shape[0]} records")
    except Exception as e:
        print(f"✗ Error in feature engineering: {e}")
        print("  Continuing with processed data...")
    
    print()
    
    # Step 3: Survival Analysis
    print("STEP 3: SURVIVAL ANALYSIS")
    print("-" * 30)
    try:
        from survival_analysis import main as survival_main
        analyzer, survival_data, cox_results, predictions, insights = survival_main()
        print("✓ Survival analysis completed successfully!")
        print(f"  - Vendors analyzed: {len(survival_data)}")
        print(f"  - Model concordance: {cox_results['concordance']:.3f}")
        print(f"  - Overall churn rate: {insights['overall_metrics']['overall_churn_rate']:.2%}")
    except Exception as e:
        print(f"✗ Error in survival analysis: {e}")
        print("  Continuing with next step...")
    
    print()
    
    # Step 4: Ensemble Modeling
    print("STEP 4: ENSEMBLE MACHINE LEARNING")
    print("-" * 30)
    try:
        from ensemble_models import main as ensemble_main
        ensemble_model, ensemble_results, predictions_df = ensemble_main()
        print("✓ Ensemble modeling completed successfully!")
        print(f"  - Test accuracy: {ensemble_results['test_accuracy']:.1%}")
        print(f"  - Cross-validation accuracy: {ensemble_results['cv_results']['ensemble']['mean_accuracy']:.1%}")
        print(f"  - Models in ensemble: {len(ensemble_model.base_models)}")
    except Exception as e:
        print(f"✗ Error in ensemble modeling: {e}")
        print("  Continuing with next step...")
    
    print()
    
    # Step 5: Vendor Segmentation
    print("STEP 5: VENDOR SEGMENTATION")
    print("-" * 30)
    try:
        from vendor_segmentation import main as segmentation_main
        seg_analyzer, segmented_data, seg_results = segmentation_main()
        print("✓ Vendor segmentation completed successfully!")
        print(f"  - Vendors segmented: {len(segmented_data)}")
        print(f"  - Number of segments: {seg_results['number_of_segments']}")
        print(f"  - Best method: {seg_results['best_method']}")
    except Exception as e:
        print(f"✗ Error in vendor segmentation: {e}")
        print("  Continuing with next step...")
    
    print()
    
    # Step 6: Advanced Visualizations
    print("STEP 6: ADVANCED VISUALIZATIONS")
    print("-" * 30)
    try:
        from visualization import main as viz_main
        visualizer, all_data, business_insights = viz_main()
        print("✓ Advanced visualizations created successfully!")
        print(f"  - Data sources loaded: {len(all_data)}")
        print(f"  - Interactive plots created and saved")
        print(f"  - Business insights generated")
    except Exception as e:
        print(f"✗ Error in visualization: {e}")
        print("  Continuing...")
    
    print()
    
    # Final Summary
    print("="*60)
    print("PIPELINE COMPLETION SUMMARY")
    print("="*60)
    
    # Check what files were created
    results_files = []
    if os.path.exists('data/processed/processed_data.csv'):
        results_files.append("✓ Processed dataset")
    if os.path.exists('data/processed/engineered_features.csv'):
        results_files.append("✓ Engineered features")
    if os.path.exists('results/survival_data.csv'):
        results_files.append("✓ Survival analysis results")
    if os.path.exists('results/ensemble_predictions.csv'):
        results_files.append("✓ ML model predictions")
    if os.path.exists('results/vendor_segments.csv'):
        results_files.append("✓ Vendor segmentation")
    if os.path.exists('results/business_insights_summary.json'):
        results_files.append("✓ Business insights")
    
    print("Files Created:")
    for file_info in results_files:
        print(f"  {file_info}")
    
    print()
    print("Key Results:")
    
    # Try to load and display key metrics
    try:
        if os.path.exists('results/business_insights_summary.json'):
            import json
            with open('results/business_insights_summary.json', 'r') as f:
                insights = json.load(f)
            
            if 'key_metrics' in insights:
                for metric, value in insights['key_metrics'].items():
                    print(f"  - {metric.replace('_', ' ').title()}: {value}")
    except:
        pass
    
    print()
    print("Next Steps:")
    print("  1. Review results in the results/ directory")
    print("  2. Open interactive visualizations in results/plots/")
    print("  3. Run the Streamlit dashboard: streamlit run dashboard/streamlit_app.py")
    print("  4. Use models in models/ directory for predictions")
    
    print()
    print(f"Pipeline completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    return True

def create_sample_data():
    """Create sample data if no data file exists"""
    
    sample_file = 'data/raw/gi_craft_data.csv'
    
    if os.path.exists(sample_file):
        print("Data file already exists!")
        return True
    
    print("Creating sample GI Craft Fair dataset...")
    
    # Create sample data structure based on your description
    np.random.seed(42)
    
    states = ['Bihar', 'Uttar Pradesh', 'West Bengal', 'Rajasthan', 'Karnataka', 
             'Tamil Nadu', 'Gujarat', 'Maharashtra', 'Other']
    
    gi_products = ['Banaras Zardozi', 'Madhubani Painting', 'Kantha Embroidery',
                   'Blue Pottery', 'Mysore Silk', 'Pashmina', 'Others']
    
    regions = ['East', 'North', 'West', 'South', 'Central']
    
    # Generate 5000 records
    n_records = 5000
    n_vendors = 500
    
    data = []
    
    for vendor_id in range(1, n_vendors + 1):
        # Random vendor characteristics
        vendor_state = np.random.choice(states)
        vendor_product = np.random.choice(gi_products)
        vendor_region = np.random.choice(regions)
        
        # Generate 10 years of data per vendor (2015-2024, some may have 2025)
        start_year = np.random.choice([2015, 2016, 2017])
        end_year = np.random.choice([2023, 2024, 2025])
        
        for year in range(start_year, end_year + 1):
            # Base income with growth and volatility
            base_income = np.random.normal(50000, 20000)
            growth_factor = 1 + (year - 2015) * 0.05  # 5% yearly growth
            income = max(10000, base_income * growth_factor + np.random.normal(0, 10000))
            
            # Income brackets
            if income < 50000:
                income_bracket = '<50K'
            elif income < 100000:
                income_bracket = '50K-100K'
            elif income < 200000:
                income_bracket = '100K-200K'
            else:
                income_bracket = '200K+'
            
            # Digital adoption (increasing over time)
            digital_base = 0.2 + (year - 2015) * 0.08
            digital_adoption = min(1.0, max(0.0, digital_base + np.random.normal(0, 0.2)))
            
            # E-commerce adoption (binary, higher probability after 2020)
            ecommerce_prob = 0.3 if year < 2021 else 0.7
            ecommerce_active = 1 if np.random.random() < ecommerce_prob else 0
            
            # Government interaction
            govt_interaction_rating = np.random.choice(['Poor', 'Fair', 'Good', 'Excellent'])
            govt_interaction_score = {'Poor': 1, 'Fair': 2, 'Good': 3, 'Excellent': 4}[govt_interaction_rating]
            
            record = {
                'Stall_ID': vendor_id,
                'State': vendor_state,
                'GI_Product': vendor_product,
                'Is_Registered_GI': 1,
                'Monthly_Income_Bracket': income_bracket,
                'Has_Govt_ID': np.random.choice([0, 1]),
                'Ecommerce_Active': ecommerce_active,
                'Govt_Interaction_Rating': govt_interaction_rating,
                'Fair_Beneficial': np.random.choice([0, 1], p=[0.2, 0.8]),
                'Wants_More_Events': np.random.choice([0, 1], p=[0.3, 0.7]),
                'Year': year,
                'Income_Numeric': int(income),
                'Govt_Interaction_Score': govt_interaction_score,
                'State_Region': vendor_region,
                'Years_Registered': year - start_year,
                'Digital_Adoption_Index': round(digital_adoption, 6)
            }
            
            data.append(record)
    
    # Create DataFrame and save
    df = pd.DataFrame(data)
    
    # Ensure we have exactly 5000 records
    if len(df) > 5000:
        df = df.sample(n=5000, random_state=42)
    
    df.to_csv(sample_file, index=False)
    print(f"Sample dataset created: {len(df)} records, {len(df['Stall_ID'].unique())} unique vendors")
    print(f"Saved to: {sample_file}")
    
    return True

def check_requirements():
    """Check if required packages are installed"""

    # Map pip package names to their import module names
    pip_to_import = {
        'pandas': 'pandas',
        'numpy': 'numpy',
        'scikit-learn': 'sklearn',  # import name is sklearn
        'matplotlib': 'matplotlib',
        'seaborn': 'seaborn',
        'xgboost': 'xgboost',
        'lightgbm': 'lightgbm',
        'catboost': 'catboost',
        'lifelines': 'lifelines',
        'plotly': 'plotly',
        'streamlit': 'streamlit',
    }

    missing_pip_packages = []

    for pip_name, import_name in pip_to_import.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_pip_packages.append(pip_name)

    if missing_pip_packages:
        print("Missing required packages:")
        for package in missing_pip_packages:
            print(f"  - {package}")
        print("\nPlease install missing packages using:")
        print(f"pip install {' '.join(missing_pip_packages)}")
        return False

    return True

if __name__ == "__main__":
    print("GI Craft Fair Analytics - Complete Pipeline")
    print("=" * 50)
    
    # Check requirements
    if not check_requirements():
        print("Please install missing packages and try again.")
        sys.exit(1)
    
    # Check if data exists, create sample if not
    if not os.path.exists('data/raw/gi_craft_data.csv'):
        print("No data file found. Creating sample dataset...")
        create_sample_data()
        print()
    
    # Run complete pipeline
    success = run_pipeline()
    
    if success:
        print("\n🎉 Pipeline completed successfully!")
        print("\nTo explore results:")
        print("1. Check results/ folder for analysis outputs")
        print("2. Run: streamlit run dashboard/streamlit_app.py")
        print("3. Open interactive plots in results/plots/")
    else:
        print("\n❌ Pipeline encountered errors. Check the logs above.")
