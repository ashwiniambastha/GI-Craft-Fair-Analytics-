"""
Vendor Lifecycle Survival Analysis
=================================
Predicts vendor dropout and lifecycle patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import KaplanMeierFitter, CoxPHFitter, LogNormalAFTFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test
from lifelines.plotting import plot_lifetimes
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

class VendorSurvivalAnalyzer:
    """Advanced survival analysis for vendor lifecycle prediction"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.kmf = KaplanMeierFitter()
        self.cph = CoxPHFitter()
        self.aft = LogNormalAFTFitter()
        self.survival_data = None
        
    def prepare_survival_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare data for survival analysis"""
        print("Preparing survival analysis data...")
        
        # Create vendor-level dataset
        vendor_data = []
        
        for vendor_id in df['Stall_ID'].unique():
            vendor_df = df[df['Stall_ID'] == vendor_id].sort_values('Year')
            
            # Basic info
            first_year = vendor_df['Year'].min()
            last_year = vendor_df['Year'].max()
            duration = last_year - first_year + 1
            
            # Event indicator (churned = 1 if last year < 2025)
            event = 1 if last_year < 2025 else 0
            
            # Features (use average or last known values)
            avg_income = vendor_df['Income_Numeric'].mean()
            last_digital_adoption = vendor_df['Digital_Adoption_Index'].iloc[-1]
            avg_govt_interaction = vendor_df['Govt_Interaction_Score'].mean()
            state = vendor_df['State'].iloc[0]
            product = vendor_df['GI_Product'].iloc[0]
            
            # Performance metrics
            income_growth = (vendor_df['Income_Numeric'].iloc[-1] - vendor_df['Income_Numeric'].iloc[0]) / vendor_df['Income_Numeric'].iloc[0] if len(vendor_df) > 1 else 0
            income_volatility = vendor_df['Income_Numeric'].std()
            digital_adoption_change = vendor_df['Digital_Adoption_Index'].iloc[-1] - vendor_df['Digital_Adoption_Index'].iloc[0] if len(vendor_df) > 1 else 0
            
            vendor_data.append({
                'Vendor_ID': vendor_id,
                'Duration': duration,
                'Event': event,
                'First_Year': first_year,
                'Last_Year': last_year,
                'Avg_Income': avg_income,
                'Income_Growth': income_growth,
                'Income_Volatility': income_volatility,
                'Digital_Adoption_Last': last_digital_adoption,
                'Digital_Adoption_Change': digital_adoption_change,
                'Avg_Govt_Interaction': avg_govt_interaction,
                'State': state,
                'GI_Product': product,
                'Has_Govt_ID': vendor_df['Has_Govt_ID'].iloc[-1],
                'Ecommerce_Active': vendor_df['Ecommerce_Active'].iloc[-1],
                'Records_Count': len(vendor_df)
            })
        
        survival_df = pd.DataFrame(vendor_data)
        self.survival_data = survival_df
        
        print(f"Survival data prepared: {len(survival_df)} vendors")
        print(f"Events (churned): {survival_df['Event'].sum()}")
        print(f"Censored (still active): {len(survival_df) - survival_df['Event'].sum()}")
        
        return survival_df
    
    def kaplan_meier_analysis(self, df: pd.DataFrame) -> dict:
        """Perform Kaplan-Meier survival analysis"""
        print("Performing Kaplan-Meier analysis...")
        
        results = {}
        
        # Overall survival curve
        self.kmf.fit(df['Duration'], df['Event'], label='All Vendors')
        results['overall_survival'] = {
            'survival_function': self.kmf.survival_function_,
            'median_survival': self.kmf.median_survival_time_,
            'confidence_interval': self.kmf.confidence_interval_
        }
        
        # Plot overall survival
        plt.figure(figsize=(12, 8))
        self.kmf.plot_survival_function()
        plt.title('Vendor Survival Curve - Overall')
        plt.xlabel('Years in Business')
        plt.ylabel('Probability of Survival')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots/overall_survival_curve.png', dpi=300)
        plt.show()
        
        # Survival by categories
        categories = ['State', 'GI_Product', 'Has_Govt_ID', 'Ecommerce_Active']
        
        for category in categories:
            if category in df.columns:
                plt.figure(figsize=(12, 8))
                
                unique_values = df[category].unique()
                survival_curves = {}
                
                for value in unique_values[:5]:  # Limit to top 5 categories for readability
                    mask = df[category] == value
                    if mask.sum() >= 10:  # Minimum sample size
                        self.kmf.fit(df[mask]['Duration'], df[mask]['Event'], label=f'{category}: {value}')
                        self.kmf.plot_survival_function()
                        
                        survival_curves[value] = {
                            'survival_function': self.kmf.survival_function_.copy(),
                            'median_survival': self.kmf.median_survival_time_
                        }
                
                plt.title(f'Vendor Survival by {category}')
                plt.xlabel('Years in Business')
                plt.ylabel('Probability of Survival')
                plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f'results/plots/survival_by_{category.lower()}.png', dpi=300, bbox_inches='tight')
                plt.show()
                
                results[f'survival_by_{category}'] = survival_curves
        
        return results
    
    def cox_regression_analysis(self, df: pd.DataFrame) -> dict:
        """Perform Cox Proportional Hazards regression"""
        print("Performing Cox regression analysis...")
        
        # Prepare features for Cox regression
        cox_features = [
            'Avg_Income', 'Income_Growth', 'Income_Volatility',
            'Digital_Adoption_Last', 'Digital_Adoption_Change',
            'Avg_Govt_Interaction', 'Has_Govt_ID', 'Ecommerce_Active',
            'Records_Count'
        ]
        
        cox_df = df[['Duration', 'Event'] + cox_features].copy()
        
        # Handle any missing values
        cox_df = cox_df.fillna(cox_df.median(numeric_only=True))
        
        # Standardize continuous variables
        continuous_vars = ['Avg_Income', 'Income_Growth', 'Income_Volatility',
                          'Digital_Adoption_Last', 'Digital_Adoption_Change',
                          'Avg_Govt_Interaction']
        
        for var in continuous_vars:
            if var in cox_df.columns:
                cox_df[var] = (cox_df[var] - cox_df[var].mean()) / cox_df[var].std()
        
        # Fit Cox model
        self.cph.fit(cox_df, duration_col='Duration', event_col='Event')
        
        # Results
        results = {
            'summary': self.cph.summary,
            'hazard_ratios': np.exp(self.cph.params_),
            'concordance': self.cph.concordance_index_,
            'log_likelihood': self.cph.log_likelihood_,
            'AIC': self.cph.AIC_
        }
        
        # Plot hazard ratios
        plt.figure(figsize=(12, 8))
        hazard_ratios = np.exp(self.cph.params_)
        confidence_intervals = np.exp(self.cph.confidence_intervals_)
        
        y_pos = np.arange(len(hazard_ratios))
        plt.errorbar(hazard_ratios.values, y_pos, 
                    xerr=[hazard_ratios.values - confidence_intervals.iloc[:, 0].values,
                          confidence_intervals.iloc[:, 1].values - hazard_ratios.values],
                    fmt='o', capsize=5)
        
        plt.axvline(x=1, color='r', linestyle='--', alpha=0.7)
        plt.yticks(y_pos, hazard_ratios.index)
        plt.xlabel('Hazard Ratio (95% CI)')
        plt.title('Cox Regression Hazard Ratios')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('results/plots/cox_hazard_ratios.png', dpi=300)
        plt.show()
        
        # Feature importance plot
        plt.figure(figsize=(12, 8))
        feature_importance = np.abs(self.cph.params_).sort_values(ascending=True)
        plt.barh(range(len(feature_importance)), feature_importance.values)
        plt.yticks(range(len(feature_importance)), feature_importance.index)
        plt.xlabel('|Coefficient|')
        plt.title('Feature Importance in Cox Regression')
        plt.tight_layout()
        plt.savefig('results/plots/cox_feature_importance.png', dpi=300)
        plt.show()
        
        return results
    
    def predict_survival_probability(self, df: pd.DataFrame, time_points: list = [1, 2, 3, 5]) -> pd.DataFrame:
        """Predict survival probabilities for different time points"""
        print("Calculating survival probabilities...")
        
        # Prepare the same features used in Cox regression
        cox_features = [
            'Avg_Income', 'Income_Growth', 'Income_Volatility',
            'Digital_Adoption_Last', 'Digital_Adoption_Change',
            'Avg_Govt_Interaction', 'Has_Govt_ID', 'Ecommerce_Active',
            'Records_Count'
        ]
        
        prediction_df = df[cox_features].copy()
        prediction_df = prediction_df.fillna(prediction_df.median(numeric_only=True))
        
        # Standardize (using same parameters as training)
        continuous_vars = ['Avg_Income', 'Income_Growth', 'Income_Volatility',
                          'Digital_Adoption_Last', 'Digital_Adoption_Change',
                          'Avg_Govt_Interaction']
        
        for var in continuous_vars:
            if var in prediction_df.columns:
                prediction_df[var] = (prediction_df[var] - prediction_df[var].mean()) / prediction_df[var].std()
        
        # Predict survival probabilities
        survival_predictions = {}
        for time_point in time_points:
            survival_probs = self.cph.predict_survival_function(prediction_df, times=[time_point])
            survival_predictions[f'Survival_Prob_{time_point}Y'] = survival_probs.iloc[0].values
        
        # Create results dataframe
        results_df = df[['Vendor_ID', 'Duration', 'Event']].copy()
        for time_point, probs in survival_predictions.items():
            results_df[time_point] = probs
        
        # Risk categorization
        results_df['Risk_Category'] = pd.cut(
            results_df['Survival_Prob_2Y'],
            bins=[0, 0.5, 0.8, 1.0],
            labels=['High Risk', 'Medium Risk', 'Low Risk']
        )
        
        return results_df
    
    def generate_survival_insights(self, df: pd.DataFrame, cox_results: dict) -> dict:
        """Generate business insights from survival analysis"""
        print("Generating survival insights...")
        
        insights = {}
        
        # Overall metrics
        median_survival = df['Duration'].median()
        churn_rate = df['Event'].mean()
        
        insights['overall_metrics'] = {
            'median_vendor_lifespan': median_survival,
            'overall_churn_rate': churn_rate,
            'total_vendors_analyzed': len(df)
        }
        
        # Risk factors (from Cox regression)
        hazard_ratios = np.exp(cox_results['summary']['coef'])
        top_risk_factors = hazard_ratios.sort_values(ascending=False).head(5)
        protective_factors = hazard_ratios.sort_values(ascending=True).head(5)
        
        insights['risk_factors'] = {
            'top_risk_factors': top_risk_factors.to_dict(),
            'protective_factors': protective_factors.to_dict()
        }
        
        # Segment analysis
        high_risk_vendors = df[df['Event'] == 1]
        successful_vendors = df[df['Duration'] >= df['Duration'].quantile(0.75)]
        
        insights['segment_analysis'] = {
            'high_risk_characteristics': {
                'avg_income': high_risk_vendors['Avg_Income'].mean(),
                'avg_digital_adoption': high_risk_vendors['Digital_Adoption_Last'].mean(),
                'govt_id_percentage': high_risk_vendors['Has_Govt_ID'].mean()
            },
            'successful_vendor_characteristics': {
                'avg_income': successful_vendors['Avg_Income'].mean(),
                'avg_digital_adoption': successful_vendors['Digital_Adoption_Last'].mean(),
                'govt_id_percentage': successful_vendors['Has_Govt_ID'].mean()
            }
        }
        
        return insights
    
    def save_results(self, survival_data: pd.DataFrame, cox_results: dict, 
                    predictions: pd.DataFrame, insights: dict):
        """Save all survival analysis results"""
        
        # Save datasets
        survival_data.to_csv('results/survival_data.csv', index=False)
        predictions.to_csv('results/survival_predictions.csv', index=False)
        
        # Save Cox model
        joblib.dump(self.cph, 'models/cox_model.pkl')
        
        # Save insights
        import json
        with open('results/survival_insights.json', 'w') as f:
            json.dump(insights, f, indent=2, default=str)
        
        # Save Cox results summary
        cox_results['summary'].to_csv('results/cox_regression_summary.csv')
        
        print("All survival analysis results saved!")

def main():
    """Run complete survival analysis pipeline"""
    
    # Load processed data
    df = pd.read_csv("data/processed/processed_data.csv")
    
    # Initialize analyzer
    analyzer = VendorSurvivalAnalyzer()
    
    # Prepare survival data
    survival_data = analyzer.prepare_survival_data(df)
    
    # Kaplan-Meier analysis
    km_results = analyzer.kaplan_meier_analysis(survival_data)
    
    # Cox regression analysis
    cox_results = analyzer.cox_regression_analysis(survival_data)
    
    # Survival predictions
    predictions = analyzer.predict_survival_probability(survival_data)
    
    # Generate insights
    insights = analyzer.generate_survival_insights(survival_data, cox_results)
    
    # Save all results
    analyzer.save_results(survival_data, cox_results, predictions, insights)
    
    print("Survival analysis completed successfully!")
    print(f"Model concordance: {cox_results['concordance']:.3f}")
    print(f"Overall churn rate: {insights['overall_metrics']['overall_churn_rate']:.2%}")
    
    return analyzer, survival_data, cox_results, predictions, insights

if __name__ == "__main__":
    analyzer, data, results, preds, insights = main()
