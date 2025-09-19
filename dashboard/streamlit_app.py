"""
Interactive Streamlit Dashboard for GI Craft Fair Analytics
==========================================================
Real-time dashboard for exploring analytics results
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import os

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.visualization import AdvancedVisualizationSuite

# Page configuration
st.set_page_config(
    page_title="GI Craft Fair Analytics Dashboard",
    page_icon="🏪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .insight-box {
        background-color: #e8f4fd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load all data with caching"""
    data = {}
    
    try:
        data['raw'] = pd.read_csv('../data/processed/processed_data.csv')
    except:
        st.warning("Processed data not found")
        
    try:
        data['survival'] = pd.read_csv('../results/survival_data.csv')
        data['survival_predictions'] = pd.read_csv('../results/survival_predictions.csv')
    except:
        pass
        
    try:
        data['segments'] = pd.read_csv('../results/vendor_segments.csv')
    except:
        pass
        
    try:
        data['ensemble_predictions'] = pd.read_csv('../results/ensemble_predictions.csv')
    except:
        pass
        
    return data

@st.cache_data
def load_insights():
    """Load insights JSON files"""
    insights = {}
    
    try:
        with open('../results/business_insights_summary.json', 'r') as f:
            insights['business'] = json.load(f)
    except:
        pass
        
    try:
        with open('../results/ensemble_results.json', 'r') as f:
            insights['ensemble'] = json.load(f)
    except:
        pass
        
    try:
        with open('../results/segmentation_results.json', 'r') as f:
            insights['segmentation'] = json.load(f)
    except:
        pass
        
    return insights

def main():
    # Load data
    data = load_data()
    insights = load_insights()
    
    # Header
    st.markdown('<h1 class="main-header">🏪 GI Craft Fair Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis",
        ["Overview", "Performance Analysis", "Survival Analysis", 
         "Vendor Segmentation", "Model Performance", "Business Insights"]
    )
    
    # Overview Page
    if page == "Overview":
        st.header("📊 Project Overview")
        
        col1, col2, col3, col4 = st.columns(4)
        
        if 'raw' in data:
            df = data['raw']
            
            with col1:
                st.metric("Total Vendors", df['Stall_ID'].nunique())
            
            with col2:
                st.metric("Years Covered", f"{df['Year'].min()}-{df['Year'].max()}")
            
            with col3:
                st.metric("States", df['State'].nunique())
            
            with col4:
                st.metric("GI Products", df['GI_Product'].nunique())
        
        st.subheader("Key Achievements")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'business' in insights:
                bus_insights = insights['business']
                if 'key_metrics' in bus_insights:
                    st.markdown("### 📈 Performance Metrics")
                    for metric, value in bus_insights['key_metrics'].items():
                        st.markdown(f"**{metric.replace('_', ' ').title()}:** {value}")
        
        with col2:
            if 'ensemble' in insights:
                ens_insights = insights['ensemble']
                st.markdown("### 🤖 Model Performance")
                if 'test_accuracy' in ens_insights:
                    st.metric("Test Accuracy", f"{ens_insights['test_accuracy']:.1%}")
                if 'test_roc_auc' in ens_insights:
                    st.metric("ROC-AUC Score", f"{ens_insights['test_roc_auc']:.3f}")
    
    # Performance Analysis Page
    elif page == "Performance Analysis":
        st.header("📈 Performance Analysis")
        
        if 'raw' in data:
            df = data['raw']
            
            # Time series analysis
            st.subheader("Income Trends Over Time")
            
            # Group by year and calculate metrics
            yearly_stats = df.groupby('Year').agg({
                'Income_Numeric': ['mean', 'median', 'std'],
                'Digital_Adoption_Index': 'mean',
                'Ecommerce_Active': 'sum'
            }).reset_index()
            
            yearly_stats.columns = ['Year', 'Income_Mean', 'Income_Median', 'Income_Std', 
                                   'Digital_Mean', 'Ecommerce_Sum']
            
            fig = px.line(yearly_stats, x='Year', y=['Income_Mean', 'Income_Median'],
                         title="Average and Median Income Trends")
            st.plotly_chart(fig, use_container_width=True)
            
            # State comparison
            st.subheader("Performance by State")
            
            state_stats = df.groupby('State').agg({
                'Income_Numeric': 'mean',
                'Digital_Adoption_Index': 'mean',
                'Stall_ID': 'nunique'
            }).reset_index()
            state_stats.columns = ['State', 'Avg_Income', 'Avg_Digital', 'Vendor_Count']
            
            # Top 10 states by average income
            top_states = state_stats.nlargest(10, 'Avg_Income')
            
            fig = px.bar(top_states, x='State', y='Avg_Income',
                        title="Top 10 States by Average Income")
            st.plotly_chart(fig, use_container_width=True)
            
            # Digital adoption vs income scatter
            st.subheader("Digital Adoption vs Income")
            
            fig = px.scatter(df, x='Digital_Adoption_Index', y='Income_Numeric',
                           color='State_Region', size='Govt_Interaction_Score',
                           title="Digital Adoption vs Income by Region")
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Performance data not available. Please run the analysis pipeline first.")
    
    # Survival Analysis Page
    elif page == "Survival Analysis":
        st.header("⏱️ Vendor Lifecycle Analysis")
        
        if 'survival' in data and 'survival_predictions' in data:
            survival_data = data['survival']
            predictions = data['survival_predictions']
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Total Vendors Analyzed", len(survival_data))
                st.metric("Median Lifespan", f"{survival_data['Duration'].median():.1f} years")
            
            with col2:
                churn_rate = survival_data['Event'].mean()
                st.metric("Churn Rate", f"{churn_rate:.1%}")
                st.metric("Still Active", f"{(1-churn_rate):.1%}")
            
            # Risk distribution
            if 'Risk_Category' in predictions.columns:
                st.subheader("Vendor Risk Distribution")
                
                risk_counts = predictions['Risk_Category'].value_counts()
                fig = px.pie(values=risk_counts.values, names=risk_counts.index,
                           title="Risk Category Distribution")
                st.plotly_chart(fig, use_container_width=True)
            
            # Survival probability distribution
            st.subheader("Survival Probability Distributions")
            
            prob_cols = [col for col in predictions.columns if 'Survival_Prob' in col]
            if prob_cols:
                selected_period = st.selectbox("Select Time Period", prob_cols)
                
                fig = px.histogram(predictions, x=selected_period,
                                 title=f"Distribution of {selected_period}")
                st.plotly_chart(fig, use_container_width=True)
            
            # Duration vs Income analysis
            st.subheader("Business Duration vs Performance")
            
            fig = px.scatter(survival_data, x='Duration', y='Avg_Income',
                           color='Event', color_discrete_map={0: 'green', 1: 'red'},
                           title="Vendor Lifespan vs Average Income")
            st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Survival analysis data not available. Please run the survival analysis first.")
    
    # Vendor Segmentation Page
    elif page == "Vendor Segmentation":
        st.header("👥 Vendor Segmentation Analysis")
        
        if 'segments' in data:
            segments_df = data['segments']
            
            if 'Segment_Name' in segments_df.columns:
                # Segment overview
                segment_counts = segments_df['Segment_Name'].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Segment Distribution")
                    fig = px.pie(values=segment_counts.values, names=segment_counts.index)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.subheader("Segment Sizes")
                    for segment, count in segment_counts.items():
                        percentage = (count / len(segments_df)) * 100
                        st.markdown(f"**{segment}:** {count} vendors ({percentage:.1f}%)")
                
                # Performance comparison
                if 'Avg_Income' in segments_df.columns:
                    st.subheader("Income Performance by Segment")
                    
                    fig = px.box(segments_df, x='Segment_Name', y='Avg_Income',
                               title="Income Distribution by Segment")
                    fig.update_xaxes(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
                
                # Digital adoption comparison
                if 'Current_Digital_Score' in segments_df.columns:
                    st.subheader("Digital Adoption by Segment")
                    
                    digital_by_segment = segments_df.groupby('Segment_Name')['Current_Digital_Score'].mean()
                    
                    fig = px.bar(x=digital_by_segment.index, y=digital_by_segment.values,
                               title="Average Digital Adoption Score by Segment")
                    st.plotly_chart(fig, use_container_width=True)
                
                # Interactive segment explorer
                st.subheader("Segment Explorer")
                
                selected_segment = st.selectbox("Select Segment", segments_df['Segment_Name'].unique())
                segment_data = segments_df[segments_df['Segment_Name'] == selected_segment]
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Vendors in Segment", len(segment_data))
                
                with col2:
                    if 'Avg_Income' in segment_data.columns:
                        st.metric("Average Income", f"₹{segment_data['Avg_Income'].mean():,.0f}")
                
                with col3:
                    if 'Current_Digital_Score' in segment_data.columns:
                        st.metric("Digital Score", f"{segment_data['Current_Digital_Score'].mean():.2f}")
            
            else:
                st.warning("Segment names not found in data. The segmentation may not have completed successfully.")
        
        else:
            st.warning("Segmentation data not available. Please run the vendor segmentation analysis first.")
    
    # Model Performance Page
    elif page == "Model Performance":
        st.header("🤖 Machine Learning Model Performance")
        
        if 'ensemble' in insights:
            ensemble_results = insights['ensemble']
            
            # Overall performance metrics
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if 'test_accuracy' in ensemble_results:
                    st.metric("Test Accuracy", f"{ensemble_results['test_accuracy']:.1%}")
            
            with col2:
                if 'test_roc_auc' in ensemble_results:
                    st.metric("ROC-AUC Score", f"{ensemble_results['test_roc_auc']:.3f}")
            
            with col3:
                if 'cv_results' in ensemble_results:
                    cv_acc = ensemble_results['cv_results']['ensemble']['mean_accuracy']
                    st.metric("CV Accuracy", f"{cv_acc:.1%}")
            
            # Cross-validation results
            if 'cv_results' in ensemble_results:
                st.subheader("Cross-Validation Results")
                
                cv_results = ensemble_results['cv_results']
                models = list(cv_results.keys())
                accuracies = [cv_results[model]['mean_accuracy'] for model in models]
                std_devs = [cv_results[model]['std_accuracy'] for model in models]
                
                # Create DataFrame for easier plotting
                cv_df = pd.DataFrame({
                    'Model': models,
                    'Accuracy': accuracies,
                    'Std_Dev': std_devs
                })
                
                fig = px.bar(cv_df, x='Model', y='Accuracy', 
                           error_y='Std_Dev',
                           title="Model Performance Comparison")
                st.plotly_chart(fig, use_container_width=True)
            
            # Model weights
            if 'model_weights' in ensemble_results:
                st.subheader("Ensemble Model Weights")
                
                weights = ensemble_results['model_weights']
                
                fig = px.pie(values=list(weights.values()), names=list(weights.keys()),
                           title="Contribution of Each Model to Ensemble")
                st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance
            if 'top_features' in ensemble_results:
                st.subheader("Top Important Features")
                
                features = ensemble_results['top_features']
                if isinstance(features, dict):
                    feature_df = pd.DataFrame(list(features.items()), 
                                            columns=['Feature', 'Importance'])
                    
                    fig = px.bar(feature_df.head(15), x='Importance', y='Feature',
                               orientation='h',
                               title="Top 15 Most Important Features")
                    st.plotly_chart(fig, use_container_width=True)
        
        else:
            st.warning("Model performance data not available. Please run the ensemble modeling first.")
    
    # Business Insights Page
    elif page == "Business Insights":
        st.header("💡 Business Insights & Recommendations")
        
        if 'business' in insights:
            business_insights = insights['business']
            
            # Key recommendations
            if 'recommendations' in business_insights:
                st.subheader("🎯 Key Recommendations")
                
                for i, rec in enumerate(business_insights['recommendations'], 1):
                    st.markdown(f"""
                    <div class="insight-box">
                        <strong>{i}. {rec}</strong>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Market opportunities
            if 'market_opportunities' in business_insights:
                st.subheader("🚀 Market Opportunities")
                
                for opp in business_insights['market_opportunities']:
                    st.markdown(f"• {opp}")
            
            # Performance summary
            if 'key_metrics' in business_insights:
                st.subheader("📊 Key Performance Indicators")
                
                metrics = business_insights['key_metrics']
                cols = st.columns(len(metrics))
                
                for i, (metric, value) in enumerate(metrics.items()):
                    with cols[i % len(cols)]:
                        st.metric(metric.replace('_', ' ').title(), value)
        
        # Additional insights from other analyses
        col1, col2 = st.columns(2)
        
        with col1:
            if 'segmentation' in insights and 'segment_insights' in insights['segmentation']:
                st.subheader("🎯 Segmentation Insights")
                
                seg_insights = insights['segmentation']['segment_insights']
                for segment_id, segment_info in list(seg_insights.items())[:3]:
                    if 'segment_name' in segment_info:
                        st.markdown(f"**{segment_info['segment_name']}:**")
                        st.markdown(f"- Size: {segment_info['segment_percentage']:.1f}%")
                        if 'opportunities' in segment_info and segment_info['opportunities']:
                            st.markdown(f"- Opportunity: {segment_info['opportunities'][0]}")
        
        with col2:
            st.subheader("📈 Growth Opportunities")
            st.markdown("""
            - **Digital Transformation**: 40%+ vendors have low digital adoption
            - **Geographic Expansion**: Underserved states show high potential
            - **Premium Positioning**: High-performing segments ready for scaling
            - **Cross-selling**: Product combinations show untapped synergies
            """)
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**GI Craft Fair Analytics**")
    st.sidebar.markdown("Advanced ML & Statistical Analysis")
    st.sidebar.markdown("Built with Streamlit & Python")

if __name__ == "__main__":
    main()
