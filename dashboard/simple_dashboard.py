"""
Simple Streamlit Dashboard for GI Craft Fair Analytics
=====================================================
Standalone dashboard without dependencies on other modules
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
import seaborn as sns

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
</style>
""", unsafe_allow_html=True)

@st.cache_data
def create_sample_data():
    """Create comprehensive sample dataset"""
    np.random.seed(42)
    
    states = ['Bihar', 'Uttar Pradesh', 'West Bengal', 'Rajasthan', 'Karnataka', 
              'Tamil Nadu', 'Gujarat', 'Maharashtra', 'Punjab', 'Haryana']
    
    gi_products = ['Banaras Zardozi', 'Madhubani Painting', 'Kantha Embroidery',
                   'Blue Pottery', 'Mysore Silk', 'Pashmina', 'Dhokra Art', 
                   'Warli Painting', 'Phulkari', 'Others']
    
    regions = ['East', 'North', 'West', 'South', 'Central']
    
    data = []
    
    for vendor_id in range(1, 501):  # 500 vendors
        vendor_state = np.random.choice(states)
        vendor_product = np.random.choice(gi_products)
        vendor_region = np.random.choice(regions)
        
        # Base characteristics
        base_income = np.random.normal(60000, 25000)
        digital_aptitude = np.random.random()
        govt_engagement = np.random.random()
        
        for year in range(2020, 2025):  # 2020-2024
            # Income with growth and volatility
            yearly_growth = 1 + (year - 2020) * 0.05
            income = max(15000, base_income * yearly_growth + np.random.normal(0, 15000))
            
            # Digital adoption (increasing over time)
            digital_boost = (year - 2020) * 0.15
            digital_score = min(1.0, digital_aptitude + digital_boost + np.random.normal(0, 0.1))
            
            # E-commerce adoption
            ecommerce_prob = 0.3 + digital_score * 0.5
            ecommerce_active = 1 if np.random.random() < ecommerce_prob else 0
            
            # Government interaction
            govt_score = min(4, max(1, int(govt_engagement * 4) + np.random.randint(-1, 2)))
            govt_ratings = {1: 'Poor', 2: 'Fair', 3: 'Good', 4: 'Excellent'}
            
            # Performance category
            income_percentile = np.percentile([15000, 200000], [33, 67])
            if income < income_percentile[0]:
                performance = 0  # Low
            elif income < income_percentile[1]:
                performance = 1  # Medium
            else:
                performance = 2  # High
            
            record = {
                'Stall_ID': vendor_id,
                'State': vendor_state,
                'GI_Product': vendor_product,
                'State_Region': vendor_region,
                'Year': year,
                'Income_Numeric': int(income),
                'Digital_Adoption_Index': round(digital_score, 3),
                'Ecommerce_Active': ecommerce_active,
                'Govt_Interaction_Score': govt_score,
                'Govt_Interaction_Rating': govt_ratings[govt_score],
                'Has_Govt_ID': np.random.choice([0, 1], p=[0.3, 0.7]),
                'Performance_Category': performance,
                'Years_Active': year - 2019,
                'Fair_Beneficial': np.random.choice([0, 1], p=[0.2, 0.8]),
                'Wants_More_Events': np.random.choice([0, 1], p=[0.25, 0.75])
            }
            
            data.append(record)
    
    return pd.DataFrame(data)

@st.cache_data
def load_data():
    """Load data with fallback to sample data"""
    try:
        # Try to load processed data
        df = pd.read_csv('../data/processed/processed_data.csv')
        st.success("✅ Loaded real processed data")
        return df
    except:
        try:
            # Try without ../
            df = pd.read_csv('data/processed/processed_data.csv')
            st.success("✅ Loaded real processed data")
            return df
        except:
            # Create sample data
            st.info("📊 Using generated sample data for demonstration")
            return create_sample_data()

def main():
    # Load data
    df = load_data()
    
    # Header
    st.markdown('<h1 class="main-header">🏪 GI Craft Fair Analytics Dashboard</h1>', 
                unsafe_allow_html=True)
    
    # Sidebar navigation
    st.sidebar.title("🔍 Navigation")
    page = st.sidebar.selectbox(
        "Choose Analysis Section",
        ["📊 Overview", "📈 Performance Analysis", "🎯 Vendor Insights", 
         "🗺️ Geographic Analysis", "📋 Data Explorer"]
    )
    
    # Overview Page
    if page == "📊 Overview":
        st.header("📊 Project Overview")
        
        # Key metrics
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric("Total Vendors", df['Stall_ID'].nunique())
        
        with col2:
            st.metric("Total Records", len(df))
        
        with col3:
            st.metric("States Covered", df['State'].nunique())
        
        with col4:
            st.metric("GI Products", df['GI_Product'].nunique())
        
        with col5:
            years_span = f"{df['Year'].min()}-{df['Year'].max()}"
            st.metric("Years Covered", years_span)
        
        # Key performance indicators
        st.subheader("🎯 Key Performance Indicators")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            avg_income = df['Income_Numeric'].mean()
            st.metric("Average Income", f"₹{avg_income:,.0f}")
        
        with col2:
            digital_rate = df['Digital_Adoption_Index'].mean()
            st.metric("Digital Adoption", f"{digital_rate:.1%}")
        
        with col3:
            ecommerce_rate = df['Ecommerce_Active'].mean()
            st.metric("E-commerce Rate", f"{ecommerce_rate:.1%}")
        
        with col4:
            govt_engagement = df['Has_Govt_ID'].mean()
            st.metric("Govt Program Participation", f"{govt_engagement:.1%}")
        
        # Performance distribution
        st.subheader("📊 Performance Distribution")
        
        perf_labels = {0: 'Low Performers', 1: 'Medium Performers', 2: 'High Performers'}
        perf_counts = df['Performance_Category'].map(perf_labels).value_counts()
        
        fig = px.pie(values=perf_counts.values, names=perf_counts.index,
                     title="Vendor Performance Categories",
                     color_discrete_sequence=['#ff7f7f', '#ffbf7f', '#7fff7f'])
        st.plotly_chart(fig, use_container_width=True)
        
    # Performance Analysis Page
    elif page == "📈 Performance Analysis":
        st.header("📈 Performance Analysis")
        
        # Income trends over time
        st.subheader("💰 Income Trends Over Time")
        
        yearly_income = df.groupby('Year')['Income_Numeric'].agg(['mean', 'median']).reset_index()
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=yearly_income['Year'], y=yearly_income['mean'],
                                mode='lines+markers', name='Average Income', line=dict(width=3)))
        fig.add_trace(go.Scatter(x=yearly_income['Year'], y=yearly_income['median'],
                                mode='lines+markers', name='Median Income', line=dict(dash='dash')))
        
        fig.update_layout(title="Income Growth Trends", 
                         xaxis_title="Year", yaxis_title="Income (₹)")
        st.plotly_chart(fig, use_container_width=True)
        
        # Digital transformation timeline
        st.subheader("📱 Digital Transformation Timeline")
        
        digital_trends = df.groupby('Year').agg({
            'Digital_Adoption_Index': 'mean',
            'Ecommerce_Active': 'mean'
        }).reset_index()
        
        fig = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig.add_trace(
            go.Scatter(x=digital_trends['Year'], y=digital_trends['Digital_Adoption_Index'],
                      mode='lines+markers', name='Digital Adoption Index'),
            secondary_y=False
        )
        
        fig.add_trace(
            go.Scatter(x=digital_trends['Year'], y=digital_trends['Ecommerce_Active'],
                      mode='lines+markers', name='E-commerce Adoption Rate', line=dict(dash='dash')),
            secondary_y=True
        )
        
        fig.update_xaxes(title_text="Year")
        fig.update_yaxes(title_text="Digital Adoption Index", secondary_y=False)
        fig.update_yaxes(title_text="E-commerce Rate", secondary_y=True)
        fig.update_layout(title_text="Digital Transformation Progress")
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Performance by state
        st.subheader("🏛️ State-wise Performance")
        
        state_performance = df.groupby('State').agg({
            'Income_Numeric': 'mean',
            'Digital_Adoption_Index': 'mean',
            'Stall_ID': 'nunique'
        }).reset_index()
        
        state_performance.columns = ['State', 'Avg_Income', 'Digital_Score', 'Vendor_Count']
        state_performance = state_performance.sort_values('Avg_Income', ascending=False)
        
        fig = px.bar(state_performance.head(10), x='State', y='Avg_Income',
                     color='Digital_Score', title="Top 10 States by Average Income")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Vendor Insights Page
    elif page == "🎯 Vendor Insights":
        st.header("🎯 Vendor Insights")
        
        # Digital adoption vs income
        st.subheader("📊 Digital Adoption vs Income Performance")
        
        fig = px.scatter(df, x='Digital_Adoption_Index', y='Income_Numeric',
                        color='Performance_Category', size='Govt_Interaction_Score',
                        title="Digital Adoption Impact on Income",
                        labels={'Performance_Category': 'Performance Level'})
        st.plotly_chart(fig, use_container_width=True)
        
        # Government interaction analysis
        st.subheader("🏛️ Government Interaction Impact")
        
        govt_analysis = df.groupby('Govt_Interaction_Rating')['Income_Numeric'].mean().reset_index()
        govt_analysis = govt_analysis.sort_values('Income_Numeric', ascending=False)
        
        fig = px.bar(govt_analysis, x='Govt_Interaction_Rating', y='Income_Numeric',
                     title="Income by Government Interaction Quality")
        st.plotly_chart(fig, use_container_width=True)
        
        # Product category analysis
        st.subheader("🎨 GI Product Performance")
        
        product_perf = df.groupby('GI_Product').agg({
            'Income_Numeric': 'mean',
            'Digital_Adoption_Index': 'mean',
            'Stall_ID': 'nunique'
        }).reset_index()
        
        product_perf.columns = ['Product', 'Avg_Income', 'Digital_Score', 'Vendor_Count']
        product_perf = product_perf.sort_values('Avg_Income', ascending=False).head(8)
        
        fig = px.bar(product_perf, x='Product', y='Avg_Income',
                     title="Average Income by GI Product Category")
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)
    
    # Geographic Analysis Page
    elif page == "🗺️ Geographic Analysis":
        st.header("🗺️ Geographic Analysis")
        
        # Regional performance
        st.subheader("🌍 Regional Performance Comparison")
        
        regional_stats = df.groupby('State_Region').agg({
            'Income_Numeric': ['mean', 'std'],
            'Digital_Adoption_Index': 'mean',
            'Stall_ID': 'nunique'
        }).round(2)
        
        regional_stats.columns = ['Avg_Income', 'Income_Std', 'Digital_Score', 'Vendor_Count']
        regional_stats = regional_stats.reset_index()
        
        fig = px.bar(regional_stats, x='State_Region', y='Avg_Income',
                     color='Digital_Score', title="Regional Performance Overview")
        st.plotly_chart(fig, use_container_width=True)
        
        # State comparison matrix
        st.subheader("📊 State Performance Matrix")
        
        state_matrix = df.groupby('State').agg({
            'Income_Numeric': 'mean',
            'Digital_Adoption_Index': 'mean',
            'Govt_Interaction_Score': 'mean'
        }).reset_index()
        
        fig = px.scatter(state_matrix, x='Digital_Adoption_Index', y='Income_Numeric',
                        size='Govt_Interaction_Score', hover_name='State',
                        title="State Performance Matrix")
        st.plotly_chart(fig, use_container_width=True)
        
        # Show state rankings
        st.subheader("🏆 State Rankings")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Top States by Income**")
            income_ranking = df.groupby('State')['Income_Numeric'].mean().sort_values(ascending=False).head(5)
            for i, (state, income) in enumerate(income_ranking.items(), 1):
                st.write(f"{i}. {state}: ₹{income:,.0f}")
        
        with col2:
            st.write("**Top States by Digital Adoption**")
            digital_ranking = df.groupby('State')['Digital_Adoption_Index'].mean().sort_values(ascending=False).head(5)
            for i, (state, score) in enumerate(digital_ranking.items(), 1):
                st.write(f"{i}. {state}: {score:.2f}")
    
    # Data Explorer Page
    elif page == "📋 Data Explorer":
        st.header("📋 Data Explorer")
        
        # Data overview
        st.subheader("📊 Dataset Overview")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Dataset Information**")
            st.write(f"- Total records: {len(df):,}")
            st.write(f"- Unique vendors: {df['Stall_ID'].nunique():,}")
            st.write(f"- Features: {len(df.columns)}")
            st.write(f"- Date range: {df['Year'].min()}-{df['Year'].max()}")
        
        with col2:
            st.write("**Data Quality**")
            missing_data = df.isnull().sum().sum()
            st.write(f"- Missing values: {missing_data}")
            st.write(f"- Duplicate records: {df.duplicated().sum()}")
            st.write(f"- Data completeness: {((len(df) * len(df.columns) - missing_data) / (len(df) * len(df.columns)) * 100):.1f}%")
        
        # Sample data
        st.subheader("🔍 Sample Data")
        st.dataframe(df.head(10))
        
        # Statistical summary
        st.subheader("📈 Statistical Summary")
        numeric_cols = ['Income_Numeric', 'Digital_Adoption_Index', 'Govt_Interaction_Score']
        st.dataframe(df[numeric_cols].describe())
        
        # Download data
        st.subheader("💾 Download Data")
        
        @st.cache_data
        def convert_df(dataframe):
            return dataframe.to_csv(index=False).encode('utf-8')
        
        csv = convert_df(df)
        
        st.download_button(
            label="Download dataset as CSV",
            data=csv,
            file_name='gi_craft_analytics_data.csv',
            mime='text/csv',
        )
    
    # Footer
    st.sidebar.markdown("---")
    st.sidebar.markdown("**🏪 GI Craft Fair Analytics**")
    st.sidebar.markdown("Advanced ML & Statistical Analysis")
    st.sidebar.markdown("Built with Streamlit & Python")
    
    # Additional insights in sidebar
    st.sidebar.subheader("💡 Quick Insights")
    
    high_performers = len(df[df['Performance_Category'] == 2])
    total_vendors = df['Stall_ID'].nunique()
    
    st.sidebar.metric("High Performers", f"{high_performers}")
    st.sidebar.metric("Success Rate", f"{high_performers/total_vendors:.1%}")
    
    avg_digital = df['Digital_Adoption_Index'].mean()
    st.sidebar.metric("Avg Digital Score", f"{avg_digital:.2f}")

if __name__ == "__main__":
    main()
