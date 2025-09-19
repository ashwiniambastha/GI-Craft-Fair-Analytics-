"""
Advanced Visualization & Insights Dashboard
==========================================
Comprehensive visualization suite for GI Craft Fair analytics
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio
import yaml
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class AdvancedVisualizationSuite:
    """Comprehensive visualization and dashboard creation"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set style preferences
        plt.style.use('seaborn-v0_8')
        sns.set_palette("husl")
        pio.templates.default = "plotly_white"
        
        self.color_palette = px.colors.qualitative.Set3
        self.sequential_colors = px.colors.sequential.Viridis
        
    def load_all_data(self) -> dict:
        """Load all processed data files"""
        data = {}
        
        try:
            data['raw'] = pd.read_csv('data/processed/processed_data.csv')
            print("Loaded processed data")
        except FileNotFoundError:
            print("Warning: Processed data not found")
            
        try:
            data['survival'] = pd.read_csv('results/survival_data.csv')
            data['survival_predictions'] = pd.read_csv('results/survival_predictions.csv')
            print("Loaded survival analysis data")
        except FileNotFoundError:
            print("Warning: Survival analysis data not found")
            
        try:
            data['segments'] = pd.read_csv('results/vendor_segments.csv')
            print("Loaded segmentation data")
        except FileNotFoundError:
            print("Warning: Segmentation data not found")
            
        try:
            data['ensemble_predictions'] = pd.read_csv('results/ensemble_predictions.csv')
            print("Loaded ensemble predictions")
        except FileNotFoundError:
            print("Warning: Ensemble predictions not found")
            
        # Load JSON results
        try:
            with open('results/survival_insights.json', 'r') as f:
                data['survival_insights'] = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open('results/segmentation_results.json', 'r') as f:
                data['segmentation_results'] = json.load(f)
        except FileNotFoundError:
            pass
            
        try:
            with open('results/ensemble_results.json', 'r') as f:
                data['ensemble_results'] = json.load(f)
        except FileNotFoundError:
            pass
        
        return data
    
    def create_performance_overview(self, df: pd.DataFrame) -> go.Figure:
        """Create comprehensive performance overview dashboard"""
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Income Distribution by State', 'Digital Adoption Over Time',
                          'Government Interaction Impact', 'Vendor Performance Categories'),
            specs=[[{"secondary_y": False}, {"secondary_y": True}],
                   [{"secondary_y": False}, {"type": "pie"}]]
        )
        
        # 1. Income Distribution by State (Box Plot)
        states = df['State'].value_counts().head(10).index
        for i, state in enumerate(states):
            state_data = df[df['State'] == state]['Income_Numeric']
            fig.add_trace(
                go.Box(y=state_data, name=state, showlegend=False),
                row=1, col=1
            )
        
        # 2. Digital Adoption Over Time (Line + Bar)
        yearly_stats = df.groupby('Year').agg({
            'Digital_Adoption_Index': 'mean',
            'Ecommerce_Active': 'sum'
        }).reset_index()
        
        fig.add_trace(
            go.Scatter(x=yearly_stats['Year'], y=yearly_stats['Digital_Adoption_Index'],
                      mode='lines+markers', name='Avg Digital Adoption'),
            row=1, col=2
        )
        
        fig.add_trace(
            go.Bar(x=yearly_stats['Year'], y=yearly_stats['Ecommerce_Active'],
                  name='E-commerce Active Count', yaxis='y2'),
            row=1, col=2, secondary_y=True
        )
        
        # 3. Government Interaction Impact (Scatter)
        fig.add_trace(
            go.Scatter(x=df['Govt_Interaction_Score'], y=df['Income_Numeric'],
                      mode='markers', opacity=0.6,
                      marker=dict(color=df['Has_Govt_ID'], 
                                colorscale='RdYlBu', showscale=False),
                      name='Vendors', showlegend=False),
            row=2, col=1
        )
        
        # 4. Performance Categories (Pie Chart)
        if 'Performance_Category' in df.columns:
            perf_counts = df['Performance_Category'].value_counts()
            fig.add_trace(
                go.Pie(labels=['Low', 'Medium', 'High'], values=perf_counts.values,
                      showlegend=False),
                row=2, col=2
            )
        
        # Update layout
        fig.update_xaxes(title_text="State", row=1, col=1)
        fig.update_yaxes(title_text="Income (₹)", row=1, col=1)
        fig.update_xaxes(title_text="Year", row=1, col=2)
        fig.update_yaxes(title_text="Digital Adoption Index", row=1, col=2)
        fig.update_yaxes(title_text="E-commerce Count", secondary_y=True, row=1, col=2)
        fig.update_xaxes(title_text="Government Interaction Score", row=2, col=1)
        fig.update_yaxes(title_text="Income (₹)", row=2, col=1)
        
        fig.update_layout(
            title_text="GI Craft Fair Performance Overview Dashboard",
            height=800,
            showlegend=True
        )
        
        return fig
    
    def create_survival_analysis_plots(self, survival_data: pd.DataFrame, 
                                     predictions: pd.DataFrame) -> list:
        """Create survival analysis visualizations"""
        
        plots = []
        
        # 1. Survival Probability Distribution
        fig1 = go.Figure()
        
        for col in ['Survival_Prob_1Y', 'Survival_Prob_2Y', 'Survival_Prob_3Y', 'Survival_Prob_5Y']:
            if col in predictions.columns:
                fig1.add_trace(go.Histogram(
                    x=predictions[col], 
                    name=col.replace('Survival_Prob_', '').replace('Y', ' Year'),
                    opacity=0.7
                ))
        
        fig1.update_layout(
            title="Distribution of Survival Probabilities",
            xaxis_title="Survival Probability",
            yaxis_title="Number of Vendors",
            barmode='overlay'
        )
        plots.append(fig1)
        
        # 2. Risk Category Analysis
        if 'Risk_Category' in predictions.columns:
            risk_summary = predictions['Risk_Category'].value_counts()
            
            fig2 = go.Figure(data=[
                go.Bar(x=risk_summary.index, y=risk_summary.values,
                       marker_color=['red', 'orange', 'green'])
            ])
            
            fig2.update_layout(
                title="Vendor Risk Distribution",
                xaxis_title="Risk Category",
                yaxis_title="Number of Vendors"
            )
            plots.append(fig2)
        
        # 3. Duration vs Income Analysis
        fig3 = go.Figure()
        
        fig3.add_trace(go.Scatter(
            x=survival_data['Duration'],
            y=survival_data['Avg_Income'],
            mode='markers',
            marker=dict(
                color=survival_data['Event'],
                colorscale='RdYlGn',
                reversescale=True,
                size=8,
                colorbar=dict(title="Churned")
            ),
            text=[f"Vendor {i}" for i in survival_data['Vendor_ID']],
            hovertemplate="Duration: %{x}<br>Income: ₹%{y}<br>%{text}<extra></extra>"
        ))
        
        fig3.update_layout(
            title="Vendor Lifespan vs Average Income",
            xaxis_title="Years in Business",
            yaxis_title="Average Income (₹)"
        )
        plots.append(fig3)
        
        return plots
    
    def create_segmentation_visualizations(self, segments_data: pd.DataFrame,
                                         segmentation_results: dict) -> list:
        """Create comprehensive segmentation visualizations"""
        
        plots = []
        
        # 1. Segment Size and Distribution
        if 'Segment_Name' in segments_data.columns:
            segment_counts = segments_data['Segment_Name'].value_counts()
            
            fig1 = go.Figure(data=[
                go.Pie(labels=segment_counts.index, values=segment_counts.values,
                       textinfo='label+percent')
            ])
            
            fig1.update_layout(title="Vendor Segment Distribution")
            plots.append(fig1)
        
        # 2. Segment Performance Comparison
        if 'Segment_Name' in segments_data.columns and 'Avg_Income' in segments_data.columns:
            
            fig2 = go.Figure()
            
            for segment in segments_data['Segment_Name'].unique():
                segment_data = segments_data[segments_data['Segment_Name'] == segment]
                
                fig2.add_trace(go.Box(
                    y=segment_data['Avg_Income'],
                    name=segment,
                    boxpoints='outliers'
                ))
            
            fig2.update_layout(
                title="Income Distribution by Segment",
                xaxis_title="Segment",
                yaxis_title="Average Income (₹)"
            )
            plots.append(fig2)
        
        # 3. Digital Adoption vs Income by Segment
        if all(col in segments_data.columns for col in ['Current_Digital_Score', 'Avg_Income', 'Segment_Name']):
            
            fig3 = go.Figure()
            
            for segment in segments_data['Segment_Name'].unique():
                segment_data = segments_data[segments_data['Segment_Name'] == segment]
                
                fig3.add_trace(go.Scatter(
                    x=segment_data['Current_Digital_Score'],
                    y=segment_data['Avg_Income'],
                    mode='markers',
                    name=segment,
                    marker=dict(size=8)
                ))
            
            fig3.update_layout(
                title="Digital Adoption vs Income by Segment",
                xaxis_title="Digital Adoption Score",
                yaxis_title="Average Income (₹)"
            )
            plots.append(fig3)
        
        # 4. Segment Geographic Distribution
        if all(col in segments_data.columns for col in ['State', 'Segment_Name']):
            
            # Create crosstab
            geo_segments = pd.crosstab(segments_data['State'], segments_data['Segment_Name'])
            
            fig4 = go.Figure()
            
            for segment in geo_segments.columns:
                fig4.add_trace(go.Bar(
                    name=segment,
                    x=geo_segments.index,
                    y=geo_segments[segment]
                ))
            
            fig4.update_layout(
                title="Segment Distribution Across States",
                xaxis_title="State",
                yaxis_title="Number of Vendors",
                barmode='stack'
            )
            plots.append(fig4)
        
        return plots
    
    def create_time_series_analysis(self, df: pd.DataFrame) -> list:
        """Create time series analysis visualizations"""
        
        plots = []
        
        # 1. Income Trends Over Time by Category
        yearly_income = df.groupby(['Year', 'Performance_Category'])['Income_Numeric'].mean().reset_index()
        
        fig1 = go.Figure()
        
        for category in yearly_income['Performance_Category'].unique():
            category_data = yearly_income[yearly_income['Performance_Category'] == category]
            fig1.add_trace(go.Scatter(
                x=category_data['Year'],
                y=category_data['Income_Numeric'],
                mode='lines+markers',
                name=f'Performance Level {category}'
            ))
        
        fig1.update_layout(
            title="Income Trends by Performance Category",
            xaxis_title="Year",
            yaxis_title="Average Income (₹)"
        )
        plots.append(fig1)
        
        # 2. Digital Transformation Timeline
        digital_trends = df.groupby('Year').agg({
            'Digital_Adoption_Index': ['mean', 'std'],
            'Ecommerce_Active': ['sum', 'count']
        }).reset_index()
        
        # Flatten column names
        digital_trends.columns = ['Year', 'Digital_Mean', 'Digital_Std', 'Ecommerce_Sum', 'Total_Count']
        digital_trends['Ecommerce_Rate'] = digital_trends['Ecommerce_Sum'] / digital_trends['Total_Count']
        
        fig2 = make_subplots(specs=[[{"secondary_y": True}]])
        
        fig2.add_trace(
            go.Scatter(x=digital_trends['Year'], y=digital_trends['Digital_Mean'],
                      mode='lines+markers', name='Digital Adoption Index'),
            secondary_y=False
        )
        
        fig2.add_trace(
            go.Scatter(x=digital_trends['Year'], y=digital_trends['Ecommerce_Rate'],
                      mode='lines+markers', name='E-commerce Adoption Rate',
                      line=dict(dash='dash')),
            secondary_y=True
        )
        
        fig2.update_xaxes(title_text="Year")
        fig2.update_yaxes(title_text="Digital Adoption Index", secondary_y=False)
        fig2.update_yaxes(title_text="E-commerce Adoption Rate", secondary_y=True)
        fig2.update_layout(title_text="Digital Transformation Timeline")
        
        plots.append(fig2)
        
        # 3. State-wise Performance Evolution
        state_evolution = df.groupby(['Year', 'State'])['Income_Numeric'].mean().reset_index()
        top_states = df.groupby('State')['Income_Numeric'].mean().nlargest(5).index
        
        fig3 = go.Figure()
        
        for state in top_states:
            state_data = state_evolution[state_evolution['State'] == state]
            fig3.add_trace(go.Scatter(
                x=state_data['Year'],
                y=state_data['Income_Numeric'],
                mode='lines+markers',
                name=state
            ))
        
        fig3.update_layout(
            title="Top 5 States - Income Evolution",
            xaxis_title="Year",
            yaxis_title="Average Income (₹)"
        )
        plots.append(fig3)
        
        return plots
    
    def create_model_performance_dashboard(self, ensemble_results: dict) -> go.Figure:
        """Create model performance comparison dashboard"""
        
        if 'cv_results' not in ensemble_results:
            print("No cross-validation results found")
            return None
        
        cv_results = ensemble_results['cv_results']
        
        # Extract model names and scores
        models = list(cv_results.keys())
        mean_scores = [cv_results[model]['mean_accuracy'] for model in models]
        std_scores = [cv_results[model]['std_accuracy'] for model in models]
        
        fig = go.Figure()
        
        # Add bar chart with error bars
        fig.add_trace(go.Bar(
            name='CV Accuracy',
            x=models,
            y=mean_scores,
            error_y=dict(type='data', array=std_scores),
            marker_color=self.color_palette[:len(models)]
        ))
        
        fig.update_layout(
            title="Model Performance Comparison (Cross-Validation)",
            xaxis_title="Models",
            yaxis_title="Accuracy",
            yaxis=dict(range=[0.7, 1.0])  # Focus on the relevant range
        )
        
        return fig
    
    def create_business_insights_summary(self, all_data: dict) -> dict:
        """Generate comprehensive business insights summary"""
        
        insights = {
            'overview': {},
            'key_metrics': {},
            'recommendations': [],
            'market_opportunities': []
        }
        
        if 'raw' in all_data:
            df = all_data['raw']
            
            # Overview metrics
            insights['overview'] = {
                'total_vendors': df['Stall_ID'].nunique(),
                'total_records': len(df),
                'years_covered': f"{df['Year'].min()}-{df['Year'].max()}",
                'states_covered': df['State'].nunique(),
                'gi_products': df['GI_Product'].nunique()
            }
            
            # Key performance metrics
            insights['key_metrics'] = {
                'average_income': f"₹{df['Income_Numeric'].mean():,.0f}",
                'income_growth_rate': f"{((df.groupby('Stall_ID')['Income_Numeric'].last() / df.groupby('Stall_ID')['Income_Numeric'].first()).mean() - 1) * 100:.1f}%",
                'digital_adoption_rate': f"{(df['Ecommerce_Active'].sum() / len(df)) * 100:.1f}%",
                'govt_program_participation': f"{(df['Has_Govt_ID'].sum() / len(df)) * 100:.1f}%"
            }
        
        # Add survival insights
        if 'survival_insights' in all_data:
            surv_insights = all_data['survival_insights']
            if 'overall_metrics' in surv_insights:
                insights['key_metrics']['median_vendor_lifespan'] = f"{surv_insights['overall_metrics']['median_vendor_lifespan']:.1f} years"
                insights['key_metrics']['churn_rate'] = f"{surv_insights['overall_metrics']['overall_churn_rate'] * 100:.1f}%"
        
        # Add model performance
        if 'ensemble_results' in all_data:
            ens_results = all_data['ensemble_results']
            if 'test_accuracy' in ens_results:
                insights['key_metrics']['model_accuracy'] = f"{ens_results['test_accuracy'] * 100:.1f}%"
        
        # Generate recommendations based on data
        insights['recommendations'] = [
            "Focus on digital transformation for traditional vendors",
            "Expand successful vendor programs to underperforming states",
            "Develop targeted interventions for high-risk vendors",
            "Leverage high-performing segments for market expansion"
        ]
        
        # Market opportunities
        insights['market_opportunities'] = [
            "Untapped potential in low-digital adoption regions",
            "Growth opportunities in emerging GI product categories",
            "Cross-selling potential between complementary products",
            "Premium market positioning for high-performing vendors"
        ]
        
        return insights
    
    def save_all_visualizations(self, all_data: dict, save_interactive: bool = True):
        """Save all visualizations as HTML files"""
        
        print("Creating and saving all visualizations...")
        
        plots_created = []
        
        # Performance Overview
        if 'raw' in all_data:
            fig = self.create_performance_overview(all_data['raw'])
            if save_interactive:
                fig.write_html('results/plots/performance_overview_interactive.html')
            plots_created.append('Performance Overview')
        
        # Survival Analysis
        if 'survival' in all_data and 'survival_predictions' in all_data:
            survival_plots = self.create_survival_analysis_plots(
                all_data['survival'], all_data['survival_predictions']
            )
            for i, plot in enumerate(survival_plots):
                if save_interactive:
                    plot.write_html(f'results/plots/survival_analysis_{i+1}_interactive.html')
            plots_created.append(f'Survival Analysis ({len(survival_plots)} plots)')
        
        # Segmentation
        if 'segments' in all_data and 'segmentation_results' in all_data:
            segment_plots = self.create_segmentation_visualizations(
                all_data['segments'], all_data['segmentation_results']
            )
            for i, plot in enumerate(segment_plots):
                if save_interactive:
                    plot.write_html(f'results/plots/segmentation_{i+1}_interactive.html')
            plots_created.append(f'Segmentation ({len(segment_plots)} plots)')
        
        # Time Series
        if 'raw' in all_data:
            ts_plots = self.create_time_series_analysis(all_data['raw'])
            for i, plot in enumerate(ts_plots):
                if save_interactive:
                    plot.write_html(f'results/plots/time_series_{i+1}_interactive.html')
            plots_created.append(f'Time Series ({len(ts_plots)} plots)')
        
        # Model Performance
        if 'ensemble_results' in all_data:
            model_fig = self.create_model_performance_dashboard(all_data['ensemble_results'])
            if model_fig and save_interactive:
                model_fig.write_html('results/plots/model_performance_interactive.html')
                plots_created.append('Model Performance')
        
        # Business Insights
        insights = self.create_business_insights_summary(all_data)
        with open('results/business_insights_summary.json', 'w') as f:
            json.dump(insights, f, indent=2)
        plots_created.append('Business Insights Summary')
        
        print(f"\nTop Recommendations:")
    for i, rec in enumerate(business_insights['recommendations'][:3], 1):
        print(f"{i}. {rec}")
    
    print(f"\nAll visualizations saved to results/plots/")
    print("Interactive HTML files created for detailed exploration")
    
    return viz, all_data, business_insights

if __name__ == "__main__":
    visualizer, data, insights = main()
    print("Advanced visualization pipeline completed successfully!")"Created and saved: {', '.join(plots_created)}")
        return plots_created, insights

def main():
    """Run complete visualization pipeline"""
    
    # Initialize visualization suite
    viz = AdvancedVisualizationSuite()
    
    # Load all available data
    all_data = viz.load_all_data()
    
    if not all_data:
        print("No data found. Please run the analysis pipelines first!")
        return
    
    print(f"Loaded data files: {list(all_data.keys())}")
    
    # Create and save all visualizations
    plots_created, business_insights = viz.save_all_visualizations(all_data, save_interactive=True)
    
    print("\nBusiness Insights Summary:")
    print(f"Total Vendors: {business_insights['overview'].get('total_vendors', 'N/A')}")
    print(f"Model Accuracy: {business_insights['key_metrics'].get('model_accuracy', 'N/A')}")
    print(f"Digital Adoption Rate: {business_insights['key_metrics'].get('digital_adoption_rate', 'N/A')}")
    
    print(f
