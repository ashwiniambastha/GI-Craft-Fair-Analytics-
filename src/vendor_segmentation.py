"""
Advanced Vendor Segmentation Analysis
====================================
Multi-dimensional clustering and behavioral segmentation
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, calinski_harabasz_score
from sklearn.manifold import TSNE
import yaml
import joblib
import warnings
warnings.filterwarnings('ignore')

class VendorSegmentationAnalyzer:
    """Advanced vendor segmentation with multiple clustering approaches"""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.scaler = StandardScaler()
        self.pca = PCA()
        self.clustering_models = {}
        self.segment_profiles = {}
        
    def prepare_segmentation_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepare vendor-level data for segmentation"""
        print("Preparing vendor segmentation data...")
        
        # Create vendor-level aggregations
        vendor_features = []
        
        for vendor_id in df['Stall_ID'].unique():
            vendor_df = df[df['Stall_ID'] == vendor_id].sort_values('Year')
            
            # Basic demographics
            state = vendor_df['State'].iloc[0]
            product = vendor_df['GI_Product'].iloc[0]
            region = vendor_df['State_Region'].iloc[0]
            
            # Performance metrics
            avg_income = vendor_df['Income_Numeric'].mean()
            income_growth = (vendor_df['Income_Numeric'].iloc[-1] - vendor_df['Income_Numeric'].iloc[0]) / vendor_df['Income_Numeric'].iloc[0] if len(vendor_df) > 1 else 0
            income_volatility = vendor_df['Income_Numeric'].std() / vendor_df['Income_Numeric'].mean() if vendor_df['Income_Numeric'].mean() > 0 else 0
            peak_income = vendor_df['Income_Numeric'].max()
            
            # Digital adoption
            current_digital_score = vendor_df['Digital_Adoption_Index'].iloc[-1]
            digital_growth = vendor_df['Digital_Adoption_Index'].iloc[-1] - vendor_df['Digital_Adoption_Index'].iloc[0] if len(vendor_df) > 1 else 0
            years_digital_active = (vendor_df['Ecommerce_Active'] == 1).sum()
            
            # Government interaction
            avg_govt_interaction = vendor_df['Govt_Interaction_Score'].mean()
            has_govt_id = vendor_df['Has_Govt_ID'].iloc[-1]
            govt_interaction_trend = vendor_df['Govt_Interaction_Score'].iloc[-1] - vendor_df['Govt_Interaction_Score'].iloc[0] if len(vendor_df) > 1 else 0
            
            # Participation patterns
            years_active = len(vendor_df)
            participation_consistency = years_active / (vendor_df['Year'].max() - vendor_df['Year'].min() + 1) if vendor_df['Year'].max() > vendor_df['Year'].min() else 1
            first_participation_year = vendor_df['Year'].min()
            
            # Business stability
            income_trend = np.polyfit(range(len(vendor_df)), vendor_df['Income_Numeric'], 1)[0] if len(vendor_df) > 1 else 0
            performance_consistency = 1 - (vendor_df['Income_Numeric'].std() / vendor_df['Income_Numeric'].mean()) if vendor_df['Income_Numeric'].mean() > 0 else 0
            
            vendor_features.append({
                'Vendor_ID': vendor_id,
                'State': state,
                'GI_Product': product,
                'State_Region': region,
                'Avg_Income': avg_income,
                'Income_Growth': income_growth,
                'Income_Volatility': income_volatility,
                'Peak_Income': peak_income,
                'Income_Trend': income_trend,
                'Current_Digital_Score': current_digital_score,
                'Digital_Growth': digital_growth,
                'Years_Digital_Active': years_digital_active,
                'Avg_Govt_Interaction': avg_govt_interaction,
                'Has_Govt_ID': has_govt_id,
                'Govt_Interaction_Trend': govt_interaction_trend,
                'Years_Active': years_active,
                'Participation_Consistency': participation_consistency,
                'First_Participation_Year': first_participation_year,
                'Performance_Consistency': performance_consistency
            })
        
        segmentation_df = pd.DataFrame(vendor_features)
        
        print(f"Segmentation data prepared: {len(segmentation_df)} vendors with {segmentation_df.shape[1]-4} features")
        return segmentation_df
    
    def select_clustering_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare features for clustering"""
        
        # Key features for segmentation
        clustering_features = [
            'Avg_Income', 'Income_Growth', 'Income_Volatility', 'Peak_Income',
            'Current_Digital_Score', 'Digital_Growth', 'Years_Digital_Active',
            'Avg_Govt_Interaction', 'Govt_Interaction_Trend',
            'Years_Active', 'Participation_Consistency', 'Performance_Consistency'
        ]
        
        # Select features that exist in the dataframe
        available_features = [f for f in clustering_features if f in df.columns]
        
        X = df[available_features].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Handle infinite values
        X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
        
        print(f"Selected {len(available_features)} features for clustering")
        return X
    
    def determine_optimal_clusters(self, X: pd.DataFrame, max_clusters: int = 10) -> dict:
        """Determine optimal number of clusters using multiple methods"""
        print("Determining optimal number of clusters...")
        
        # Scale the data
        X_scaled = self.scaler.fit_transform(X)
        
        results = {
            'silhouette_scores': {},
            'inertia_scores': {},
            'calinski_harabasz_scores': {},
            'bic_scores': {},
            'aic_scores': {}
        }
        
        cluster_range = range(2, max_clusters + 1)
        
        for n_clusters in cluster_range:
            # K-Means
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            kmeans_labels = kmeans.fit_predict(X_scaled)
            
            # Gaussian Mixture Model
            gmm = GaussianMixture(n_components=n_clusters, random_state=42)
            gmm_labels = gmm.fit_predict(X_scaled)
            
            # Calculate metrics
            sil_kmeans = silhouette_score(X_scaled, kmeans_labels)
            sil_gmm = silhouette_score(X_scaled, gmm_labels)
            
            ch_kmeans = calinski_harabasz_score(X_scaled, kmeans_labels)
            ch_gmm = calinski_harabasz_score(X_scaled, gmm_labels)
            
            results['silhouette_scores'][n_clusters] = {
                'kmeans': sil_kmeans, 
                'gmm': sil_gmm
            }
            results['inertia_scores'][n_clusters] = kmeans.inertia_
            results['calinski_harabasz_scores'][n_clusters] = {
                'kmeans': ch_kmeans, 
                'gmm': ch_gmm
            }
            results['bic_scores'][n_clusters] = gmm.bic(X_scaled)
            results['aic_scores'][n_clusters] = gmm.aic(X_scaled)
        
        # Plot evaluation metrics
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Silhouette scores
        axes[0, 0].plot(cluster_range, [results['silhouette_scores'][k]['kmeans'] for k in cluster_range], 'b-o', label='K-Means')
        axes[0, 0].plot(cluster_range, [results['silhouette_scores'][k]['gmm'] for k in cluster_range], 'r-s', label='GMM')
        axes[0, 0].set_xlabel('Number of Clusters')
        axes[0, 0].set_ylabel('Silhouette Score')
        axes[0, 0].set_title('Silhouette Score vs Number of Clusters')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Elbow method (Inertia)
        axes[0, 1].plot(cluster_range, [results['inertia_scores'][k] for k in cluster_range], 'g-o')
        axes[0, 1].set_xlabel('Number of Clusters')
        axes[0, 1].set_ylabel('Inertia')
        axes[0, 1].set_title('Elbow Method (K-Means Inertia)')
        axes[0, 1].grid(True)
        
        # Calinski-Harabasz scores
        axes[1, 0].plot(cluster_range, [results['calinski_harabasz_scores'][k]['kmeans'] for k in cluster_range], 'b-o', label='K-Means')
        axes[1, 0].plot(cluster_range, [results['calinski_harabasz_scores'][k]['gmm'] for k in cluster_range], 'r-s', label='GMM')
        axes[1, 0].set_xlabel('Number of Clusters')
        axes[1, 0].set_ylabel('Calinski-Harabasz Score')
        axes[1, 0].set_title('Calinski-Harabasz Score vs Number of Clusters')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
        
        # BIC/AIC for GMM
        axes[1, 1].plot(cluster_range, [results['bic_scores'][k] for k in cluster_range], 'r-o', label='BIC')
        axes[1, 1].plot(cluster_range, [results['aic_scores'][k] for k in cluster_range], 'b-s', label='AIC')
        axes[1, 1].set_xlabel('Number of Clusters')
        axes[1, 1].set_ylabel('Information Criterion')
        axes[1, 1].set_title('BIC/AIC vs Number of Clusters (GMM)')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.savefig('results/plots/cluster_optimization.png', dpi=300)
        plt.show()
        
        # Recommend optimal number of clusters
        best_silhouette_kmeans = max(results['silhouette_scores'].items(), key=lambda x: x[1]['kmeans'])
        best_silhouette_gmm = max(results['silhouette_scores'].items(), key=lambda x: x[1]['gmm'])
        best_bic = min(results['bic_scores'].items(), key=lambda x: x[1])
        
        recommendations = {
            'silhouette_kmeans': best_silhouette_kmeans[0],
            'silhouette_gmm': best_silhouette_gmm[0],
            'bic_gmm': best_bic[0]
        }
        
        print("Optimal cluster recommendations:")
        for method, n_clusters in recommendations.items():
            print(f"  {method}: {n_clusters} clusters")
        
        return results, recommendations
    
    def perform_clustering(self, X: pd.DataFrame, n_clusters: int = 7) -> dict:
        """Perform multiple clustering algorithms"""
        print(f"Performing clustering with {n_clusters} clusters...")
        
        X_scaled = self.scaler.fit_transform(X)
        
        # K-Means
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans_labels = kmeans.fit_predict(X_scaled)
        
        # Gaussian Mixture Model
        gmm = GaussianMixture(n_components=n_clusters, random_state=42, covariance_type='full')
        gmm_labels = gmm.fit_predict(X_scaled)
        
        # DBSCAN (for comparison)
        dbscan = DBSCAN(eps=0.5, min_samples=5)
        dbscan_labels = dbscan.fit_predict(X_scaled)
        
        # Store models
        self.clustering_models = {
            'kmeans': kmeans,
            'gmm': gmm,
            'dbscan': dbscan,
            'scaler': self.scaler
        }
        
        # Evaluate clustering results
        evaluation_results = {}
        
        for name, labels in [('kmeans', kmeans_labels), ('gmm', gmm_labels)]:
            if len(set(labels)) > 1:  # Valid clustering
                sil_score = silhouette_score(X_scaled, labels)
                ch_score = calinski_harabasz_score(X_scaled, labels)
                evaluation_results[name] = {
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                    'n_clusters': len(set(labels)),
                    'labels': labels
                }
        
        # DBSCAN evaluation
        if len(set(dbscan_labels)) > 1:
            # Exclude noise points (-1) from evaluation
            mask = dbscan_labels != -1
            if mask.sum() > 0:
                sil_score = silhouette_score(X_scaled[mask], dbscan_labels[mask])
                ch_score = calinski_harabasz_score(X_scaled[mask], dbscan_labels[mask])
                evaluation_results['dbscan'] = {
                    'silhouette_score': sil_score,
                    'calinski_harabasz_score': ch_score,
                    'n_clusters': len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0),
                    'noise_points': (dbscan_labels == -1).sum(),
                    'labels': dbscan_labels
                }
        
        print("Clustering evaluation results:")
        for method, results in evaluation_results.items():
            print(f"  {method}: Silhouette={results['silhouette_score']:.3f}, "
                  f"CH Score={results['calinski_harabasz_score']:.1f}, "
                  f"Clusters={results['n_clusters']}")
        
        return evaluation_results
    
    def create_segment_profiles(self, df: pd.DataFrame, X: pd.DataFrame, 
                              labels: np.ndarray, method_name: str = 'gmm') -> dict:
        """Create detailed profiles for each segment"""
        print(f"Creating segment profiles using {method_name}...")
        
        df_with_segments = df.copy()
        df_with_segments['Segment'] = labels
        
        profiles = {}
        
        for segment_id in sorted(set(labels)):
            if segment_id == -1:  # Skip noise points in DBSCAN
                continue
                
            segment_data = df_with_segments[df_with_segments['Segment'] == segment_id]
            segment_size = len(segment_data)
            segment_percent = segment_size / len(df_with_segments) * 100
            
            # Numerical features profile
            numerical_profile = {}
            for col in X.columns:
                if col in segment_data.columns:
                    numerical_profile[col] = {
                        'mean': segment_data[col].mean(),
                        'median': segment_data[col].median(),
                        'std': segment_data[col].std()
                    }
            
            # Categorical features profile
            categorical_profile = {}
            for col in ['State', 'GI_Product', 'State_Region']:
                if col in segment_data.columns:
                    value_counts = segment_data[col].value_counts()
                    categorical_profile[col] = {
                        'top_categories': value_counts.head(3).to_dict(),
                        'diversity_index': len(value_counts) / len(segment_data)
                    }
            
            # Business performance summary
            performance_summary = {
                'avg_income_level': 'High' if segment_data['Avg_Income'].mean() > df_with_segments['Avg_Income'].quantile(0.75) else
                                   'Medium' if segment_data['Avg_Income'].mean() > df_with_segments['Avg_Income'].quantile(0.25) else 'Low',
                'growth_pattern': 'Growth' if segment_data['Income_Growth'].mean() > 0.1 else
                                 'Stable' if segment_data['Income_Growth'].mean() > -0.1 else 'Decline',
                'digital_maturity': 'High' if segment_data['Current_Digital_Score'].mean() > df_with_segments['Current_Digital_Score'].quantile(0.75) else
                                   'Medium' if segment_data['Current_Digital_Score'].mean() > df_with_segments['Current_Digital_Score'].quantile(0.25) else 'Low'
            }
            
            profiles[segment_id] = {
                'segment_size': segment_size,
                'segment_percentage': segment_percent,
                'numerical_profile': numerical_profile,
                'categorical_profile': categorical_profile,
                'performance_summary': performance_summary
            }
        
        self.segment_profiles[method_name] = profiles
        return profiles
    
    def visualize_segments(self, X: pd.DataFrame, labels: np.ndarray, method_name: str = 'gmm'):
        """Create comprehensive segment visualizations"""
        print(f"Creating visualizations for {method_name} segmentation...")
        
        X_scaled = self.scaler.transform(X)
        
        # PCA visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        plt.figure(figsize=(12, 5))
        
        # PCA plot
        plt.subplot(1, 2, 1)
        scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)')
        plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)')
        plt.title(f'{method_name.upper()} Segments - PCA View')
        plt.colorbar(scatter)
        
        # t-SNE visualization
        plt.subplot(1, 2, 2)
        tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(X)//4))
        X_tsne = tsne.fit_transform(X_scaled)
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title(f'{method_name.upper()} Segments - t-SNE View')
        plt.colorbar(scatter)
        
        plt.tight_layout()
        plt.savefig(f'results/plots/{method_name}_segmentation_2d.png', dpi=300)
        plt.show()
        
        # Feature importance heatmap
        segment_means = []
        for segment_id in sorted(set(labels)):
            if segment_id != -1:  # Skip noise points
                segment_mask = labels == segment_id
                segment_mean = X_scaled[segment_mask].mean(axis=0)
                segment_means.append(segment_mean)
        
        segment_means = np.array(segment_means)
        
        plt.figure(figsize=(14, 8))
        sns.heatmap(segment_means, 
                    xticklabels=X.columns, 
                    yticklabels=[f'Segment {i}' for i in range(len(segment_means))],
                    annot=True, fmt='.2f', cmap='RdBu_r', center=0)
        plt.title(f'{method_name.upper()} Segments - Feature Profiles')
        plt.xlabel('Features')
        plt.ylabel('Segments')
        plt.tight_layout()
        plt.savefig(f'results/plots/{method_name}_segment_profiles.png', dpi=300)
        plt.show()
    
    def assign_segment_names(self, profiles: dict, method_name: str = 'gmm') -> dict:
        """Assign meaningful names to segments based on their characteristics"""
        
        segment_names = {}
        
        for segment_id, profile in profiles.items():
            perf = profile['performance_summary']
            size = profile['segment_percentage']
            
            # Generate name based on characteristics
            income_level = perf['avg_income_level']
            growth = perf['growth_pattern']
            digital = perf['digital_maturity']
            
            if income_level == 'High' and growth == 'Growth':
                name = "Premium Growth Champions"
            elif income_level == 'High' and digital == 'High':
                name = "Digital Leaders"
            elif income_level == 'High':
                name = "Established Performers"
            elif growth == 'Growth' and digital == 'High':
                name = "Digital Rising Stars"
            elif growth == 'Growth':
                name = "Growth Oriented"
            elif digital == 'High':
                name = "Digital Adopters"
            elif income_level == 'Medium' and growth == 'Stable':
                name = "Steady Performers"
            elif size < 10:  # Small segments
                name = "Niche Players"
            else:
                name = "Traditional Vendors"
            
            segment_names[segment_id] = name
        
        return segment_names
    
    def generate_segment_insights(self, profiles: dict, method_name: str = 'gmm') -> dict:
        """Generate actionable business insights for each segment"""
        
        insights = {}
        segment_names = self.assign_segment_names(profiles, method_name)
        
        for segment_id, profile in profiles.items():
            name = segment_names[segment_id]
            insights[segment_id] = {
                'segment_name': name,
                'segment_size': profile['segment_size'],
                'segment_percentage': profile['segment_percentage'],
                'key_characteristics': [],
                'opportunities': [],
                'recommendations': []
            }
            
            perf = profile['performance_summary']
            numerical = profile['numerical_profile']
            
            # Key characteristics
            if perf['avg_income_level'] == 'High':
                insights[segment_id]['key_characteristics'].append("High average income")
            if perf['growth_pattern'] == 'Growth':
                insights[segment_id]['key_characteristics'].append("Positive growth trajectory")
            if perf['digital_maturity'] == 'High':
                insights[segment_id]['key_characteristics'].append("High digital adoption")
            
            if 'Years_Active' in numerical:
                years_active = numerical['Years_Active']['mean']
                if years_active > 5:
                    insights[segment_id]['key_characteristics'].append("Experienced vendors")
                elif years_active < 2:
                    insights[segment_id]['key_characteristics'].append("New entrants")
            
            # Opportunities and recommendations
            if perf['avg_income_level'] == 'Low' and perf['digital_maturity'] == 'Low':
                insights[segment_id]['opportunities'].append("Digital transformation potential")
                insights[segment_id]['recommendations'].append("Provide digital training and e-commerce support")
                
            if perf['growth_pattern'] == 'Growth':
                insights[segment_id]['opportunities'].append("Expansion ready")
                insights[segment_id]['recommendations'].append("Offer premium fair placements and marketing support")
                
            if perf['digital_maturity'] == 'High' and perf['avg_income_level'] != 'High':
                insights[segment_id]['opportunities'].append("Untapped revenue potential")
                insights[segment_id]['recommendations'].append("Connect with high-value customers and markets")
                
            if profile['segment_percentage'] < 5:
                insights[segment_id]['opportunities'].append("Specialized niche segment")
                insights[segment_id]['recommendations'].append("Develop targeted programs for this unique segment")
        
        return insights, segment_names
    
    def save_segmentation_results(self, df: pd.DataFrame, clustering_results: dict, 
                                profiles: dict, insights: dict, segment_names: dict):
        """Save all segmentation results"""
        
        # Choose best clustering method (highest silhouette score)
        best_method = max(clustering_results.keys(), 
                         key=lambda x: clustering_results[x]['silhouette_score'])
        best_labels = clustering_results[best_method]['labels']
        
        # Add segments to dataframe
        df_with_segments = df.copy()
        df_with_segments['Segment_ID'] = best_labels
        df_with_segments['Segment_Name'] = [segment_names.get(label, f'Segment_{label}') 
                                          for label in best_labels]
        
        # Save segmented data
        df_with_segments.to_csv('results/vendor_segments.csv', index=False)
        
        # Save clustering models
        joblib.dump(self.clustering_models, 'models/clustering_models.pkl')
        
        # Save detailed results
        import json
        
        results_summary = {
            'clustering_evaluation': clustering_results,
            'segment_profiles': profiles,
            'segment_insights': insights,
            'segment_names': segment_names,
            'best_method': best_method,
            'total_vendors': len(df),
            'number_of_segments': len(set(best_labels)) - (1 if -1 in best_labels else 0)
        }
        
        with open('results/segmentation_results.json', 'w') as f:
            json.dump(results_summary, f, indent=2, default=str)
        
        print("Segmentation results saved successfully!")
        return df_with_segments, results_summary

def main():
    """Run complete vendor segmentation pipeline"""
    
    # Load processed data
    try:
        df = pd.read_csv("data/processed/processed_data.csv")
    except FileNotFoundError:
        print("Please run data preprocessing first!")
        return None
    
    # Initialize segmentation analyzer
    analyzer = VendorSegmentationAnalyzer()
    
    # Prepare segmentation data
    segmentation_data = analyzer.prepare_segmentation_data(df)
    
    # Select clustering features
    X = analyzer.select_clustering_features(segmentation_data)
    
    # Determine optimal number of clusters
    cluster_results, recommendations = analyzer.determine_optimal_clusters(X, max_clusters=10)
    
    # Use recommended number of clusters (or default to 7)
    n_clusters = recommendations.get('silhouette_gmm', 7)
    print(f"Using {n_clusters} clusters for final segmentation")
    
    # Perform clustering
    clustering_results = analyzer.perform_clustering(X, n_clusters=n_clusters)
    
    # Choose best method for detailed analysis
    best_method = max(clustering_results.keys(), 
                     key=lambda x: clustering_results[x]['silhouette_score'])
    best_labels = clustering_results[best_method]['labels']
    
    print(f"Best clustering method: {best_method}")
    
    # Create segment profiles
    profiles = analyzer.create_segment_profiles(segmentation_data, X, best_labels, best_method)
    
    # Visualize segments
    analyzer.visualize_segments(X, best_labels, best_method)
    
    # Generate insights
    insights, segment_names = analyzer.generate_segment_insights(profiles, best_method)
    
    # Print segment summary
    print(f"\nSegmentation Summary ({best_method}):")
    print(f"Total vendors: {len(segmentation_data)}")
    print(f"Number of segments: {len(set(best_labels)) - (1 if -1 in best_labels else 0)}")
    print("\nSegment Overview:")
    
    for segment_id in sorted(set(best_labels)):
        if segment_id != -1:
            name = segment_names.get(segment_id, f'Segment {segment_id}')
            size = insights[segment_id]['segment_size']
            percentage = insights[segment_id]['segment_percentage']
            print(f"  {name}: {size} vendors ({percentage:.1f}%)")
    
    # Save all results
    segmented_df, results_summary = analyzer.save_segmentation_results(
        segmentation_data, clustering_results, profiles, insights, segment_names
    )
    
    print(f"\nSegmentation completed successfully!")
    print(f"Best method: {best_method} with silhouette score: {clustering_results[best_method]['silhouette_score']:.3f}")
    
    return analyzer, segmented_df, results_summary

if __name__ == "__main__":
    analyzer, segmented_data, results = main()
    print("Vendor segmentation analysis completed!")
