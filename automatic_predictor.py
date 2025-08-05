"""
performs K-means clustering to identify patterns and creates 
separate CSV files for each cluster.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples
import matplotlib.pyplot as plt
import seaborn as sns
import os
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import nltk


def preprocess_text(text):
    """Simple text preprocessing for clustering."""
    if pd.isna(text):
        return ""
    
    # Convert to lowercase and remove special characters
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    
    # Tokenize and remove stopwords
    tokens = word_tokenize(text)
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    processed_tokens = []
    for token in tokens:
        if token not in stop_words and len(token) > 2:
            stemmed_token = stemmer.stem(token)
            processed_tokens.append(stemmed_token)
    
    return ' '.join(processed_tokens)


def find_optimal_clusters(tfidf_matrix, max_clusters=10):
    """Find optimal number of clusters using silhouette score."""
    best_score = -1
    best_clusters = 2
    
    n_samples = tfidf_matrix.shape[0]
    max_possible_clusters = min(max_clusters + 1, n_samples // 2)
    
    for n_clusters in range(2, max_possible_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(tfidf_matrix)
        score = silhouette_score(tfidf_matrix, cluster_labels)
        
        if score > best_score:
            best_score = score
            best_clusters = n_clusters
    
    return best_clusters


def create_visualizations(data, tfidf_matrix, clusters, kmeans, vectorizer, output_dir):
    """Create comprehensive visualizations for cluster analysis."""
    print("Creating cluster visualizations...")
    
    # Create plots directory
    plots_dir = os.path.join(output_dir, 'plots')
    os.makedirs(plots_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_palette("husl")
    
    # PCA Scatter Plot (2D)
    print("  Creating PCA scatter plot...")
    plt.figure(figsize=(12, 8))
    
    # Reduce dimensionality with PCA
    pca = PCA(n_components=2, random_state=42)
    tfidf_pca = pca.fit_transform(tfidf_matrix.toarray())
    
    # Plot each cluster with different colors
    unique_clusters = np.unique(clusters)
    colors = sns.color_palette("husl", len(unique_clusters))
    
    for i, cluster_id in enumerate(unique_clusters):
        cluster_mask = clusters == cluster_id
        plt.scatter(
            tfidf_pca[cluster_mask, 0], 
            tfidf_pca[cluster_mask, 1],
            c=[colors[i]], 
            label=f'Cluster {cluster_id} ({np.sum(cluster_mask)} samples)',
            alpha=0.7,
            s=50
        )
    
    # Plot cluster centers in PCA space
    centers_pca = pca.transform(kmeans.cluster_centers_)
    plt.scatter(centers_pca[:, 0], centers_pca[:, 1], 
               c='red', marker='x', s=200, linewidths=3, label='Centroids')
    
    plt.xlabel(f'First Principal Component ({pca.explained_variance_ratio_[0]:.1%} variance)')
    plt.ylabel(f'Second Principal Component ({pca.explained_variance_ratio_[1]:.1%} variance)')
    plt.title('K-means Clustering - PCA Scatter Plot')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    scatter_path = os.path.join(plots_dir, 'pca_scatter_plot.png')
    plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Silhouette Plot
    print("  Creating silhouette plot...")
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Calculate silhouette scores for each sample
    sample_silhouette_values = silhouette_samples(tfidf_matrix, clusters)
    
    y_lower = 10
    for i in range(len(unique_clusters)):
        cluster_silhouette_values = sample_silhouette_values[clusters == i]
        cluster_silhouette_values.sort()
        
        size_cluster_i = cluster_silhouette_values.shape[0]
        y_upper = y_lower + size_cluster_i
        
        color = colors[i]
        ax.fill_betweenx(np.arange(y_lower, y_upper),
                        0, cluster_silhouette_values,
                        facecolor=color, edgecolor=color, alpha=0.7)
        
        # Label the silhouette plots with their cluster numbers at the middle
        ax.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))
        y_lower = y_upper + 10
    
    ax.set_xlabel('Silhouette coefficient values')
    ax.set_ylabel('Cluster label')
    ax.set_title('Silhouette Plot for K-means Clustering')
    
    # Add vertical line for average silhouette score
    silhouette_avg = silhouette_score(tfidf_matrix, clusters)
    ax.axvline(x=silhouette_avg, color="red", linestyle="--", 
              label=f'Average Score: {silhouette_avg:.3f}')
    ax.legend()
    
    plt.tight_layout()
    silhouette_path = os.path.join(plots_dir, 'silhouette_plot.png')
    plt.savefig(silhouette_path, dpi=300, bbox_inches='tight')
    plt.close()

    # Feature Importance Heatmap
    print("  Creating feature importance heatmap...")
    
    # Get feature names and cluster centers
    feature_names = vectorizer.get_feature_names_out()
    cluster_centers = kmeans.cluster_centers_
    
    # Select top features for each cluster
    n_top_features = 20
    top_features_per_cluster = []
    
    for i, center in enumerate(cluster_centers):
        top_indices = center.argsort()[-n_top_features:][::-1]
        top_features_per_cluster.extend(top_indices)
    
    # Get unique top features
    unique_top_features = list(set(top_features_per_cluster))
    top_feature_names = [feature_names[i] for i in unique_top_features]
    
    # Create heatmap data
    heatmap_data = cluster_centers[:, unique_top_features]
    
    plt.figure(figsize=(16, 8))
    sns.heatmap(heatmap_data, 
                xticklabels=top_feature_names,
                yticklabels=[f'Cluster {i}' for i in range(len(cluster_centers))],
                cmap='viridis',
                annot=False,
                cbar_kws={'label': 'TF-IDF Weight'})
    
    plt.title('Cluster-Feature Importance Heatmap')
    plt.xlabel('Top Features')
    plt.ylabel('Clusters')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    heatmap_path = os.path.join(plots_dir, 'cluster_feature_heatmap.png')
    plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 4. Cluster Distribution Bar Chart
    print("  Creating cluster distribution chart...")
    plt.figure(figsize=(12, 6))
    cluster_counts = data['cluster'].value_counts().sort_index()
    
    bars = plt.bar(cluster_counts.index, cluster_counts.values, 
                   color=colors[:len(cluster_counts)])
    
    plt.xlabel('Cluster ID')
    plt.ylabel('Number of Samples')
    plt.title('Distribution of Samples Across Clusters')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, count in zip(bars, cluster_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    # Add percentage labels
    total_samples = len(data)
    for i, (cluster_id, count) in enumerate(cluster_counts.items()):
        percentage = (count / total_samples) * 100
        plt.text(cluster_id, count/2, f'{percentage:.1f}%', 
                ha='center', va='center', fontweight='bold', color='white')
    
    plt.tight_layout()
    bar_plot_path = os.path.join(plots_dir, 'cluster_distribution.png')
    plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. Problem Type Distribution Heatmap
    print("  Creating problem type distribution heatmap...")
    
    # Create problem type vs cluster cross-tabulation
    problem_cluster_crosstab = pd.crosstab(data['problem_type'], data['cluster'])
    
    plt.figure(figsize=(12, 8))
    sns.heatmap(problem_cluster_crosstab, 
                annot=True, 
                fmt='d',
                cmap='Blues',
                cbar_kws={'label': 'Number of Samples'})
    
    plt.title('Problem Type Distribution Across Clusters')
    plt.xlabel('Cluster ID')
    plt.ylabel('Problem Type')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    problem_heatmap_path = os.path.join(plots_dir, 'problem_type_cluster_heatmap.png')
    plt.savefig(problem_heatmap_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  All visualizations saved to: {plots_dir}")
    return plots_dir


def main():
    """Main clustering function."""
    print("Starting Simple Unsupervised Clustering")
    print("=" * 50)
    
    # Load data
    csv_path = 'data/raw/problems_dataset.csv'
    print(f"Loading data from: {csv_path}")
    
    try:
        data = pd.read_csv(csv_path)
        print(f"Loaded {len(data)} samples")
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Preprocess descriptions
    print("Preprocessing text descriptions...")
    data['processed_description'] = data['description'].apply(preprocess_text)
    
    # Remove empty descriptions
    data = data[data['processed_description'].str.len() > 0]
    print(f"Preprocessed {len(data)} samples")
    
    # Create TF-IDF features
    print("Creating TF-IDF features...")
    vectorizer = TfidfVectorizer(
        max_features=500,
        min_df=2,
        max_df=0.95,
        ngram_range=(1, 2),
        stop_words='english'
    )
    
    tfidf_matrix = vectorizer.fit_transform(data['processed_description'])
    print(f"Created TF-IDF matrix: {tfidf_matrix.shape}")
    
    # Find optimal number of clusters
    print("Finding optimal number of clusters...")
    n_clusters = find_optimal_clusters(tfidf_matrix)
    print(f"Optimal clusters: {n_clusters}")
    
    # Perform K-means clustering
    print(f"Performing K-means clustering with {n_clusters} clusters...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(tfidf_matrix)
    
    # Add cluster labels to data
    data['cluster'] = clusters
    
    # Calculate silhouette score
    silhouette_avg = silhouette_score(tfidf_matrix, clusters)
    print(f"Clustering completed. Silhouette score: {silhouette_avg:.3f}")

    # Create output directory
    output_dir = 'data/processed/clusters'
    os.makedirs(output_dir, exist_ok=True)
    
    # Create comprehensive visualizations
    plots_dir = create_visualizations(data, tfidf_matrix, clusters, kmeans, vectorizer, output_dir)
    
    # Save separate CSV for each cluster
    print("Saving cluster CSV files...")
    cluster_files = []
    
    for cluster_id in range(n_clusters):
        # Get data for this cluster
        cluster_data = data[data['cluster'] == cluster_id]
        
        # Select required columns
        output_data = cluster_data[['problem_number', 'description', 'problem_type', 'cluster']].copy()
        
        # Save to CSV
        filename = f'cluster_{cluster_id}_problems.csv'
        filepath = os.path.join(output_dir, filename)
        output_data.to_csv(filepath, index=False)
        
        cluster_files.append({
            'cluster_id': cluster_id,
            'filename': filename,
            'size': len(output_data)
        })
        
        print(f"Cluster {cluster_id}: {len(output_data)} samples → {filename}")
    
    # Print cluster distribution
    print(f"\nCluster Distribution:")
    cluster_counts = data['cluster'].value_counts().sort_index()
    for cluster_id, count in cluster_counts.items():
        percentage = (count / len(data)) * 100
        print(f"   Cluster {cluster_id}: {count} samples ({percentage:.1f}%)")
    
    # Save summary file
    summary_path = os.path.join(output_dir, 'all_clusters_summary.csv')
    summary_data = data[['problem_number', 'description', 'problem_type', 'cluster']]
    summary_data.to_csv(summary_path, index=False)
    print(f"\nSummary file saved: {summary_path}")

    print(f"\nClustering completed successfully!")
    print(f"Output directory: {output_dir}")
    print(f"Generated {len(cluster_files)} cluster CSV files")
    print(f"Visualizations saved in: {plots_dir}")

    return True


if __name__ == "__main__":
    main()
