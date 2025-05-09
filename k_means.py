import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

def preprocess_data(df):
    """
    Preprocess the data by scaling the features
    """
    scaler = StandardScaler()
    features = df.iloc[:, 1:9]
    scaled_features = scaler.fit_transform(features)
    return scaled_features

def find_optimal_clusters(data, max_k=10):
    """
    Find optimal number of clusters using silhouette score
    """
    silhouette_scores = []
    for k in range(2, max_k + 1):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clusters = kmeans.fit_predict(data)
        score = silhouette_score(data, clusters)
        silhouette_scores.append(score)
    
    return np.argmax(silhouette_scores) + 2  # Add 2 because we started from k=2

def perform_kmeans(df, n_clusters=None):
    """
    Perform k-means clustering on the data
    """
    # Preprocess the data
    scaled_data = preprocess_data(df)
    
    # Find optimal number of clusters if not specified
    if n_clusters is None:
        n_clusters = find_optimal_clusters(scaled_data)
    
    # Perform clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    # Add cluster labels to original dataframe
    result_df = df.copy()
    result_df['Cluster'] = clusters
    
    return result_df, kmeans.cluster_centers_