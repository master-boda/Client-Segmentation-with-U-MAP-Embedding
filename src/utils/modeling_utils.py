import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances_argmin_min

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def append_kmeans_clusters(df, n_clusters=7, random_state=42):
    """
    Perform KMeans clustering and append the cluster labels to the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    n_clusters (int): The number of clusters for KMeans.
    random_state (int): The random state for reproducibility.

    Returns:
    pd.DataFrame: The DataFrame with an additional column for cluster labels.
    """
    new_df = df.copy()
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(new_df)
    
    new_df['cluster'] = cluster_labels
    
    return new_df

def adjust_and_concat_clusters(main_df, outliers_df):
    """
    Adjusts the cluster labels in the outliers DataFrame and concatenates it with the main DataFrame.

    Parameters:
    main_df (pd.DataFrame): The DataFrame containing the main data with cluster labels.
    outliers_df (pd.DataFrame): The DataFrame containing the outliers data with cluster labels.

    Returns:
    pd.DataFrame: The concatenated DataFrame with adjusted cluster labels for the outliers.
    """
    main_df_copy = main_df.copy()
    outliers_df_copy = outliers_df.copy()
    
    max_cluster_label = main_df_copy['cluster'].max()
    
    outliers_df_copy['cluster'] += (max_cluster_label + 1)
    
    combined_df = pd.concat([main_df_copy, outliers_df_copy])
    
    return combined_df

def append_kmeans_clusters(df, n_clusters=7, random_state=42):
    new_df = df.copy()
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(new_df)
    new_df['cluster'] = cluster_labels
    return new_df, kmeans

def evaluate_and_merge_outliers(latent_df_outliers, centroids_non_outliers, threshold=0.5):
    nearest_clusters, distances = pairwise_distances_argmin_min(latent_df_outliers, centroids_non_outliers)
    merge_decisions = distances < threshold
    return merge_decisions, nearest_clusters

def update_cluster_labels(latent_df_outliers, merge_decisions, nearest_clusters, n_clusters_non_outliers):
    for idx, (merge_decision, nearest_cluster) in enumerate(zip(merge_decisions, nearest_clusters)):
        if merge_decision:
            latent_df_outliers.iloc[idx, latent_df_outliers.columns.get_loc('cluster')] = nearest_cluster
        else:
            latent_df_outliers.iloc[idx, latent_df_outliers.columns.get_loc('cluster')] = n_clusters_non_outliers + latent_df_outliers.iloc[idx, latent_df_outliers.columns.get_loc('cluster')]
    return latent_df_outliers
