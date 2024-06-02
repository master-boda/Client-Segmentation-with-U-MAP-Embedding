import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

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