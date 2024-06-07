import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import umap
from sklearn.metrics import silhouette_score
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

def swap_columns(df, col1_idx, col2_idx):
    cols = df.columns.tolist()
    cols[col1_idx], cols[col2_idx] = cols[col2_idx], cols[col1_idx]
    return df[cols]

def col_plotter(df):
    """
    Plots histograms for columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to plot histograms for.
    """
    # remove rows with missing values
    df = df.dropna()

    # calculate the number of rows and columns for subplots
    num_plots = len(df.columns)
    num_cols_in_plot = 3
    num_rows = int(np.ceil(num_plots / num_cols_in_plot))

    plt.figure(figsize=(4 * num_cols_in_plot, 4 * num_rows))

    for i, col in enumerate(df.columns):
        plt.subplot(num_rows, num_cols_in_plot, i + 1)
        
        sns.histplot(df[col], kde=True, stat="density", color='blue')
        plt.ylabel('Density')
        
        plt.title(col)
        plt.xlabel(col)

    plt.tight_layout()
    plt.show()

def boxplotter(df):
    """
    Plots boxplots for columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to plot boxplots for.
    """
    # remove rows with missing values
    df = df.dropna()

    # calculate the number of rows and columns for subplots
    num_plots = len(df.columns)
    num_cols_in_plot = 3
    num_rows = int(np.ceil(num_plots / num_cols_in_plot))

    plt.figure(figsize=(3 * num_cols_in_plot, 3 * num_rows))

    for i, col in enumerate(df.columns):
        plt.subplot(num_rows, num_cols_in_plot, i + 1)
        
        sns.boxplot(y=df[col], color='blue')
        plt.ylabel(col)
        plt.title(f'Boxplot of {col}')

    plt.tight_layout()
    plt.show()
    
def correlation_plotter(df):
    """
    Plots a heatmap of the correlation matrix for columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to include in the correlation analysis.
    """
    corr = df.corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1, linewidths=0.5)
    plt.title('Correlation Matrix')

    plt.show()
    
def plot_variance(df):
    """
    Plots the variance of each variable in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.

    Returns:
    None
    """
    variances = df.var()

    plt.figure(figsize=(10, 5))
    variances.plot(kind='bar')
    plt.title('Variance of Variables')
    plt.xlabel('Variables')
    plt.ylabel('Variance')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
def assign_city(lat, lon):
    """
    This function assigns a city name based on given latitude and longitude 
    coordinates. The cities and their corresponding boundaries are predefined 
    within the function:
    
    - Lisbon: Latitude between 38.6 and 38.85, Longitude between -9.25 and -9.05
    - Peniche: Latitude between 39.3 and 39.4, Longitude between -9.5 and -9.3
    - Ericeira: Latitude between 38.9 and 39.0, Longitude between -9.5 and -9.3
    - Other: Any other coordinates outside the defined boundaries
    
    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.

    Returns:
    str: The name of the city corresponding to the given coordinates, 
         or 'Other' if the coordinates do not match any predefined city boundaries.
    """
    if 38.6 <= lat <= 38.85 and -9.25 <= lon <= -9.05:
        return 'Lisbon'
    elif 39.3 <= lat <= 39.4 and -9.5 <= lon <= -9.3:
        return 'Peniche'
    elif 38.9 <= lat <= 39.0 and -9.5 <= lon <= -9.3:
        return 'Ericeira'
    else:
        return 'Other'
    
def plot_elbow_method(data, max_clusters=10, random_state=42):
    """
    This function calculates WCSS for different numbers of clusters and plots the Elbow Method graph.

    Parameters:
    data (pd.DataFrame or np.array): The dataset on which to perform the clustering.
    max_clusters (int): The maximum number of clusters to test.
    random_state (int): The random state for KMeans clustering.

    Returns:
    list: A list containing WCSS values for each number of clusters.
    """
    wcss = []
    for i in range(1, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
    plt.xlabel('Number of Clusters')
    plt.ylabel('WCSS')
    plt.title('Elbow Method for Optimal Number of Clusters')
    plt.show()  

def plot_elbow_and_silhouette(data, max_clusters=10, random_state=42):
    """
    This function calculates WCSS and silhouette scores for different numbers of clusters and plots both graphs on the same plot with dual y-axes.

    Parameters:
    data (pd.DataFrame or np.array): The dataset on which to perform the clustering.
    max_clusters (int): The maximum number of clusters to test.
    random_state (int): The random state for KMeans clustering.

    Returns:
    tuple: A tuple containing two lists - WCSS values and silhouette scores for each number of clusters.
    """
    wcss = []
    silhouette_scores = []

    for i in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=i, random_state=random_state)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
        silhouette_avg = silhouette_score(data, kmeans.labels_)
        silhouette_scores.append(silhouette_avg)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    color = 'tab:blue'
    ax1.set_xlabel('Number of Clusters')
    ax1.set_ylabel('WCSS (SSE)', color=color)
    ax1.plot(range(2, max_clusters + 1), wcss, marker='o', linestyle='--', color=color, label='WCSS (SSE)')
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
    color = 'tab:red'
    ax2.set_ylabel('Silhouette Coefficient', color=color)
    ax2.plot(range(2, max_clusters + 1), silhouette_scores, marker='o', linestyle='--', color=color, label='Silhouette Coefficient')
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2)
    plt.title('Elbow Method and Silhouette Coefficient for Optimal Number of Clusters')
    plt.show()

    return wcss, silhouette_scores

def plot_umap_clusters(df, random_state=42):
    """
    Use UMAP to plot clusters in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame, which must contain a 'cluster' column.
    random_state (int): The random state for reproducibility.

    Returns:
    None
    """
    if 'cluster' not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster' column.")
    
    features = df.drop(columns='cluster')
    labels = df['cluster']
    
    umap_model = umap.UMAP(n_components=2, random_state=random_state)
    umap_components = umap_model.fit_transform(features)
    
    umap_df = pd.DataFrame(umap_components, columns=['x', 'y'])
    umap_df['cluster'] = labels.values
    
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='x', y='y', hue='cluster', data=umap_df, palette='tab10')
    plt.title('UMAP Visualization of Clusters')
    plt.show()

def plot_feature_importance(df, n_clusters=7, random_state=42):
    """
    Perform KMeans clustering, use Random Forest to determine feature importance,
    and plot the feature importances.

    Parameters:
    latent_rep_scaled (pd.DataFrame): The scaled latent representation data.
    n_clusters (int): The number of clusters for KMeans.
    random_state (int): The random state for reproducibility.

    Returns:
    None
    """
    new_df = df.copy()
    
    # KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
    cluster_labels = kmeans.fit_predict(new_df)

    new_df['cluster'] = cluster_labels

    # use Random Forest to determine feature importance
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(new_df.drop(columns='cluster'), new_df['cluster'])

    # get feature importances
    feature_importances = rf.feature_importances_
    features = new_df.columns[:-1]

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances, y=features, palette='viridis')
    plt.xlabel('Feature Importance')
    plt.ylabel('Feature')
    plt.title('Feature Importance based on Random Forest')
    plt.show()