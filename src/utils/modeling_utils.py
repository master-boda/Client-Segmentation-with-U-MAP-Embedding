import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift, estimate_bandwidth
from minisom import MiniSom 
from minisom import MiniSom
import math

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

def train_and_plot_som(df, som_shape=(10, 10), sigma=1.0, learning_rate=0.5, iterations=1000, plot=True, random_state=42):
    """
    Train a SOM, plot the distance map, and add clusters to the dataset.

    Parameters:
    df (pandas.DataFrame): The dataset to train the SOM.
    som_shape (tuple): The dimensions of the SOM (x, y).
    sigma (float): Sigma for SOM training.
    learning_rate (float): Learning rate for SOM training.
    iterations (int): Number of iterations for SOM training.
    plot (bool): If True, plot the distance map and activations.

    Returns:
    pandas.DataFrame: The original dataset with an additional column for clusters.
    """
    
    # Copy the original DataFrame
    new_df = df.copy()
    
    # Configure and initialize the SOM
    som = MiniSom(som_shape[0], som_shape[1], new_df.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=random_state)
    som.random_weights_init(new_df.values)
    
    # Train the SOM
    som.train_random(new_df.values, iterations)
    
    # Get the clusters for each data point
    clusters = np.array([som.winner(x) for x in new_df.values])
    
    # Add clusters to the original dataset
    new_df['cluster'] = clusters[:, 0] * som_shape[1] + clusters[:, 1]
    
    if plot:
        # Plot the distance map
        plt.figure(figsize=(10, 10))
        plt.title('SOM - Distance Map')
        plt.pcolor(som.distance_map().T, cmap='coolwarm')  # Distance matrix
        plt.colorbar()

        plt.show()
    
    return new_df


def train_and_plot_som_per_variable(df, som_shape=(10, 10), sigma=1.0, learning_rate=0.5, iterations=1000, random_state=42):
    """
    Train a SOM for each variable in the dataset and plot a grid of distance maps.

    Parameters:
    df (pandas.DataFrame): The dataset to train the SOM.
    som_shape (tuple): The dimensions of the SOM (x, y).
    sigma (float): Sigma for SOM training.
    learning_rate (float): Learning rate for SOM training.
    iterations (int): Number of iterations for SOM training.
    random_state (int): Random seed for reproducibility.
    """
    
    # Calculate the number of plots needed and the grid shape
    num_vars = df.shape[1]
    grid_cols = int(math.ceil(math.sqrt(num_vars)))
    grid_rows = int(math.ceil(num_vars / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(9, 9))
    axes = axes.flatten()

    for idx, column in enumerate(df.columns):
        # Extract the variable
        data = df[[column]].values
        
        # Initialize and train the SOM
        som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=random_state)
        som.random_weights_init(data)
        som.train_random(data, iterations)
        
        # Plot the distance map
        ax = axes[idx]
        ax.set_title(f'SOM - {column}')
        distance_map = som.distance_map().T
        im = ax.pcolor(distance_map, cmap='coolwarm')
        fig.colorbar(im, ax=ax)

    # Remove any unused subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()


def mean_shift_clustering(df, bandwidth=None, bin_seeding=False):
    """
    Perform Mean Shift clustering on each variable in the dataset and plot the results.

    Parameters:
    df (pandas.DataFrame): The dataset to perform clustering.
    bandwidth (float): The bandwidth parameter for the Mean Shift algorithm.
    bin_seeding (bool): If True, initial kernel locations are not locations of all points, but rather the location of the discretized version of points.

    Returns:
    pandas.DataFrame: The original dataset with an additional column for clusters for each variable.
    """
    
    # Copy the original DataFrame
    new_df = df.copy()
    
    # Calculate the number of plots needed and the grid shape
    num_vars = df.shape[1]
    grid_cols = int(math.ceil(math.sqrt(num_vars)))
    grid_rows = int(math.ceil(num_vars / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(15, 15))
    axes = axes.flatten()

    for idx, column in enumerate(df.columns):
        # Extract the variable
        data = df[[column]].values
        
        # Estimate the bandwidth if not provided
        if bandwidth is None:
            bandwidth = estimate_bandwidth(data, quantile=0.2, n_samples=500)
        
        # Perform Mean Shift clustering
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=bin_seeding)
        ms.fit(data)
        labels = ms.labels_
        
        # Add clusters to the original dataset
        new_df[f'cluster_{column}'] = labels
        
        # Plot the results
        ax = axes[idx]
        ax.set_title(f'Mean Shift - {column} (bandwidth={bandwidth:.2f})')
        scatter = ax.scatter(range(len(data)), data, c=labels, cmap='viridis')
        fig.colorbar(scatter, ax=ax)

    # Remove any unused subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

    return new_df