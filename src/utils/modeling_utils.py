import matplotlib.pyplot as plt
from minisom import MiniSom 
from minisom import MiniSom
import math

import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    
    # calculate the number of plots needed and the grid shape
    num_vars = df.shape[1]
    grid_cols = int(math.ceil(math.sqrt(num_vars)))
    grid_rows = int(math.ceil(num_vars / grid_cols))
    
    fig, axes = plt.subplots(grid_rows, grid_cols, figsize=(9, 9))
    axes = axes.flatten()

    for idx, column in enumerate(df.columns):
        # extract the variable
        data = df[[column]].values
        
        # initialize and train the SOM
        som = MiniSom(som_shape[0], som_shape[1], data.shape[1], sigma=sigma, learning_rate=learning_rate, random_seed=random_state)
        som.random_weights_init(data)
        som.train_random(data, iterations)
        
        # Plot the distance map
        ax = axes[idx]
        ax.set_title(f'SOM - {column}')
        distance_map = som.distance_map().T
        im = ax.pcolor(distance_map, cmap='coolwarm')
        fig.colorbar(im, ax=ax)

    # remove any unused subplots
    for j in range(idx+1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()

def export_clusters(data, clusters):
    """
    Export the clusters to separate CSV files.

    Args:
        data (pandas.DataFrame): The input data.
        clusters (str): The name of the column containing the cluster labels.
        base_dir (str): The path to the directory where the CSV files will be saved.

    Returns:
        None
    """
    base_dir = os.path.abspath(os.path.join(os.getcwd(), '../../data/clusters/'))
    
    os.makedirs(base_dir, exist_ok=True)
    for cluster in data[clusters].unique():
        cluster_data = data[data[clusters] == cluster].drop(columns=clusters)
        cluster_data.to_csv(os.path.join(base_dir, f'cluster_{cluster}.csv'))
