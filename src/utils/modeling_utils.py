import matplotlib.pyplot as plt
import os

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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