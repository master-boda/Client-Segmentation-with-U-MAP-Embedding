import pandas as pd
from sklearn.cluster import KMeans
import umap
import matplotlib.pyplot as plt
import sys
import os
import numpy as np
import umap
import seaborn as sns

sys.path.append(os.path.abspath('..')) 

from utils.preproc_utils import *
from utils.plot_utils import *
from utils.modeling_utils import *

sys.path.append(os.path.abspath('..')) 
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/processed'))

preproc_path = os.path.join(base_dir, 'customer_info_preproc.csv')
customer_info_preproc = pd.read_csv(preproc_path, index_col='customer_id')

preproc_path_outliers = os.path.join(base_dir, 'customer_info_preproc_outliers.csv')
customer_info_prepro_outliers = pd.read_csv(preproc_path_outliers, index_col='customer_id')

def run_clustering(df, n_clusters=7, random_state=42, visualize=True):
    
    umap_reducer = umap.UMAP(n_neighbors=35, min_dist=0.0, n_components=2, random_state=random_state)
    umap_embeddings = umap_reducer.fit_transform(customer_info_preproc)

    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans_labels = kmeans.fit_predict(umap_embeddings)

    umap_df = pd.DataFrame(umap_embeddings, columns=['UMAP1', 'UMAP2'], index=df.index)
    umap_df['cluster'] = kmeans_labels

    if visualize:
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='UMAP1', y='UMAP2', hue='cluster', data=umap_df, palette='tab10')
        plt.title('UMAP projection with KMeans clusters')
        plt.show()
    
    return umap_df

umap_df = run_clustering(customer_info_preproc, n_clusters=7, random_state=42, visualize=True)

customer_info_preproc_labeled = pd.merge(customer_info_preproc, umap_df['cluster'], how='left', on='customer_id')