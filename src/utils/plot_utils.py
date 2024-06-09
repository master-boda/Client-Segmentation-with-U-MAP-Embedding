import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
import umap
from sklearn.metrics import silhouette_score
import os
import folium

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

    plt.figure(figsize=(4 * num_cols_in_plot, 4 * num_rows))

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
    
def plot_cluster(df, cluster_number):
    """
    Plots the data points of a specific cluster on a folium map.

    Args:
        df (pandas.DataFrame): The dataframe containing the data points.
        cluster_number (int): The cluster number to plot.

    Returns:
        folium.Map: The folium map object with the data points of the specified cluster plotted.
    """
    m = folium.Map(location=[39.3999, -8.2245], zoom_start=6)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'darkred']

    df_cluster = df[df['cluster'] == cluster_number]

    for index, row in df_cluster.iterrows():
        folium.CircleMarker(
            location=[row['latitude'], row['longitude']],
            radius=0.1,
            color=colors[cluster_number % len(colors)],
            fill=True,
            fill_color=colors[cluster_number % len(colors)]
        ).add_to(m)

    return m

def plot_pie_chart(data, variable, colors, labels=None, legend=[], autopct='%1.1f%%'):
    """
    Plot a pie chart based on the values of a variable in the given data.

    Args:
        data (pandas.DataFrame): The input data containing the variable.
        variable (str): The name of the variable to plot.
        colors (list): The colors for each pie slice.
        labels (list, optional): The labels for each pie slice. Defaults to None.
        legend (list, optional): The legend labels. Defaults to [].
        autopct (str, optional): The format for autopct labels. Defaults to '%1.1f%%'.

    Returns:
        None
    """
    counts = data[variable].value_counts()  # count the occurrences of each value in the variable

    plt.pie(counts, colors=colors, labels=labels, startangle=90, autopct=autopct, textprops={'fontsize': 25})
    
    if len(legend) != 0:
        plt.legend(legend, fontsize=16, bbox_to_anchor=(0.7, 0.9))  # Add a legend if provided
    
    plt.show()

