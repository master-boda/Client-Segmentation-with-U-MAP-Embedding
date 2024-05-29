import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import folium

def var_plotter(df, columns):
    """
    Plots histograms for specified columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    columns (list): List of column names to plot histograms for.
    """
    # remove rows with missing values in the specified columns
    df = df.dropna(subset=columns)

    # calculate the number of rows and columns for subplots
    num_plots = len(columns)
    num_cols_in_plot = 3
    num_rows = int(np.ceil(num_plots / num_cols_in_plot))

    plt.figure(figsize=(5 * num_cols_in_plot, 5 * num_rows))

    for i, col in enumerate(columns):
        plt.subplot(num_rows, num_cols_in_plot, i + 1)
        
        sns.histplot(df[col], kde=True, stat="density", color='blue')
        plt.ylabel('Density')
        
        plt.title(col)
        plt.xlabel(col)

    plt.tight_layout()
    plt.show()
