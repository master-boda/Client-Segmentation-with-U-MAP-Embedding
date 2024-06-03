import sys
import os
import pandas as pd
import numpy as np 
import folium   
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.preproc_utils import *
from preproc.autoencoder_mixed_datatypes import run_autoencoder

# data Importing
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))

basket_path = os.path.join(base_dir, 'customer_basket.csv')
customer_info_path = os.path.join(base_dir, 'customer_info.csv')
product_path = os.path.join(base_dir, 'product_mapping.xlsx')

basket = pd.read_csv(basket_path, index_col='invoice_id')
customer_info = pd.read_csv(customer_info_path, index_col='customer_id')
product = pd.read_excel(product_path)

def preproc_pipeline_customer_info(df, scaler=RobustScaler()):
    
    # column names for different types of variables (continuous, discrete, binary), important for different types of preprocessing in the pipeline
    # note: some discrete variables are treated as continuous in the pipeline
    cols_binary = [
    'loyalty_member', 
    'gender_binary',
    'kids_home_under_2', 
    'kids_home_over_2', 
    'teens_home_under_2', 
    'teens_home_over_2', 
    'number_complaints_under_1', 
    'number_complaints_over_1', 
    'distinct_stores_visited_under_3', 
    'distinct_stores_visited_over_3', 
    'educ_years_under_15', 
    'educ_years_over_15']
    
    cols_cont = df.columns[~df.columns.isin(cols_binary)]
    
    df_filtered, df_outliers = remove_outliers_iqr(df, cols_cont)
    
    # pipeline for continuous variables        
    df_filtered_cont = df_filtered[cols_cont]
    df_outliers_cont = df_outliers[cols_cont]
    
    imputer = SimpleImputer(strategy='median')
    imputer.fit(df_filtered_cont)
    df_imputed_cont = pd.DataFrame(imputer.transform(df_filtered_cont), columns=df_filtered_cont.columns, index=df_filtered_cont.index)
    df_imputed_cont_outliers = pd.DataFrame(imputer.transform(df_outliers_cont), columns=df_outliers_cont.columns, index=df_outliers_cont.index)

    df_new_features = feat_engineering(df_imputed_cont)
    df_new_features_outliers = feat_engineering(df_imputed_cont_outliers)
    
    df_sqrt = sqrt_transform(df_new_features)
    df_sqrt_outliers = sqrt_transform(df_new_features_outliers)
    
    scaler = scaler
    scaler.fit(df_sqrt)
    df_scaled = pd.DataFrame(scaler.transform(df_sqrt), columns=df_sqrt.columns, index=df_sqrt.index)
    df_scaled_outliers = pd.DataFrame(scaler.transform(df_sqrt_outliers), columns=df_sqrt_outliers.columns, index=df_sqrt_outliers.index)
    
    # pipeline for binary variables
    df_filtered_binary = df_filtered[cols_binary]
    df_outliers_binary = df_outliers[cols_binary]
    
    imputer = SimpleImputer(strategy='most_frequent')
    imputer.fit(df_filtered_binary)
    df_imputed_binary = pd.DataFrame(imputer.transform(df_filtered_binary), columns=df_filtered_binary.columns, index=df_filtered_binary.index)
    df_imputed_binary_outliers = pd.DataFrame(imputer.transform(df_outliers_binary), columns=df_outliers_binary.columns, index=df_outliers_binary.index)
    
    # merge the continuous and binary variables
    df_filtered = pd.merge(df_scaled, df_imputed_binary, on='customer_id')
    df_outliers = pd.merge(df_scaled_outliers, df_imputed_binary_outliers, on='customer_id')
    
    cols_cont = df_filtered.columns[~df_filtered.columns.isin(cols_binary)]

    run_autoencoder(df_filtered, cols_cont, cols_binary, 'data/processed/latent_representation.csv', epochs=50, batch_size=32, latent_dim=4)
    run_autoencoder(df_outliers, cols_cont, cols_binary, 'data/processed/latent_representation_outliers.csv', epochs=50, batch_size=32, latent_dim=4)
        
    print('Preprocessing Pipeline Completed')
    
    return df_filtered, df_outliers # return the preprocessed DataFrame without passing through autoencoder

# data cleaning before pipeline
customer_info_clean = clean_customer_data(customer_info)
customer_info_clean_binned = binning(customer_info_clean)

customer_info_preproc, customer_info_preproc_outliers = preproc_pipeline_customer_info(customer_info_clean_binned, scaler=MinMaxScaler())

base_dir = 'data/processed'
customer_info_preproc.to_csv(os.path.join(base_dir, 'customer_info_preproc.csv'))
customer_info_preproc_outliers.to_csv(os.path.join(base_dir, 'customer_info_preproc_outliers.csv'))