import sys
import os
import pandas as pd
import numpy as np 
import folium   
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.preproc_utils import *
from autoencoder import run_autoencoder

# data Importing
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))

basket_path = os.path.join(base_dir, 'customer_basket.csv')
customer_info_path = os.path.join(base_dir, 'customer_info.csv')
product_path = os.path.join(base_dir, 'product_mapping.xlsx')

basket = pd.read_csv(basket_path, index_col='invoice_id')
customer_info = pd.read_csv(customer_info_path, index_col='customer_id')
product = pd.read_excel(product_path)

def preproc_pipeline_customer_info(df, scaler=MinMaxScaler()):
    
    # column names for different types of variables (continuous, discrete, binary), important for different types of preprocessing in the pipeline
    # note: some discrete variables are treated as continuous in the pipeline (age, typical_hour)
        
    df['has_offspring'] = 1*(df['kids_home'] + df['teens_home'] > 0)
    df['has_complaints'] = 1*(df['number_complaints'] > 2)
    df['has_educ'] = 1*(df['educ_years'] > 12)
    
    cols_disc = ['kids_home', 'teens_home', 'distinct_stores_visited', 'educ_years', 'number_complaints']#, 'typical_hour', 'age']
    cols_binary = ['loyalty_member', 'gender_binary', 'has_offspring', 'has_complaints', 'has_educ']
    cols_cont = customer_info_clean.columns.difference(cols_disc + cols_binary)
    
    # filter for continuous variables
    df_filtered = df[cols_cont]
        
    df_scaled = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns, index=df_filtered.index)
    
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)
    
    df_imputed, fishy_outliers = remove_fishy_outliers(df_imputed)
    df_imputed, iso_outliers = isolation_forest(df_imputed, df_imputed.columns, contamination=0.01)

    df_outliers = pd.concat([fishy_outliers, iso_outliers]).drop_duplicates()
    
    df_imputed_original_scale = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)
    df_imputed_original_scale_outliers = pd.DataFrame(scaler.inverse_transform(df_outliers), columns=df_outliers.columns, index=df_outliers.index)

    df_new_features = feat_engineering(df_imputed_original_scale)
    df_new_features_outliers = feat_engineering(df_imputed_original_scale_outliers)
    
    df_sqrt = sqrt_transform(df_new_features)
    df_sqrt_outliers = sqrt_transform(df_new_features_outliers)

    df_scaled = pd.DataFrame(scaler.fit_transform(df_sqrt), columns=df_sqrt.columns, index=df_sqrt.index)
    df_scaled_outliers = pd.DataFrame(scaler.transform(df_sqrt_outliers), columns=df_sqrt_outliers.columns, index=df_sqrt_outliers.index)

    df_scaled = pd.merge(df_scaled, df[cols_binary], how='left', on='customer_id')    
    df_scaled_outliers = pd.merge(df_scaled_outliers, df[cols_binary], how='left', on='customer_id')
    
    #latent_representation, encoder = run_autoencoder(df_scaled, epochs=30, batch_size=32, latent_dim=4)
    #latent_representation_outliers = run_autoencoder(df_scaled_outliers, encoder=encoder)
        
    #latent_df = pd.DataFrame(latent_representation, index=df_scaled.index, columns=[f'latent_{i}' for i in range(latent_representation.shape[1])])
    #latent_df.to_csv('data/processed/latent_representation.csv')
    
    #latent_df_outliers = pd.DataFrame(latent_representation_outliers, index=df_scaled_outliers.index, columns=[f'latent_{i}' for i in range(latent_representation_outliers.shape[1])])
    #latent_df_outliers.to_csv('data/processed/latent_representation_outliers.csv')
    
    print('Preprocessing Pipeline Completed')
    
    return df_scaled, df_scaled_outliers # return the preprocessed DataFrame without passing through autoencoder

# data cleaning before pipeline
customer_info_clean = clean_customer_data(customer_info)

customer_info_preproc, customer_info_preproc_outliers = preproc_pipeline_customer_info(customer_info_clean, scaler=StandardScaler())

base_dir = 'data/processed'
customer_info_preproc.to_csv(os.path.join(base_dir, 'customer_info_preproc.csv'))
customer_info_preproc_outliers.to_csv(os.path.join(base_dir, 'customer_info_preproc_outliers.csv'))