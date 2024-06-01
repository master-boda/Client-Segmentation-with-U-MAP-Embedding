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

def preproc_pipeline_customer_info(df, scaler=RobustScaler()):
    
    # column names for different types of variables (continuous, discrete, binary), important for different types of preprocessing in the pipeline
    # note: some discrete variables are treated as continuous in the pipeline
    cols_disc = ['kids_home', 'teens_home', 'number_complaints', 'distinct_stores_visited', 'educ_years']
    cols_binary = ['loyalty_member', 'gender_binary']
    cols_cont = customer_info_clean.columns.difference(cols_disc + cols_binary)
    
    # filter for continuous variables
    df_filtered = df[cols_cont]
    
    cont_imputer = SimpleImputer(strategy='median')
    cont_imputer.fit(df_filtered)
    df_cont_imputed = pd.DataFrame(cont_imputer.transform(df_filtered), columns=df_filtered.columns, index=df_filtered.index)

    df_cont_new_features = feat_engineering(df_cont_imputed)
    
    df_cont_sqrt = sqrt_transform(df_cont_new_features)
    
    cont_scaler = scaler
    cont_scaler.fit(df_cont_sqrt)
    df_cont_scale = pd.DataFrame(cont_scaler.transform(df_cont_sqrt), columns=df_cont_sqrt.columns, index=df_cont_sqrt.index)
    
    run_autoencoder(df_cont_scale, 'data/processed/latent_representation.csv', epochs=50, batch_size=32, latent_dim=6)
        
    print('Preprocessing Pipeline Completed')
    
    return df_cont_scale # return the preprocessed DataFrame without passing through autoencoder

# data cleaning before pipeline
customer_info_clean = clean_customer_data(customer_info)

customer_info_preproc = preproc_pipeline_customer_info(customer_info_clean, scaler=RobustScaler())

base_dir = 'data/processed'
customer_info_preproc.to_csv(os.path.join(base_dir, 'customer_info_preproc.csv'))