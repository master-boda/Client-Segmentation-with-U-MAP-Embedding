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

# data Importing
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))

basket_path = os.path.join(base_dir, 'customer_basket.csv')
customer_info_path = os.path.join(base_dir, 'customer_info.csv')
product_path = os.path.join(base_dir, 'product_mapping.xlsx')

basket = pd.read_csv(basket_path, index_col='invoice_id')
customer_info = pd.read_csv(customer_info_path, index_col='customer_id')
product = pd.read_excel(product_path)

# data cleaning before pipeline
customer_info_clean = clean_customer_data(customer_info)

# column names for different types of variables (continuous, discrete, binary), important for different types of preprocessing in the pipeline
# note: some discrete variables are treated as continuous in the pipeline
customer_info_cols_disc = ['kids_home', 'teens_home', 'number_complaints', 'distinct_stores_visited', 'educ_years']
customer_info_cols_binary = ['loyalty_member', 'gender_binary']
customer_info_cols_cont = customer_info_clean.columns.difference(customer_info_cols_disc + customer_info_cols_binary)

#customer_info_clean, outliers = remove_outliers_percentile(customer_info_clean, customer_info_cols_cont)

def preproc_pipeline(df):
    # separate the data into different types of variables
    df_cont = df[customer_info_cols_cont]
    
    cont_imputer = SimpleImputer(strategy='median')
    cont_imputer.fit(df_cont)
    df_cont_imputed = pd.DataFrame(cont_imputer.transform(df_cont), columns=df_cont.columns, index=df_cont.index)

    df_cont_new_features = feat_engineering(df_cont_imputed)
    
    df_cont_sqrt = sqrt_transform(df_cont_new_features, df_cont_new_features.columns)
    
    cont_scaler = RobustScaler()
    cont_scaler.fit(df_cont_sqrt)
    df_cont_preproc = pd.DataFrame(cont_scaler.transform(df_cont_sqrt), columns=df_cont_sqrt.columns, index=df_cont_sqrt.index)
    
#    df_cont_preproc, b = remove_outliers_dbscan(df_cont_scaled)
    
    return df_cont_preproc

customer_info_preproc = preproc_pipeline(customer_info_clean)
base_dir = 'data/processed'
customer_info_preproc.to_csv(os.path.join(base_dir, 'customer_info_preproc_test.csv'))