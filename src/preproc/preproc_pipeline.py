import sys
import os
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.impute import KNNImputer

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

from utils.preproc_utils import *

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
    cols_disc = ['kids_home', 'teens_home', 'distinct_stores_visited', 'educ_years', 'number_complaints']
    cols_binary = ['loyalty_member', 'gender_binary']
    cols_cont_and_disc = customer_info_clean.columns.difference(cols_binary)
    
    # filter for continuous and numerical discrete variables
    df_filtered = df[cols_cont_and_disc]
    
    # scale the data for KNN imputer
    df_scaled = pd.DataFrame(scaler.fit_transform(df_filtered), columns=df_filtered.columns, index=df_filtered.index)
    
    # impute missing values with KNN imputer
    imputer = KNNImputer(n_neighbors=5)
    df_imputed = pd.DataFrame(imputer.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index)
    
    # remove fishy outliers
    df_imputed, fishy_outliers = remove_fishy_outliers(df_imputed)
    
    # scale the data back to the original scale for feature engineering
    df_imputed_original_scale = pd.DataFrame(scaler.inverse_transform(df_imputed), columns=df_imputed.columns, index=df_imputed.index)

    # create new features
    df_new_features = feat_engineering(df_imputed_original_scale)
    
    # assure that only continuous variables are included in the dataframe before applying transformations
    new_binary_cols = ['has_complaints', 'has_offspring', 'has_educ']
    df_new_features_cont = df_new_features[df_new_features.columns.difference(cols_disc + new_binary_cols)]
    
    # apply sqrt transformation to the dataframe to reduce skewness and outlier impact
    df_sqrt = sqrt_transform(df_new_features_cont)

    # scale the data again after the transformations
    df_scaled = pd.DataFrame(scaler.fit_transform(df_sqrt), columns=df_sqrt.columns, index=df_sqrt.index)

    # there are no missing values in the binary columns, we can safely merge them with the rest of the data
    # note: feature engeneering created 3 additional binary columns
    # final dataframe has a mix of continuous, and binary variables
    df_scaled = pd.merge(df_scaled, df_new_features[new_binary_cols], how='left', on='customer_id')
    df_scaled = pd.merge(df_scaled, df[cols_binary], how='left', on='customer_id')    

    print('Preprocessing Pipeline Completed')
    print(df_scaled.info())
    return df_scaled, fishy_outliers

# data cleaning before pipeline
customer_info_clean = clean_customer_data(customer_info)

customer_info_preproc, fishy_outliers = preproc_pipeline_customer_info(customer_info_clean, scaler=StandardScaler())

base_dir = 'data/processed'
customer_info_preproc.to_csv(os.path.join(base_dir, 'customer_info_preproc.csv'))

# seperate the fishy outliers
# as their values differ significantly from the rest of the data, it is not viable to analyze them together with the other outliers
fishy_outliers.to_csv(os.path.join(base_dir, 'fishy_outliers.csv'))