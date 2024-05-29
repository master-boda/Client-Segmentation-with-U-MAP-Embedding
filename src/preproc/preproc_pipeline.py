import sys
import os
import pandas as pd
import numpy as np 
import folium   
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.decomposition import PCA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.preproc_utils import *

# Data Importing

base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))

basket_path = os.path.join(base_dir, 'customer_basket.csv')
customer_info_path = os.path.join(base_dir, 'customer_info.csv')
product_path = os.path.join(base_dir, 'product_mapping.xlsx')

# Load the data
basket = pd.read_csv(basket_path, index_col='invoice_id')
customer_info = pd.read_csv(customer_info_path, index_col='customer_id')
product = pd.read_excel(product_path)

# Data Cleaning, Transformations and Scaling (scaling is fundamental for KNN-Imputation and DBSCAN outlier detection)

# cleaning (no information is gained nor lost, mostly renaming columns and data type conversion)
customer_info_clean = clean_customer_data(customer_info)

# dropping coords columns (locations outside Lisbon are huge outliers)
customer_info_clean.drop(columns=['latitude', 'longitude'], inplace=True)

# note: 'total_distinct_products' is being treated as a continuous variable
customer_info_cont = ['spend_groceries', 'spend_electronics', 'spend_vegetables', 'spend_nonalcohol_drinks', 'spend_alcohol_drinks',
                      'spend_meat', 'spend_fish', 'spend_hygiene', 'spend_videogames', 'spend_petfood', 'percentage_of_products_bought_promotion', 'total_distinct_products']

customer_info_discrete = ['kids_home', 'teens_home', 'number_complaints', 'distinct_stores_visited', 'typical_hour', 'age', 'years_as_customer', 'educ_years']

# square root transformation on continuous variables due to right-skewness
customer_info_cont_sqrt = sqrt_transform(customer_info_clean, customer_info_cont)

# scaling (only continuous variables) and imputation of missing values (continuous variables use KNN imputation, discrete variables use mode imputation)
customer_info_cont_scaled = scale_and_impute(customer_info_cont_sqrt, customer_info_cont, customer_info_discrete, MinMaxScaler())

# data exporting (useful for further exploratary data analysis)
customer_info_cont_scaled.to_csv(os.path.join(base_dir, 'customer_info_cont_scaled.csv'))

pca = PCA(n_components=6)
principal_components = pca.fit_transform(customer_info_cont_scaled)

df_principal_components = pd.DataFrame(principal_components, columns=[f'PC{i+1}' for i in range(6)], index=customer_info_cont_scaled.index)

df_principal_components.to_csv(os.path.join(base_dir, 'customer_info_pca.csv'))