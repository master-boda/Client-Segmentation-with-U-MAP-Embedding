import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cluster import DBSCAN

def refactor_column_names(df):
    """
    Refactors the column names of a DataFrame by removing common prefixes.

    This function checks if there are 3 or more columns starting with the same prefix 
    (e.g., 'customer_' or 'lifetime_') and if so, removes that prefix from the column names.

    Parameters:
    df (pandas.DataFrame): The DataFrame whose column names are to be refactored.

    Returns:
    pandas.DataFrame: The DataFrame with refactored column names.
    """
    prefixes = {}
    for col in df.columns:
        prefix = col.split('_')[0] + '_'
        if prefix not in prefixes:
            prefixes[prefix] = [col]
        else:
            prefixes[prefix].append(col)

    for prefix, cols in prefixes.items():
        if len(cols) >= 3:
            refactored_columns = {col: col.replace(prefix, '') for col in cols}
            df.rename(columns=refactored_columns, inplace=True)

    return df

def calculate_age(df, birthdate_column):
    """
    Calculates the age of customers based on their birthdate.

    This function takes a DataFrame and the name of the column containing birthdates,
    converts the birthdates to datetime format, calculates the age of each customer based on the current date,
    and adds a new column 'age' to the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing customer information.
    birthdate_column (str): The name of the column containing birthdates.

    Returns:
    pandas.DataFrame: The DataFrame with a new 'age' column.
    """
    df[birthdate_column] = pd.to_datetime(df[birthdate_column])
    today = datetime.today()
    
    df['age'] = today.year - df[birthdate_column].dt.year
    
    # create a boolean series indicating whether the birthday has occurred this year
    birthday_passed = (today.month > df[birthdate_column].dt.month) | ((today.month == df[birthdate_column].dt.month) & (today.day >= df[birthdate_column].dt.day))
    
    # boolean is converted to an integer (True -> 1, False -> 0) and subtracted from age
    # so if the birthday has already occurred this year, the age will remain the same (~birthday_passed = False -> 0, age - 0 = age)
    # if the birthday has not occurred yet, the age will be decremented by 1 (~birthday_passed = True -> 1, age - 1 = age - 1)
    df['age'] -= ~birthday_passed
    
    return df
    
def add_education_years(df, name_column):
    """
    Extracts the degree prefix from the name column, maps it to corresponding education years, 
    and adds a new column for education years.

    Parameters:
    df (pd.DataFrame): The input dataframe containing customer information.
    name_column (str): The name of the column containing customer names with degree prefixes.

    Returns:
    pd.DataFrame: The input dataframe with an added column for education years.
    """
    def extract_degree(name):
        if pd.notnull(name) and '.' in name:
            parts = name.split()
            if len(parts) > 1 and '.' in parts[0]:
                return parts[0].split('.')[0].lower()
        return 'no_degree'
    
    def get_educ_years(degree):
        if degree == 'bsc':
            return 15
        elif degree in ['msc', 'msh']:
            return 17
        elif degree == 'phd':
            return 20
        return 12  # default for 'no_degree' or no degree

    # extract the degree prefix and map it to education years
    df['degree'] = df[name_column].apply(extract_degree)
    df['educ_years'] = df['degree'].apply(get_educ_years)

    # drop the temporary 'degree' column
    df.drop(['degree'], axis=1, inplace=True)

    return df

def clean_customer_data(df):
    """
    Cleans and preprocesses the customer data.

    This function performs several preprocessing steps including:
    - Removing repeating prefixes in column names
    - Dropping the first column (assumed to be an index column)
    - Calculating age from the birthdate column and dropping the birthdate column
    - Creating a binary column for loyalty membership and dropping the original column
    - Calculating years as a customer from the year_first_transaction column and dropping the original column
    - Extracting education years from the name column and dropping the name column
    - Converting gender to a binary column and dropping the original column
    - Dropping coordinates columns

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing customer information.

    Returns:
    pandas.DataFrame: The cleaned and preprocessed DataFrame.
    """
    # remove repeating prefixes in column names
    df = refactor_column_names(df)
    
    # drop the first column which is assumed to be an index column
    if 'Unnamed: 0' in df.columns:
        df.drop('Unnamed: 0', axis=1, inplace=True)

    # swap "birthdate" column with "age" column
    df = calculate_age(df, 'birthdate')
    df.drop('birthdate', axis=1, inplace=True)

    # swap "loyalty_card_number" column with "loyalty_member" column
    df['loyalty_member'] = np.where(df['loyalty_card_number'].isna(), 0, 1)
    df.drop('loyalty_card_number', axis=1, inplace=True)

    # swap "year_first_transaction" column with "years_as_customer" column
    df['years_as_customer'] = datetime.now().year - df['year_first_transaction']
    df.drop('year_first_transaction', axis=1, inplace=True)

    # extract the education years from the "name" column
    df = add_education_years(df, 'name')
    df.drop('name', axis=1, inplace=True)

    # change gender from string to binary
    df['gender_binary'] = np.where(df['gender'] == 'male', 1, 0)
    df.drop('gender', axis=1, inplace=True)

    df.drop(columns=['latitude', 'longitude'], inplace=True)

    return df

def binning(df):
    """
    Swaps specified variables for new binary dummy variables using binning.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    binning_dict (dict): A dictionary where keys are column names and values are tuples defining the bins.

    Returns:
    pd.DataFrame: The DataFrame with new binary dummy variables added.
    """
    
    binning_dict = {
        'kids_home': [0, 2, float('inf')],
        'teens_home': [0, 2, float('inf')],
        'number_complaints': [0, 1, float('inf')],
        'distinct_stores_visited': [0, 3, float('inf')],
        'educ_years': [12, 15, float('inf')]
    }

    for col, bins in binning_dict.items():
        bin_labels = [f"{col}_under_{bins[1]}", f"{col}_over_{bins[1]}"]
        df[f'{col}_binned'] = pd.cut(df[col], bins=bins, include_lowest=True, labels=bin_labels)

        # create binary dummy variables for each bin
        for bin_label in bin_labels:
            df[bin_label] = (df[f'{col}_binned'] == bin_label).astype(int)

        # drop the temporary binned column
        df.drop(columns=[f'{col}_binned'], inplace=True)
        
    df.drop(columns=list(binning_dict.keys()), inplace=True)
        
    return df
    
def feat_engineering(df):
    """
    Adds new columns to the DataFrame representing the proportion of each spending category
    relative to the total spending in specified categories, and applies binning to specific variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with new proportion columns, monetary column, and binary dummy variables added.
    """
    spend_cols = ['spend_groceries', 'spend_electronics', 'spend_vegetables', 
                  'spend_nonalcohol_drinks', 'spend_alcohol_drinks', 'spend_meat', 
                  'spend_fish', 'spend_hygiene', 'spend_videogames']
    
    df['monetary'] = df[spend_cols].sum(axis=1)

    # calculate the proportion for each spending category
    for col in spend_cols:
        proportion_col = f'{col}_proportion'
        df[proportion_col] = df[col] / df['monetary']
     
    df.drop(columns=spend_cols, inplace=True)
    
    return df

def sqrt_transform(df):
    """
    Applies square root transformation to the specified columns in the dataframe.
    Shifts the data if there are negative values in a column to make all values in that column non-negative.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the data.
    columns (list of str): The list of column names to apply the transformation to.
    
    Returns:
    pd.DataFrame: A dataframe with the square root transformation applied to the specified columns.
    """
    transformed_data = df.copy()
    
    for column in df.columns:
        min_value = df[column].min()
        if min_value < 0:
            # shift the data by adding a constant to make all values non-negative
            shift_value = abs(min_value)
            transformed_data[column] = np.sqrt(df[column] + shift_value)
            print(f"Column '{column}' was shifted by {shift_value} to handle negative values.")
        else:
            # apply the square root transformation directly
            transformed_data[column] = np.sqrt(df[column])
    
    return transformed_data   

def remove_outliers_percentile(df, lower_percentile=0.01, upper_percentile=0.99):
    """
    Removes outliers from specified columns in a DataFrame using the percentile method.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    columns (list): List of column names to check for outliers.
    lower_percentile (float): The lower percentile threshold. Default is 0.01 (1st percentile).
    upper_percentile (float): The upper percentile threshold. Default is 0.99 (99th percentile).

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing only outliers.
    """
    initial_row_count = df.shape[0]
    outliers = pd.DataFrame()
    
    for column in df.columns:
        lower_bound = df[column].quantile(lower_percentile)
        upper_bound = df[column].quantile(upper_percentile)
        
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers = pd.concat([outliers, column_outliers])
        
        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]
    
    outliers = outliers.drop_duplicates()
    final_row_count = df.shape[0]
    num_rows_removed = initial_row_count - final_row_count
    percentage_removed = (num_rows_removed / initial_row_count) * 100
    
    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage of dataset removed: {percentage_removed:.2f}%")
    
    return df, outliers

def remove_outliers_manual(df):
    """
    Removes outliers from specified columns in a DataFrame based on manual thresholds.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    thresholds (dict): Dictionary where keys are column names and values are tuples (lower_bound, upper_bound).

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing only outliers.
    """
    
    thresholds = {
        'spend_videogames' : (1, 2200),
        'spend_fish' : (1, 3000),
        'spend_meat' : (3, 4000),
        'spend_electronics' : (0, 8000),
        'spend_petfood' : (0, 4200)
    }
    
    initial_row_count = df.shape[0]
    outliers = pd.DataFrame()

    for column, (lower_bound, upper_bound) in thresholds.items():
        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers = pd.concat([outliers, column_outliers])

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    outliers = outliers.drop_duplicates()
    final_row_count = df.shape[0]
    num_rows_removed = initial_row_count - final_row_count
    percentage_removed = (num_rows_removed / initial_row_count) * 100

    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage of dataset removed: {percentage_removed:.2f}%")

    return df, outliers

def remove_outliers_dbscan(df, eps=0.5, min_samples=5):
    """
    Removes multidimensional outliers from a DataFrame using DBSCAN.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
    min_samples (int): The number of samples (or total weight) in a neighborhood for a point to be considered as a core point.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing only outliers.
    """
    # Fit DBSCAN
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(df)

    # Identify outliers (labeled as -1 by DBSCAN)
    outliers = df[labels == -1]
    df_cleaned = df[labels != -1]

    # Calculate the number of rows removed and the percentage of the dataset removed
    num_rows_removed = len(outliers)
    initial_row_count = len(df)
    percentage_removed = (num_rows_removed / initial_row_count) * 100

    # Print the number of rows removed and the percentage of the dataset removed
    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage of dataset removed: {percentage_removed:.2f}%")

    return df_cleaned, outliers

def remove_outliers_iqr(df):
    """
    Removes outliers from specified columns in a DataFrame using the IQR method.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing only outliers.
    """
    initial_row_count = df.shape[0]
    outliers = pd.DataFrame()

    for column in df.columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        column_outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
        outliers = pd.concat([outliers, column_outliers])

        df = df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

    outliers = outliers.drop_duplicates()
    final_row_count = df.shape[0]
    num_rows_removed = initial_row_count - final_row_count
    percentage_removed = (num_rows_removed / initial_row_count) * 100

    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage of dataset removed: {percentage_removed:.2f}%")

    return df, outliers