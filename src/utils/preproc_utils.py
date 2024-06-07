import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.ensemble import IsolationForest
import os


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

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
    new_df = df.copy()
    
    prefixes = {}
    for col in new_df.columns:
        prefix = col.split('_')[0] + '_'
        if prefix not in prefixes:
            prefixes[prefix] = [col]
        else:
            prefixes[prefix].append(col)

    for prefix, cols in prefixes.items():
        if len(cols) >= 3:
            refactored_columns = {col: col.replace(prefix, '') for col in cols}
            new_df.rename(columns=refactored_columns, inplace=True)

    return new_df

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
    new_df = df.copy()
    
    new_df[birthdate_column] = pd.to_datetime(new_df[birthdate_column])
    today = datetime.today()
    
    new_df['age'] = today.year - new_df[birthdate_column].dt.year
    
    # create a boolean series indicating whether the birthday has occurred this year
    birthday_passed = (today.month > new_df[birthdate_column].dt.month) | ((today.month == new_df[birthdate_column].dt.month) & (today.day >= new_df[birthdate_column].dt.day))
    
    # boolean is converted to an integer (True -> 1, False -> 0) and subtracted from age
    # so if the birthday has already occurred this year, the age will remain the same (~birthday_passed = False -> 0, age - 0 = age)
    # if the birthday has not occurred yet, the age will be decremented by 1 (~birthday_passed = True -> 1, age - 1 = age - 1)
    new_df['age'] -= ~birthday_passed
    
    return new_df
    
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
    new_df = df.copy()
    
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
    new_df['degree'] = new_df[name_column].apply(extract_degree)
    new_df['educ_years'] = new_df['degree'].apply(get_educ_years)

    # drop the temporary 'degree' column
    new_df.drop(['degree'], axis=1, inplace=True)

    return new_df

def clean_customer_data(df):
    """
    Cleans and preprocesses the customer data.

    This function performs several preprocessing steps including:
    - Removing repeating prefixes in column names
    - Dropping the first column (assumed to be an index column)
    - Calculating age from the birthdate column and dropping the birthdate column
    - Creating a binary column for loyalty membership and dropping the original column
    - Calculating years as a customer from the year_first_transaction column and dropping the original column
    - Extracting education years from the name column and dropping the original column
    - Converting gender to a binary column and dropping the original column
    - Dropping coordinates columns

    Parameters:
    df (pandas.DataFrame): The input DataFrame containing customer information.

    Returns:
    pandas.DataFrame: The cleaned and preprocessed DataFrame.
    """
    new_df = df.copy()
    
    # remove repeating prefixes in column names
    new_df = refactor_column_names(new_df)
    
    # drop the first column which is assumed to be an index column
    if 'Unnamed: 0' in df.columns:
        new_df.drop('Unnamed: 0', axis=1, inplace=True)

    # swap "birthdate" column with "age" column
    new_df = calculate_age(new_df, 'birthdate')
    new_df.drop('birthdate', axis=1, inplace=True)

    # swap "loyalty_card_number" column with "loyalty_member" column
    new_df['loyalty_member'] = np.where(new_df['loyalty_card_number'].isna(), 0, 1)
    new_df.drop('loyalty_card_number', axis=1, inplace=True)

    # swap "year_first_transaction" column with "years_as_customer" column
    new_df['years_as_customer'] = datetime.now().year - new_df['year_first_transaction']
    new_df.drop('year_first_transaction', axis=1, inplace=True)

    # extract the education years from the "name" column
    new_df = add_education_years(new_df, 'name')
    new_df.drop('name', axis=1, inplace=True)
    
    # change gender from string to binary
    new_df['gender_binary'] = np.where(new_df['gender'] == 'male', 1, 0)
    new_df.drop('gender', axis=1, inplace=True)

    new_df.drop(columns=['latitude', 'longitude'], inplace=True)

    return new_df
    
def feat_engineering(df):
    """
    Adds new columns to the DataFrame representing the proportion of each spending category
    relative to the total spending in specified categories, and applies binning to specific variables.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with new proportion columns, monetary column, and binary dummy variables added.
    """
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))
    basket = pd.read_csv(os.path.join(base_dir, 'customer_basket.csv'))

    freq = basket.groupby('customer_id').size().reset_index(name='frequency')
    freq.set_index('customer_id', inplace=True)

    new_df = df.copy()
    
    new_df = pd.merge(new_df, freq, how='left', on='customer_id')
    new_df['frequency'].fillna(0, inplace=True)
    
    spend_cols = ['spend_groceries', 'spend_electronics', 'spend_vegetables', 
                  'spend_nonalcohol_drinks', 'spend_alcohol_drinks', 'spend_meat', 
                  'spend_fish', 'spend_hygiene', 'spend_videogames', 'spend_petfood']
    
    new_df['monetary'] = new_df[spend_cols].sum(axis=1)

    # calculate the proportion for each spending category
    for col in spend_cols:
        proportion_col = f'{col}_proportion'
        new_df[proportion_col] = new_df[col] / new_df['monetary']
    
    #new_df.drop(columns=spend_cols, inplace=True)
    
    return new_df

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
    new_df = df.copy()
    
    for column in df.columns:
        min_value = df[column].min()
        if min_value < 0:
            # shift the data by adding a constant to make all values non-negative
            shift_value = abs(min_value)
            new_df[column] = np.sqrt(df[column] + shift_value)
            print(f"Column '{column}' was shifted by {shift_value} to handle negative values.")
        else:
            # apply the square root transformation directly
            new_df[column] = np.sqrt(df[column])
    
    return new_df

def isolation_forest(df, columns, contamination=0.01, random_state=42):
    """
    Removes outliers from specified columns in a DataFrame using the Isolation Forest method.

    Parameters:
    df (pandas.DataFrame): The input DataFrame.
    columns (list): The list of columns to check for outliers.
    contamination (float): The proportion of outliers in the data set, should be between 0 and 0.5.
    random_state (int): The random seed for reproducibility.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing only outliers.
    """
    new_df = df.copy()
    initial_row_count = new_df.shape[0]

    clf = IsolationForest(contamination=contamination, random_state=random_state)
    clf.fit(new_df[columns])

    outliers = clf.predict(new_df[columns])
    outliers = pd.DataFrame(outliers, columns=['outlier'], index=new_df.index)

    inliers = new_df[outliers['outlier'] == 1].copy()
    iso_outliers = new_df[outliers['outlier'] == -1].copy()

    final_row_count = inliers.shape[0]
    num_rows_removed = initial_row_count - final_row_count
    percentage_removed = (num_rows_removed / initial_row_count) * 100

    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage of dataset removed: {percentage_removed:.2f}%")

    return inliers, iso_outliers

def remove_fishy_outliers(df):
    """
    Removes outliers from the DataFrame based on the existence of "Fishy" in their name.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: The DataFrame with outliers removed.
    pd.DataFrame: The DataFrame containing only outliers.
    """
    new_df = df.copy()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../data/raw'))

    path = os.path.join(base_dir, 'customer_info.csv')
    customer_info = pd.read_csv(path, index_col='customer_id')
    
    fishy_indexes = customer_info[customer_info['customer_name'].str.contains('Fishy')].index
    
    initial_row_count = new_df.shape[0]
    outliers = pd.DataFrame()

    fishy_outliers = new_df.loc[fishy_indexes]
    outliers = pd.concat([outliers, fishy_outliers])

    new_df = new_df.drop(fishy_indexes)
    
    outliers = outliers.drop_duplicates()
    final_row_count = new_df.shape[0]
    num_rows_removed = initial_row_count - final_row_count
    percentage_removed = (num_rows_removed / initial_row_count) * 100

    print(f"Number of rows removed: {num_rows_removed}")
    print(f"Percentage of dataset removed: {percentage_removed:.2f}%")

    return new_df, outliers