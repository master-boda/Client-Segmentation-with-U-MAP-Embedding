import pandas as pd
from datetime import datetime
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import KNNImputer, SimpleImputer

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

    return df


def assign_city(lat, lon):
    """
    This function assigns a city name based on given latitude and longitude 
    coordinates. The cities and their corresponding boundaries are predefined 
    within the function:
    
    - Lisbon: Latitude between 38.6 and 38.85, Longitude between -9.25 and -9.05
    - Peniche: Latitude between 39.3 and 39.4, Longitude between -9.5 and -9.3
    - Ericeira: Latitude between 38.9 and 39.0, Longitude between -9.5 and -9.3
    - Other: Any other coordinates outside the defined boundaries
    
    Parameters:
    lat (float): Latitude of the location.
    lon (float): Longitude of the location.

    Returns:
    str: The name of the city corresponding to the given coordinates, 
         or 'Other' if the coordinates do not match any predefined city boundaries.
    """
    if 38.6 <= lat <= 38.85 and -9.25 <= lon <= -9.05:
        return 'Lisbon'
    elif 39.3 <= lat <= 39.4 and -9.5 <= lon <= -9.3:
        return 'Peniche'
    elif 38.9 <= lat <= 39.0 and -9.5 <= lon <= -9.3:
        return 'Ericeira'
    else:
        return 'Other'
    

def scale_and_impute(df, numerical_cols, ordinal_cols, scaler):
    """
    Scales the numerical and ordinal columns in the DataFrame and applies KNN imputation for missing values.

    Parameters:
    df (pd.DataFrame): The input DataFrame containing the data.
    numerical_cols (list): A list of column names corresponding to numerical features.
    ordinal_cols (list): A list of column names corresponding to ordinal features.
    scaler: An instantiated scaler from sklearn.preprocessing (e.g., StandardScaler, MinMaxScaler, RobustScaler).

    Returns:
    pd.DataFrame: The DataFrame with scaled and imputed numerical and ordinal features.
    """
    # define the pipeline for numerical features
    numerical_pipeline = Pipeline(steps=[
        ('scaler', scaler),          # Scale the data
        ('imputer', KNNImputer(n_neighbors=5))   # Impute missing values
    ])

    # define the pipeline for ordinal features
    ordinal_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent'))  # Impute missing values with the most frequent value
    ])

    # combine pipelines using ColumnTransformer
    preprocessor = ColumnTransformer(transformers=[
        ('num', numerical_pipeline, numerical_cols),
        ('ord', ordinal_pipeline, ordinal_cols)
    ], remainder='passthrough')  # Pass through all other columns

    # fit and transform the data using the preprocessor
    df_processed = preprocessor.fit_transform(df)

    # combine the processed numerical data with the other data
    processed_cols = numerical_cols + ordinal_cols + [col for col in df.columns if col not in numerical_cols + ordinal_cols]
    df_processed = pd.DataFrame(df_processed, columns=processed_cols, index=df.index)

    return df_processed


def sqrt_transform(dataframe, columns):
    """
    Applies square root transformation to the specified columns in the dataframe.
    Shifts the data if there are negative values in a column to make all values in that column non-negative.
    
    Parameters:
    dataframe (pd.DataFrame): The input dataframe containing the data.
    columns (list of str): The list of column names to apply the transformation to.
    
    Returns:
    pd.DataFrame: A dataframe with the square root transformation applied to the specified columns.
    """
    transformed_data = dataframe.copy()
    
    for column in columns:
        min_value = dataframe[column].min()
        if min_value < 0:
            # shift the data by adding a constant to make all values non-negative
            shift_value = abs(min_value)
            transformed_data[column] = np.sqrt(dataframe[column] + shift_value)
            print(f"Column '{column}' was shifted by {shift_value} to handle negative values.")
        else:
            # apply the square root transformation directly
            transformed_data[column] = np.sqrt(dataframe[column])
    
    return transformed_data