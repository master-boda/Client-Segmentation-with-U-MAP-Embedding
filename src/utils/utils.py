import pandas as pd
from datetime import datetime
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import geohash2

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
    
def var_plotter(df):
    """
    Plots histograms for numerical columns in the DataFrame.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the data.
    """
    # remove rows with missing values
    df = df.dropna()
    
    num_cols = df.select_dtypes(include=['int64', 'float64'])
    
    # calculate the number of rows and columns for subplots
    num_plots = len(num_cols.columns)
    num_cols_in_plot = 3
    num_rows = int(np.ceil(num_plots / num_cols_in_plot))

    plt.figure(figsize=(5 * num_cols_in_plot, 5 * num_rows))

    for i, col in enumerate(num_cols.columns):
        plt.subplot(num_rows, num_cols_in_plot, i + 1)
        sns.histplot(num_cols[col], kde=True, stat="density", linewidth=0, color='blue')
        plt.title(col)
        plt.xlabel(col)
        plt.ylabel('Density')

    plt.tight_layout()
    plt.show()

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

def geohash_loc(df, lat_column, lon_column, precision=5):
    """
    Generate geohashes for a list of coordinates.

    Parameters:
    df (pandas.DataFrame): The DataFrame containing the coordinates.
    lat_column (str): The name of the column containing the latitude values.
    lon_column (str): The name of the column containing the longitude values.
    precision (int, optional): The precision of the generated geohashes. Defaults to 5. (5≤ 4.89km*4.89km)

    Returns:
    pandas.DataFrame: The DataFrame with an additional 'geo' column containing the geohash values.
    """
    
    df['geohash'] = df.apply(lambda x: geohash2.encode(x[lat_column], x[lon_column], precision), axis=1)

    df['geo'] = df['geohash'].apply(lambda x: int.from_bytes(x.encode(), 'big'))

    df.drop(['geohash'], axis=1, inplace=True)

    return df