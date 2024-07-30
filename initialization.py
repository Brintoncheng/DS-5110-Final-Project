import pandas as pd
from typing import List, Tuple
import numpy as np
import os
from constants import NON_COUNTRIES

def import_data(filename: str) -> pd.DataFrame:
    """
    Imports data from a CSV file into a pandas DataFrame.

    Parameters:
    filename (str): The name of the CSV file to import.

    Returns:
    DataFrame: The pandas DataFrame containing the imported data.
    """
    data_path = os.path.join('data', filename)
    df=pd.read_csv(data_path)
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the provided DataFrame by removing the 'Code' column,
    excluding non-country rows, and converting cancer death numbers to integers.

    Parameters:
    df (DataFrame): The pandas DataFrame to clean.

    Returns:
    DataFrame: The cleaned pandas DataFrame.
    """
    # Remove Code Column
    df = df.drop(['Code'], axis=1)
    
    # Remove non_countries
    df = df[~df['Country'].isin(NON_COUNTRIES)]
    
    # Turn all Cancer Death numbers into ints
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(int)

    return df

def initialize_variables(df: pd.DataFrame) -> Tuple[List[str], List[str], List[int], List[str], str, str, str, List[str], str]:
    """
    Initializes variables from the DataFrame for use in analysis.

    Parameters:
    df (DataFrame): The pandas DataFrame containing the data.

    Returns:
    Tuple: A tuple containing initialized variables:
        - cols (List[str]): List of column names.
        - countries (List[str]): List of unique country names.
        - years (List[int]): List of unique years.
        - cancer_types (List[str]): List of cancer type column names.
        - country (str): Default country value.
        - year (str): Default year value.
        - cancer (str): Default cancer type value.
        - new_cols (List[str]): Copy of the column names list.
        - default_column (str): Default column name for analysis.
    """
    # Define lists of feature labels
    cols = df.columns.tolist()
    countries = df["Country"].unique().tolist()
    years = df["Year"].unique().tolist()
    cancer_types = cols[2:] # First 2 columns are Country and Year respectively

    # Set default values
    country = "ALL"
    year = "ALL"
    cancer = "ALL"
    new_cols = cols.copy()

    default_column = "Country"

    return cols, countries, years, cancer_types, country, year, cancer, new_cols, default_column

def filter_data(data: pd.DataFrame, country: str = 'ALL', year: str = 'ALL', cancer: str = 'ALL'):
    """
    Filters the data based on the specified country, year, and cancer type.

    Parameters:
    data (DataFrame): The pandas DataFrame containing the data.
    country (str): The country to filter by. Defaults to 'ALL' for no filtering.
    year (str): The year to filter by. Defaults to 'ALL' for no filtering.
    cancer (str): The cancer type to filter by. Defaults to 'ALL' for no filtering.

    Returns:
    DataFrame: The filtered pandas DataFrame.
    """
    df = data
    if country != 'ALL':
        df = df[df["Country"] == country]
    if year != 'ALL':
        df = df[df["Year"] == year]
    if cancer != 'ALL':
        df = df[['Country', 'Year', cancer]]
    return df
