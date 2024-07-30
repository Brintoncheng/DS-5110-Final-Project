import pandas as pd
import os
from constants import NON_COUNTRIES

def import_data(filename):
    data_path = os.path.join('data', filename)
    df = pd.read_csv(data_path)
    return df

def clean_data(df):
    # Remove Code Column
    df = df.drop(['Code'], axis=1)

    # Remove non_countries
    df = df[~df['Country'].isin(NON_COUNTRIES)]

    # Turn all Cancer Death numbers into ints
    df.iloc[:, 2:] = df.iloc[:, 2:].astype(int)

    return df

def initialize_variables(df):
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
    xaxis = 'Year'
    yaxis = 'Liver cancer'

    return cols, countries, years, cancer_types, country, year, cancer, new_cols, default_column, xaxis, yaxis

def filter_data(data, country='ALL', year='ALL', cancer='ALL'):
    df = data
    if country != 'ALL':
        df = df[df["Country"] == country]
    if year != 'ALL':
        df = df[df["Year"] == year]
    if cancer != 'ALL':
        df = df[['Country', 'Year', cancer]]
    return df
