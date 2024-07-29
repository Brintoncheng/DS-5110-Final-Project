import pandas as pd
import numpy as np


# Import data
df = pd.read_csv(r'data/Cancer Deaths by Country and Type Dataset.csv')

# Remove NaNs
df['Code'] = df.groupby('Country')['Code'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
df['Code'] = df['Code'].apply(lambda x: x.strip() if pd.notna(x) else x)
df.loc[df['Country'] == 'North America', 'Code'] = 'NA'

# Define lists of feature labels
cols = df.columns.to_numpy()
cols = np.array([col.strip() for col in cols])
cancer_types = cols[3:]
countries = df["Country"].unique()
codes = df["Code"].unique()
years = df["Year"].unique()

print(years)

def filter_data(data, country=None, year=None, cancer=None):
    df = data
    if country is not None:
        df = df[df["Country"] == country]
    if year is not None:
        df = df[df["Year"] == year]
    if cancer is not None:
        df = df[['Country', 'Year', cancer]]
    return df

def plot(data, country=None, year=None, cancer=None, plot_type = "scatter"):
    # Plot scatterplot if country and cancer are named
    
    # Plot histogram if year and cancer are named
    
    return #plot_pic
