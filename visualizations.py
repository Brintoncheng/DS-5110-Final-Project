import matplotlib
matplotlib.use('Agg')
from flask import send_file
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pandas as pd
import numpy as np
import seaborn as sns
import os.path as pt
import os

from initialization import import_data, clean_data
from constants import FILENAME, D_REGIONS

'''
Plot scenarios:
    1. single-country, single-cancer, cancer-number (Y) vs year (X).            Scatter
    2. single-country, single-year,   cancer-number (Y) vs cancers (X).         Bar
    3. single-year,    single-cancer, cancer-number (Y) vs countries (X).       Bar

    4. single-country, cancers (Y) vs year (X) vs cancer-number(Z).             3D-Bar
    5. single-cancer,  country (Y) vs year (X) vs cancer-number(Z).             3D-Bar
    6. single-year,    country (Y) vs cancers (X) vs cancer-number(Z).          3D-Bar
'''
#! ToDo:
#!      1. Scaling on 2D-y and 3D-z, apply log only necessary.
#!      2. Simplify/combine the 3 2D-plot-function into one.
#!      3. Write a function using linear regression plot.
#!      4. Perhaps limit the len/size of Cancer/Country/Year, for better visualization.
#!         That would require limiting user's selection in FE UI, and perhaps an additional function to pre-process data.


def choose_plot(data, country:str=None, year:int=None, cancer:str=None, plot_type=''):
    # the defaults should be "all", like all-country, all-years, all-cancer-types, etc.
    s_plot = 'hist' #@ might remove this.
    if (country is not None) and (cancer is not None):
        return country_cancer(data, cancer, country)

    if (country is not None) and (year is not None):
        return country_year(data, country, year)

    if (year is not None) and (cancer is not None):
        return year_cancer(data, cancer, year)

    ## 3D part.
    if(cancer):
        return ThreeD_plot(data, 'Cancer', cancer)
    if(country):
        return ThreeD_plot(data, 'Country', country)
    if(year):
        return ThreeD_plot(data, 'Year', year)

    return None

def country_cancer(data:pd.DataFrame, cancer:str=None, country:str=None):
    '''
    @Purpose: Create a Scatter plot showing number of Death of a given Cancer in a given Country with respect to time.\n
    @Param:
        data: pandas DataFrame, raw data.
        cancer: str, cancer name/type.
        country: str, country name.
    @Return:
        A pyplot object.
    '''
    s_title = f'Death of {cancer}over years in {country}'
    df_filter = data[data['Country'] == country]

    fig, ax = plt.subplots(figsize=(12,6))
    sns.scatterplot(data=df_filter, x='Year', y=cancer, ax=ax)
    ax.set(ylabel=cancer + 'Death')
    ax.set_title(s_title)
    return fig


def country_year(data:pd.DataFrame, country:str=None, year:int=None):
    '''
    @Purpose: Create a Bar plot showing number of Death with respect to each Cancer in a given Country at a given Year.\n
    @Param:
        data: pandas DataFrame, raw data.
        country: str, country name.
        year: int, year in AD.
    @Return:
        A pyplot object.
    '''
    s_title = f'Cancer Deaths in {year} in {country}'
    df_filter = data[(data['Country'] == country) & (data['Year'] == year)]
    df_filter = df_filter.drop(columns=['Country', 'Year'])
    df_filter = df_filter.melt()

    x_data = np.array(df_filter['variable'].astype(str))
    y_data = np.array(df_filter['value'])

    fig, ax = plt.subplots(figsize=(12,6))
    sns.barplot(x=x_data, y=y_data, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    # ax.tick_params(axis='x', rotation=90)
    plt.tight_layout()
    plt.subplots_adjust(top = 0.9, bottom=0.4)
    ax.set(xlabel='Cancer Types', ylabel='Deaths')
    ax.set_title(s_title)
    return fig


def year_cancer(data:pd.DataFrame, cancer:str=None, year:int=None, sort=False, top=False, bottom=False):
    '''
    @Purpose: Create a Bar plot showing number of Death of a given Cancer at a given Year with respect to Countries.
    @Param:
        data: pandas DataFrame, raw data.
        cancer: str, cancer name/type.
        year: int, year in AD.
    @Return:
        A pyplot Figure object.
    '''
    s_title = f'Death of {cancer} in {year} in every country'
    df_filter = data.loc[data['Year'] == year, ['Country', cancer]]
    
    # Filter for top and bottom
    
    # If sort:

    if df_filter.empty:
        raise ValueError("No data available for the specified year and cancer type.")

    fig, ax = plt.subplots(figsize=(30, 6))  # Adjust figure size based on number of countries
        # figsize=[data['Country'].unique().size, 20]
    sns.barplot(data=df_filter, x='Country', y=cancer, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    # ax.tick_params(axis='x', rotation=90)
    ax.set(yscale='log', xlabel='Country', ylabel=f'{cancer} deaths (log scale)') # yscale='linear' or 'log'
    ax.set_title(s_title)

    plt.tight_layout()
    return fig

def ThreeD_plot(data:pd.DataFrame, key:str, value:str|int):
    '''
    @Purpose: Create a 3D bar plot based on input.\n
    @Param:
        data: pandas DataFrame, raw data.
        key: str, one of ['Cancer', 'Country', 'Year'], to indicate purpose of "value".
        value: str|int, value of the "key".
    @Return:
        A pyplot object.
    '''
    ## 3D plot shows the result of 2-all and 1-single from Cancer, Country, Year.
    ## Input Filter.
    l_xy = ['Cancer', 'Country', 'Year']
    if(key not in l_xy):
        raise Exception('Incorrect \"key\" input')
    if(data.empty):
        raise Exception('Empty \"data\" input')

    ## Decide x, y, z.
    l_xy.remove(key)
    s_xlabel = l_xy[0]
    s_ylabel = l_xy[1]
    s_zlabel = 'Cancer Death'

    ## x, y, z are starting coord, dx, dy, dz are change of x,y,z values.
    ## So all 'z' are 0 (starts from z=0), dx=dy=1 for change of (x,y) is constant, we only want to see dz, aka change of z.
    ## x,y are the starting point of each dz, so sizes of x,y,dz must be equal.

    ## Pre-process data + set plot title + set dz for bar3d(). These values vary by scenarios.
    match key:
        case 'Cancer':
            data = data[['Country', 'Year', value]]
            dz = data[value].to_numpy().flatten() #@ fixed cancer-type, case-5 (all country * all year)
            s_title = f'{value}Death in {s_xlabel} over {s_ylabel}'
        case 'Country':
            data = data[data['Country'] == value]
            dz = data.drop(columns=['Country', 'Year']).values.flatten() #@ fixed country, case-4
            s_title = f'Each {s_xlabel} Death over {s_ylabel} in {value}'
        case 'Year':
            data = data[data['Year'] == value]
            dz = data.drop(columns=['Country', 'Year']).values.flatten() #@ fixed year, case-6 --> all the cancer (x, 27) * all the countries (y, 184)
            s_title = f'{s_xlabel} Death in {s_ylabel} in {value}'

    ## set x, y for bar3d(). These values vary by scenarios.
    if('Cancer' in l_xy):
        i_num1 = data.columns[2:].unique().size
        s_indexing = 'xy'
        l_xlabels = data.columns[2:].unique()
    else:
        i_num1 = data[s_xlabel].unique().size
        s_indexing = 'ij'
        l_xlabels = data[s_xlabel].unique()
    i_num2 = data[s_ylabel].unique().size
    x, y = np.meshgrid(np.arange(i_num1), np.arange(i_num2), indexing=s_indexing) #! indexing = xy or ij, it depends on dz
    x = x.flatten()
    y = y.flatten()

    ## set dx, dy, z for bar3d(). These values remain unchanged across scenarios.
    dx = np.full(len(x), 0.5)
    dy = np.full(len(y), 0.5)
    z = np.zeros(len(x))

    ## Plot 3D.
    fig = plt.figure(figsize=(12, 8)) #! might need change
    ax = fig.add_subplot(projection='3d')
    ax.xaxis.set_ticks([i+0.25 for i in range(i_num1)])
    ax.xaxis.set_ticklabels(l_xlabels)
    ax.yaxis.set_ticks([i+0.25 for i in range(i_num2)])
    ax.yaxis.set_ticklabels(data[s_ylabel].unique())
    ax.set_xlabel(f'x-{s_xlabel}')
    ax.set_ylabel(f'y-{s_ylabel}')
    ax.set_zlabel(f'z-{s_zlabel}')
    ax.set_zlim(0, dz.max())
    ax.set_title(s_title)
    ax.bar3d(x, y, z, dx, dy, dz)
    return fig


def regionAnalysis(data:pd.DataFrame, cancer:str=None, s_region:str='asia', year:int=None):
    #! data needs to be un-filtered data.
    if(s_region in list(D_REGIONS.keys())): # input s_region is "Region" not "Country"
        l_region = D_REGIONS[s_region.lower()]
    else: # input s_region is just a "Country"
        data = clean_data(data)
        return choose_plot(data, country, year, cancer)

    data = data[(data['Country'].isin(l_region))]

    #! will need to add a "region" flag for changing title of plots.
    if((cancer is not None) and (year is not None)):
        return year_cancer(data, cancer, year)
    if(cancer):
        return ThreeD_plot(data, 'Cancer', cancer)
    if(year):
        return ThreeD_plot(data, 'Year', year)



def predictFunc(data:pd.DataFrame):
    #! input "data" should be reduced from complete raw data. 80% for training and 20% for testing.
    pass

def dataClean():
    ''' Just the process of data import and clean from initialization.py '''
    # Import data
    df = import_data(FILENAME)

    # Remove NaNs
    # df['Code'] = df.groupby('Country')['Code'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
    # df['Code'] = df['Code'].apply(lambda x: x.strip() if pd.notna(x) else x)
    # df.loc[df['Country'] == 'North America', 'Code'] = 'NA'
    # Remove Code Column
    # df = df.drop(['Code'], axis=1)

    # Clean Data
    df = clean_data(df)

    return df

def save_plot_to_png(plot_object, filename):
    if plot_object is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No plot available', horizontalalignment='center', verticalalignment='center')
        plot_object = fig

    plot_filepath = os.path.join('static', 'plots', filename)
    plot_object.savefig(plot_filepath)
    plot_png = f"{filename}.png"
    return plot_png


if __name__ == '__main__':
    data = dataClean()
    # print(data)

    #@ Test Case 1
    country = 'Afghanistan'
    year = None
    cancer = 'Liver cancer '
    # choose_plot(data, country, year, cancer)

    #@ Test Case 2
    country = 'Afghanistan'
    year = 1990
    cancer = None
    # choose_plot(data, country, year, cancer)

    #@ Test Case 3
    country = None
    year = 1990
    cancer = 'Liver cancer '
    # choose_plot(data, country, year, cancer)

    #@ Test Case 4
    country = 'Afghanistan'
    year = None
    cancer = None
    # ret = choose_plot(data, country, year, cancer)

    #@ Test Case 5
    country = None
    year = None
    cancer = 'Liver cancer '
    # ret = choose_plot(data, country, year, cancer)

    #@ Test Case 6
    country = None
    year = 1990
    cancer = None
    # ret = choose_plot(data, country, year, cancer)

    #@ Test Case 7
    data = import_data(FILENAME)
    # Remove Code Column
    data = data.drop(['Code'], axis=1)
    # Turn all Cancer Death numbers into ints
    data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)
    region = 'africa'
    year = None
    cancer = 'Liver cancer '
    regionAnalysis(data, cancer, region, year)

    plt.show()