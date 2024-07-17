from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb
import os.path as pt

'''
Plot scenarios:
    1. single-country, single-cancer, cancer-number (Y) vs year (X).            Scatter
    2. single-country, single-year,   cancer-number (Y) vs cancers (X).         Bar
    3. single-year,    single-cancer, cancer-number (Y) vs countries (X).       Bar

    4. single-country, cancers (Y) vs year (X) vs cancer-number(Z).             3D-Bar
    5. single-cancer,  country (Y) vs year (X) vs cancer-number(Z).             3D-Bar
    6. single-year,    country (Y) vs cancers (X) vs cancer-number(Z).          3D-Bar

    Only Country & Year can be filtered !!!
'''


def plot(data, country:str=None, year:int=None, cancer:str=None, plot_type=''):
    # the defaults should be "all", like all-country, all-years, all-cancer-types, etc.
    s_plot = 'hist' #@ might remove this.

    if (country is not None) and (cancer is not None):
        return country_cancer(data, cancer, country)

    if (country is not None) and (year is not None):
        return country_year(data, country, year)

    if (year is not None) and (cancer is not None):
        return year_cancer(data, cancer, year)

    pass


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

    g = sb.scatterplot(data=df_filter, x='Year', y=cancer)
    g.set(ylabel=cancer + 'Death')
    g.set_title(s_title)
    return g


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
    s_title = f'Death of each Cancer on {year} in {country}'
    df_filter = data[(data['Country'] == country) & (data['Year'] == year)]
    df_filter = df_filter.drop(columns=['Country', 'Year'])
    df_filter = df_filter.melt()
    
    x_data = np.array(df_filter['variable'].astype(str))
    y_data = np.array(df_filter['value'])

    g = sb.barplot(x=x_data, y=y_data)
    g.tick_params(axis='x', rotation=90)
    g.set(xlabel='Cancer Types', ylabel='Deaths')
    g.set_title(s_title)
    return g


def year_cancer(data:pd.DataFrame, cancer:str=None, year:int=None):
    '''
    @Purpose: Create a Bar plot showing number of Death of a given Cancer at a given Year with respect to Countries.\n
    @Param:
        data: pandas DataFrame, raw data.
        cancer: str, cancer name/type.
        year: int, year in AD.
    @Return:
        A pyplot object.
    '''
    s_title = f'Death of {cancer}on {year} in every country'
    df_filter = data.loc[data['Year'] == year, ['Country', cancer]]

    plt.figure(figsize=[80, 20])
    g = sb.barplot(data=df_filter, x='Country', y=cancer)
    g.tick_params(axis='x', rotation=90)
    g.set(yscale='log', xlabel='Country', ylabel=cancer+'death (log)')
    g.set_title(s_title)
    return g


def ThreeD_plot(data:pd.DataFrame, d_singleGiven:dict):
    pass



if __name__ == '__main__':
    s_dataPath = pt.join('data', 'Cancer Deaths by Country and Type Dataset.csv')
    data = pd.read_csv(s_dataPath)

    #@ Test Case 1
    country = 'Afghanistan'
    year = None
    cancer = 'Liver cancer '
    plot(data, country, year, cancer) # x_axis = 'Year'
    plt.show()

    #@ Test Case 2
    country = 'Afghanistan'
    year = 1990
    cancer = None
    #plot(data, country, year, cancer) # x_axis = 'Cancer'
    #plt.show()

    #@ Test Case 3
    country = None
    year = 1990
    cancer = 'Liver cancer '
    # plot(data, country, year, cancer) # x_axis = 'Country'
    #plt.show()