from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sb

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
    df_data = data.copy()
    s_plot = 'hist' #@ might remove this.
    s_country = country
    i_year = year
    s_cancer = cancer
    l_inputTypes = [1, 1, 1] # [cancer, country, year]
    if(cancer == None or cancer == ''):
        s_cancer = 'all'
        l_inputTypes[0] = 2
    if(country == None or country == ''):
        s_country = 'all'
        l_inputTypes[1] = 2
    if(year == None or year == ''):
        i_year = 'all'
        s_plot = 'scat'
        l_inputTypes[2] = 2

    if(l_inputTypes[0] == 1 and l_inputTypes[1] == 1):
        return single_cancer_conutry(df_data, s_cancer, s_country)
    elif(l_inputTypes[1] == 1 and l_inputTypes[2] == 1):
        return single_country_year(df_data, s_country, i_year)
    elif(l_inputTypes[0] == 1 and l_inputTypes[2] == 1):
        return single_cancer_year(df_data, s_cancer, i_year)
    else: # only 1 single-factor, need 3D plot
        pass


def single_cancer_conutry(data:pd.DataFrame, cancer:str=None, country:str=None):
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


def single_country_year(data:pd.DataFrame, country:str=None, year:int=None):
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
    df_fitler = data[data['Country'] == country]
    df_fitler = df_fitler[df_fitler['Year'] == year]
    df_fitler = df_fitler.drop(columns=['Country', 'Code', 'Year'])
    df_fitler = df_fitler.melt()

    g = sb.barplot(x=np.array(df_fitler.variable), y=np.array(df_fitler.value))
    g.tick_params(axis='x', rotation=90)
    g.set(xlabel='Cancer Types', ylabel='Death Number')
    g.set_title(s_title)
    return g


def single_cancer_year(data:pd.DataFrame, cancer:str=None, year:int=None):
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
    df_filter = data[data['Year'] == year]
    df_filter = df_filter[['Country', cancer]]
    l_regions = ['American Samoa', 'Andean Latin America', 'Australasia', 'Bermuda', 'Caribbean', 'Central Asia', 'Central Europe', 
             'Central Latin America', 'Central Sub-Saharan Africa', 'East Asia', 'Eastern Europe', 'Eastern Sub-Saharan Africa', 
             'England', 'Greenland', 'Guam', 'Latin America and Caribbean', 'North Africa and Middle East', 'North America', 
             'Northern Ireland', 'Northern Mariana Islands', 'Oceania', 'Palestine', 'Puerto Rico', 'Scotland', 'South Asia', 
             'Southeast Asia', 'Southern Latin America', 'Southern Sub-Saharan Africa', 'Sub-Saharan Africa', 'Timor', 
             'Tropical Latin America', 'United States Virgin Islands', 'Wales', 'Western Europe', 'Western Sub-Saharan Africa', 
             'World']
    for s_item in l_regions:
        df_filter = df_filter.drop(index=df_filter[df_filter['Country'] == s_item].index)

    plt.figure(figsize=[80, 20])
    g = sb.barplot(x=np.array(df_filter['Country']), y=np.array(df_filter[cancer]))
    g.tick_params(axis='x', rotation=90)
    g.set(yscale='log', xlabel='Country', ylabel=cancer+'death (log)')
    g.set_title(s_title)
    return g


def ThreeD_plot(data:pd.DataFrame, d_singleGiven:dict):
    pass




import os.path as pt
if __name__ == '__main__':
    s_dataPath = pt.join('data', 'Cancer Deaths by Country and Type Dataset.csv')
    df_data = pd.read_csv(s_dataPath)

    #@ Test Case 1
    country = 'Afghanistan'
    year = None
    cancer = 'Liver cancer '
    plot(df_data, country, year, cancer) # x_axis = 'Year'


    #@ Test Case 2
    country = 'Afghanistan'
    year = 1990
    cancer = ''
    plot(df_data, country, year, cancer) # x_axis = 'Cancer'


    #@ Test Case 3
    country = ''
    year = 1990
    cancer = 'Liver cancer '
    # plot(df_data, country, year, cancer) # x_axis = 'Country'


    plt.show()