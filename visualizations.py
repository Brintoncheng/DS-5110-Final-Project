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
from constants import FILENAME, D_REGIONS, TOP_5_CANCER, TOP_10_CANCER

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


def choose_plot(data:pd.DataFrame, country:str=None, year:int=None, cancer:str=None):
    # the defaults should be "all", like all-country, all-years, all-cancer-types, etc.
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
    @Purpose:
        Plotting a Regression plot showing number of Death of a given Cancer in a given Country with respect to time.
    @Param:
        data: pandas DataFrame, raw data.
        cancer: str, cancer name/type.
        country: str, country name.
    @Return:
        A pyplot Figure object.
    '''
    s_title = f'Death of {cancer}over years in {country}'
    df_filter = data[data['Country'] == country]

    fig, ax = plt.subplots(figsize=(12,6))
    # sns.scatterplot(data=df_filter, x='Year', y=cancer, ax=ax)
    sns.regplot(data=df_filter, x='Year', y=cancer, ax=ax)
    ax.set(ylabel=cancer + 'Death')
    ax.set_title(s_title)
    return fig


def country_year(data:pd.DataFrame, country:str=None, year:int=None):
    '''
    @Purpose:
        Plotting a Bar plot showing number of Death with respect to each Cancer in a given Country at a given Year.
    @Param:
        data: pandas DataFrame, raw data.
        country: str, country name.
        year: int, year in AD.
    @Return:
        A pyplot Figure object.
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
    plt.tight_layout()
    plt.subplots_adjust(top = 0.9, bottom=0.4)
    ax.set(xlabel='Cancer Types', ylabel='Deaths')
    ax.set_title(s_title)
    return fig


def year_cancer(data:pd.DataFrame, cancer:str=None, year:int=None):
    '''
    @Purpose:
        Plotting a Bar plot showing number of Death of a given Cancer at a given Year with respect to Countries.
    @Param:
        data: pandas DataFrame, raw data.
        cancer: str, cancer name/type.
        year: int, year in AD.
    @Return:
        A pyplot Figure object.
    '''
    s_title = f'Death of {cancer} in {year} in every country'
    df_filter = data.loc[data['Year'] == year, ['Country', cancer]]
    
    if df_filter.empty:
        raise ValueError("No data available for the specified year and cancer type.")

    fig, ax = plt.subplots(figsize=(30, 6))  # Adjust figure size based on number of countries
        # figsize=[data['Country'].unique().size, 20]
    sns.barplot(data=df_filter, x='Country', y=cancer, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    ax.set(yscale='log', xlabel='Country', ylabel=f'{cancer} deaths (log scale)') # yscale='linear' or 'log'
    ax.set_title(s_title)

    plt.tight_layout()
    return fig


def ThreeD_plot(data:pd.DataFrame, key:str, value:str|int):
    '''
    @Purpose:
        Plotting a 3D bar graph based on input.
    @Param:
        data: pandas DataFrame, raw data.
        key: str, one of ['Cancer', 'Country', 'Year'], to indicate purpose of "value".
        value: str|int, value of the "key".
    @Return:
        A pyplot Figure object.
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


def worldTopCancerRegression(top:int=5):
    '''
    @Purpose:
        Plotting top-N cancer deaths over years in the whole world.
    @Param:
        top: number of top cancere by death number.
    @Return:
        A pyplot Figure object.
    '''
    data = reduceToTopCancer(top=top)
    data = data[data['Country'] == 'World'] #;print(f'data = {data}')

    ## Plotting.
    df_melt = data.drop(columns=['Country'])
    df_melt = pd.melt(df_melt, ['Year'], var_name='cancer') #;print(df_melt, ['Year'])
    #* multi lineplot way
    # fig, ax = plt.subplots(1)
    # sns.lineplot(data=df_melt, x='Year', y='value', hue='cancer')
    # ax.set(yscale='linear', xlabel='Year', ylabel='Cancer Death', title=f'Death of top {top} cancers over years in World')
    # ax.grid(visible=True)
    #* multi regression way
    lm = sns.lmplot(data=df_melt, x='Year', y='value', hue='cancer', aspect=2) #@ seaborn doesn't provide slope info, according to seaborn lead dev.
    lm.set(yscale='linear', xlabel='Year', ylabel='Cancer Death', title=f'Death of top {top} cancers over years in World')
    lm.ax.grid(visible=True)

    return lm

# Lung, bladder, breast, colorectal/colon, prostate cancers are likely due to older age.
def ageVsCancer():
    '''
    Try to see if those cancers are correlated to longer life expectency. Couldn't find patterns.
    '''
    cancers = ["Tracheal, bronchus, and lung cancer ", "Bladder cancer ", "Breast cancer ", "Colon and rectum cancer ", "Prostate cancer "]
    data = import_data(FILENAME)
    # Remove Code Column
    data = data.drop(['Code'], axis=1)
    # Turn all Cancer Death numbers into ints
    data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)

    # Only keep World rows.
    data = data[data['Country'] == 'World']
    # Only keep cancers rows.
    data = data[cancers+['Country', 'Year']] #;print(data)

    # want to check cancers vs years, Country=World.
    ThreeD_plot(data, 'Country', 'World')


def regionOrCountry(data:pd.DataFrame, cancer:str=None, regionCountry:str='asia', year:int=None):
    '''
    @Purpose:
        Identify "Country" or "Region". Then pass on to functions accordingly.
    @Param:
        data: pandas DataFrame, unfiltered raw data.
        cancer: str, cancer name/type.
        regionCountry: str, country or region name.
        year: int, year in AD.
    @Return:
        A pyplot Figure object.
    '''
    #! data needs to be un-filtered data.
    if(regionCountry in list(D_REGIONS.keys())): # input regionCountry is "Region" not "Country"
        l_region = D_REGIONS[regionCountry.lower()]
    else: # input regionCountry is just a "Country"
        data = clean_data(data)
        return choose_plot(data, country, year, cancer)

    # Process data for analysis in regions.
    if('Code' in data.columns):
        data = data.drop(['Code'], axis=1)
    data = data[(data['Country'].isin(l_region))]

    #! will need to add a "region" flag for changing title of plots.
    if((cancer is not None) and (year is not None)):
        return year_cancer(data, cancer, year)
    if(cancer):
        return ThreeD_plot(data, 'Cancer', cancer)
    if(year):
        return ThreeD_plot(data, 'Year', year)

    return None


def topCancerAnalysis(top:int=5) -> list[str]:
    '''
    @Purpose:
        Data process, find top X cancers by death number.
    @Param:
        top: number of top cancere by death number.
    @Return:
        list of cancer names.
    '''
    # Note "Other cancers" ranks 6, we want to exclude it if it's covered.
    if top >= 6:
        top += 1

    data = import_data(FILENAME)
    # Remove Code Column
    data = data.drop(['Code'], axis=1)
    # Turn all Cancer Death numbers into ints
    data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)

    data = data[data['Country'] == 'World']
    # ThreeD_plot(data, 'Country', 'World')

    for eachYear in data['Year'].unique():
        df_temp = data[data['Year'] == eachYear]
        df_temp = df_temp.drop(columns=['Country', 'Year']).melt().sort_values(by='value').tail(top)
        l_topX = list(df_temp.variable.to_list())
        l_topX.reverse()
        # print(f'\nIn {eachYear}, top {top} cancers are -->\n{l_topX}')

    if 'Other cancers ' in l_topX:
        l_topX.remove('Other cancers ')

    return l_topX


def reduceCountryToContinent(data:pd.DataFrame=None) -> pd.DataFrame:
    '''
    @Purpose:
        Data process, collapse regions into continents in the "Country"-column, and corresonding "sub-sum" of each Cancer-death from the regions.
        E.g. "Liver cancer" of "Country"=asia row/index is the sum of "Liver cancer" of ["Central Asia", "East Asia","South Asia", "Southeast Asia"], so on for the rest of cancers.
    @Param:
        data: pandas DataFrame, raw data, can be None.
    @Return:
        A DataFrame of processed data.
    '''
    if(data is None or data.empty):
        data = import_data(FILENAME)
    if('Code' in data.columns):
        # Remove Code Column
        data = data.drop(['Code'], axis=1)
    try:
        # Turn all Cancer Death numbers into ints
        data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)
    except:
        pass

    df_continent = pd.DataFrame(columns=data.columns)

    for key, values in D_REGIONS.items():
        df_temp = data[(data['Country'].isin(values))]
        cancerColumns = df_temp.columns[2:]
        df_aggreg = df_temp.groupby('Year')[cancerColumns].sum().reset_index()
        df_aggreg['Country'] = key
        df_continent = pd.concat([df_continent, df_aggreg])

    return df_continent


def reduceToTopCancer(data:pd.DataFrame=None, top:int=5) -> pd.DataFrame:
    '''
    @Purpose:
        Data process, reduce the size of dataset, by only keeping "Country", "Year" and top-5 cancer columns.
    @Param:
        data: pandas DataFrame, raw data, can be None.
        top: number of top cancere by death number.
    @Return:
        A DataFrame of reduced data (2+top columns total).
    '''
    topCancers = ['Country', 'Year']
    # topCancers += TOP_5_CANCER
    topCancers += topCancerAnalysis(top)
    # print(topCancers)
    if(data is None or data.empty):
        data = import_data(FILENAME)
    if('Code' in data.columns):
        # Remove Code Column
        data = data.drop(['Code'], axis=1)
    try:
        # Turn all Cancer Death numbers into ints
        data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)
    except:
        pass

    data = data[topCancers]
    # print(data)
    return data


def countryData():
    '''
    @Purpose:
        Data process, remove "Code" column, and "Country"=regions rows, also turn numbers to int.
        Same process of data import and cleaning from initialization.py
    @Return:
        A DataFrame of cleaned data.
    '''
    # Import data
    df = import_data(FILENAME)
    # Clean Data
    df = clean_data(df)
    return df


def save_plot_to_png(plot_object, filename) -> str:
    '''
    @Purpose:
        Save a Figure object in png format.
    @Param:
        plot_object: A graph in form of matplotlib Figure object.
    @Return:
        str, name of the png file.
    '''
    if plot_object is None:
        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, 'No plot available', horizontalalignment='center', verticalalignment='center')
        plot_object = fig

    plot_filepath = os.path.join('static', 'plots', filename)
    plot_object.savefig(plot_filepath)
    plot_png = f"{filename}.png"
    return plot_png

def lung_region_year():
    df_lungRegion = reduceCountryToContinent()
    df_lungRegion = df_lungRegion[['Country', 'Year', 'Tracheal, bronchus, and lung cancer ']]
    return ThreeD_plot(df_lungRegion, 'Cancer', 'Tracheal, bronchus, and lung cancer ')

def liver_region_year():
    df_liver = reduceToTopCancer(top=5)
    df_liver = reduceCountryToContinent(data=df_liver)
    return ThreeD_plot(data=df_liver, key='Cancer', value='Liver cancer ')

def liver_asia_year():
    df_asia = reduceToTopCancer(top=5)
    return regionOrCountry(data=df_asia, cancer='Liver cancer ', regionCountry='asia', year=None)

def prostate_africa_year():
    df_africa = import_data(FILENAME)
    # Remove Code Column
    df_africa = df_africa.drop(['Code'], axis=1)
    # Turn all Cancer Death numbers into ints
    df_africa.iloc[:, 2:] = df_africa.iloc[:, 2:].astype(int)
    df_africa = df_africa[['Country', 'Year', 'Prostate cancer ']]
    return regionOrCountry(data=df_africa, cancer='Prostate cancer ', regionCountry='africa', year=None)

if __name__ == '__main__':
    data = countryData()
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
    # Get unfiltered raw data
    data = import_data(FILENAME)
    # Remove Code Column
    data = data.drop(['Code'], axis=1)
    # Turn all Cancer Death numbers into ints
    data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)

    region = 'africa'
    year = 2015
    cancer = None
    # Liver cancer ,Kidney cancer ,Larynx cancer ,Breast cancer ,Thyroid cancer ,Stomach cancer ,Bladder cancer ,Uterine cancer ,Ovarian cancer ,
    # Cervical cancer ,Prostate cancer ,Pancreatic cancer ,Esophageal cancer ,Testicular cancer ,Nasopharynx cancer ,Other pharynx cancer ,
    # Colon and rectum cancer ,Non-melanoma skin cancer ,Lip and oral cavity cancer ,Brain and nervous system cancer ,"Tracheal, bronchus, and lung cancer ",
    # Gallbladder and biliary tract cancer ,Malignant skin melanoma ,Leukemia ,Hodgkin lymphoma ,Multiple myeloma ,Other cancers
    # regionOrCountry(data, cancer, region, year)

    #@ Test Case 8
    # data_region = reduceCountryToContinent(data)
    country = None
    year = 1990 # 2000 2010
    cancer = None #'Liver cancer '
    # choose_plot(data_region, country, year, cancer)

    #@ Test Case 9: only 5 Cancers, and using Regions not countries
    # topCancerAnalysis()
    # df_5cancer = reduceToTopCancer(top=8)
    # df_5cancerRegion = reduceCountryToContinent(df_5cancer)
    # country = None # all regions
    # year = None
    # cancers = ['Tracheal, bronchus, and lung cancer ', 'Stomach cancer ', 'Colon and rectum cancer ', 'Liver cancer ', 'Breast cancer ']
    # ThreeD_plot(df_5cancerRegion, 'Year', 2016, True)

    #@
    # worldTopCancerRegression(27)
    # regression3D()
    # ageVsCancer()

    #@ analyze Lung cancer, regions vs years
    # df_lungRegion = reduceCountryToContinent()
    # df_lungRegion = df_lungRegion[['Country', 'Year', 'Tracheal, bronchus, and lung cancer ']]
    # ThreeD_plot(df_lungRegion, 'Cancer', 'Tracheal, bronchus, and lung cancer ')

    #@ Health care system study
    # # Get unfiltered raw data
    # data = import_data(FILENAME)
    # # Remove Code Column
    # data = data.drop(['Code'], axis=1)
    # # Turn all Cancer Death numbers into ints
    # data.iloc[:, 2:] = data.iloc[:, 2:].astype(int)

    # candidates = ['Sweden', 'Switzerland', 'Germany', 'Netherlands', 'Norway', 
    #               'South Korea', 'United Kingdom', 'Australia', 'Japan', 'Canada', 
    #               'France', 'Taiwan', 'Austria', 'Spain'] # checked, all in data['Country']
    # # countries = list(data['Country'].unique()) ;print(f'countries = {countries}')
    # data = data[data['Country'].isin(candidates)]
    # data = reduceToTopCancer(data=data, top=5)
    # ThreeD_plot(data, 'Year', 2016)

    #@ Liver cancer, continent vs year
    # df_liver = reduceToTopCancer(top=5)
    # df_liver = reduceCountryToContinent(data=df_liver)
    # ThreeD_plot(data=df_liver, key='Cancer', value='Liver cancer ')

    #@ Analyze Liver cancer within Asia.
    # df_asia = reduceToTopCancer(top=5)
    # regionOrCountry(data=df_asia, cancer='Liver cancer ', regionCountry='asia', year=None)

    #@ Analyze Prostate cancer within Africa.
    # df_africa = import_data(FILENAME)
    # # Remove Code Column
    # df_africa = df_africa.drop(['Code'], axis=1)
    # # Turn all Cancer Death numbers into ints
    # df_africa.iloc[:, 2:] = df_africa.iloc[:, 2:].astype(int)
    # df_africa = df_africa[['Country', 'Year', 'Prostate cancer ']]
    # regionOrCountry(data=df_africa, cancer='Prostate cancer ', regionCountry='africa', year=None)


    plt.show()