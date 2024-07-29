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
'''
#! ToDo:
#!      1. Scaling on 2D-y and 3D-z, apply log only necessary.
#!      2. Simplify/combine the 3 2D-plot-function into one.
#!      3. Perhaps limit the len/size of Cancer/Country/Year, for better visualization.
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

    plt.figure(figsize=[data['Country'].unique().size, 20])
    g = sb.barplot(data=df_filter, x='Country', y=cancer)
    g.tick_params(axis='x', rotation=90)
    g.set(yscale='log', xlabel='Country', ylabel=cancer+'death (log)')
    g.set_title(s_title)
    return g


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
    fig = plt.figure(figsize=[i_num1, i_num2]) #! might need change
    ax = fig.add_subplot(projection='3d')
    ax.xaxis.set_ticks([i for i in range(i_num1)])
    ax.xaxis.set_ticklabels(l_xlabels)
    ax.yaxis.set_ticks([i for i in range(i_num2)])
    ax.yaxis.set_ticklabels(data[s_ylabel].unique())
    ax.set_xlabel(f'x-{s_xlabel}')
    ax.set_ylabel(f'y-{s_ylabel}')
    ax.set_zlabel(f'z-{s_zlabel}')
    ax.set_zlim(0, dz.max())
    ax.set_title(s_title)
    ax.bar3d(x, y, z, dx, dy, dz)
    return fig


def dataClean():
    ''' Just the process of data import and clean from app.py '''
    # Import data
    s_dataPath = pt.join('data', 'Cancer Deaths by Country and Type Dataset.csv')
    df=pd.read_csv(s_dataPath)

    # Remove NaNs
    df['Code'] = df.groupby('Country')['Code'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
    df['Code'] = df['Code'].apply(lambda x: x.strip() if pd.notna(x) else x)
    df.loc[df['Country'] == 'North America', 'Code'] = 'NA'

    # Remove Code Column
    df = df.drop(['Code'], axis=1)

    # Remove non_countries
    non_countries = [
        "American Samoa",
        "Andean Latin America",
        "Australasia",
        "Bermuda",
        "Caribbean",
        "Central Asia",
        "Central Europe",
        "Central Latin America",
        "Central Sub-Saharan Africa",
        "East Asia",
        "Eastern Europe",
        "Eastern Sub-Saharan Africa",
        "England",
        "Greenland",
        "Guam",
        "Latin America and Caribbean",
        "Micronesia (country)",
        "North Africa and Middle East",
        "North America",
        "Northern Ireland",
        "Northern Mariana Islands",
        "Oceania",
        "Palestine",
        "Puerto Rico",
        "Scotland",
        "South Asia",
        "Southeast Asia",
        "Southern Latin America",
        "Southern Sub-Saharan Africa",
        "Sub-Saharan Africa",
        "Taiwan",
        "Timor",
        "Tropical Latin America",
        "United States Virgin Islands",
        "Wales",
        "Western Europe",
        "Western Sub-Saharan Africa",
        "World"
    ]
    df = df[~df['Country'].isin(non_countries)]
    return df

if __name__ == '__main__':
    data = dataClean()
    # print(data)

    #@ Test Case 1
    country = 'Afghanistan'
    year = None
    cancer = 'Liver cancer '
    # plot(data, country, year, cancer)

    #@ Test Case 2
    country = 'Afghanistan'
    year = 1990
    cancer = None
    # plot(data, country, year, cancer)

    #@ Test Case 3
    country = None
    year = 1990
    cancer = 'Liver cancer '
    # plot(data, country, year, cancer)

    #@ Test Case 4
    country = 'Afghanistan'
    year = None
    cancer = None
    # ret = plot(data, country, year, cancer)

    #@ Test Case 5
    country = None
    year = None
    cancer = 'Liver cancer '
    # ret = plot(data, country, year, cancer)

    #@ Test Case 6
    country = None
    year = 1990
    cancer = None
    ret = choose_plot(data, country, year, cancer)



    plt.show()