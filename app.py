from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
import os.path as pt

app = Flask(__name__)


# Import data
s_dataPath = pt.join('data', 'Cancer Deaths by Country and Type Dataset.csv')
# df = pd.read_csv(r'data/Cancer Deaths by Country and Type Dataset.csv')
df=pd.read_csv(s_dataPath)

# Remove NaNs
df['Code'] = df.groupby('Country')['Code'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
df['Code'] = df['Code'].apply(lambda x: x.strip() if pd.notna(x) else x)
df.loc[df['Country'] == 'North America', 'Code'] = 'NA'

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

# Define lists of feature labels
cols = df.columns.to_numpy()
#cols = np.array([col.strip() for col in cols])
countries = df["Country"].unique().tolist()
codes = df["Code"].unique()
years = df["Year"].unique()
cancer_types = cols[3:]

default_column = "Country"
xaxis = 'Year'
yaxis = 'Liver cancer'

@app.route('/')
def index():
    data = df.copy()
    sort_by = request.args.get('sort_by', default_column)
    order = request.args.get('order', 'asc')
    data = data.sort_values(by=sort_by, ascending=(order == 'asc'))
    return render_template('index.html', data=data.values.tolist(), columns=cols)


@app.route('/plot')
def plot():
    ...
    plot_pic = None
    country_columns = countries.append('All Countries')
    return render_template('plot.html', country_columns = country_columns, year_columns = years, cancer_columns = cancer_types, xaxis = xaxis, yaxis = yaxis, plot_pic=plot_pic)


if __name__ == '__main__':
    app.run(debug=True)
