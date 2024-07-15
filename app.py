from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np

app = Flask(__name__)


# Import data
df = pd.read_csv(r'data/Cancer Deaths by Country and Type Dataset.csv')

# Remove NaNs
df['Code'] = df.groupby('Country')['Code'].transform(lambda x: x.mode()[0] if not x.mode().empty else None)
df['Code'] = df['Code'].apply(lambda x: x.strip() if pd.notna(x) else x)
df.loc[df['Country'] == 'North America', 'Code'] = 'NA'

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
    data = data.sort_values(by=sort_by, ascending= order=='asc')
    return render_template('index.html', data = data.values.tolist(), columns = cols)


@app.route('/plot')
def plot():
    ...
    plot_pic = None
    country_columns = countries.append('All Countries')
    return render_template('plot.html', country_columns = country_columns, year_columns = years, cancer_columns = cancer_types, xaxis = xaxis, yaxis = yaxis, plot_pic=plot_pic)


if __name__ == '__main__':
    app.run(debug=True)
