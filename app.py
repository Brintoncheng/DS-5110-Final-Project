from flask import Flask, request, render_template, redirect, url_for
import pandas as pd
import numpy as np
from initialization import import_data, clean_data, initialize_variables, filter_data
from constants import FILENAME
from visualizations import *

app = Flask(__name__)


# Import data
df = import_data(FILENAME)

# Clean Data
df = clean_data(df)

# Initialize Variables
cols, countries, years, cancer_types, country, year, cancer, new_cols, default_column, xaxis, yaxis = initialize_variables(df)
data = df.copy()


print(country, year, cancer)

@app.route('/', methods=["GET", "POST"])
def index():
    global data
    # print(country, year, cancer)
    if request.method == "POST":
        country = request.form.get('country', 'ALL')
        year = request.form.get('year', 'ALL')
        if year != 'ALL':
            year = int(year)
        cancer = request.form.get('cancer', 'ALL')
        print(country, year, cancer)
        data = filter_data(df, country=country, year=year, cancer=cancer)
        sort_by = default_column
        order = 'asc'
        print(data.shape)
    else:
        sort_by = request.args.get('sort_by', default_column)
        order = request.args.get('order', 'asc')
    
    data = data.sort_values(by=sort_by, ascending=(order=='asc'))
    return render_template('index.html', data = data.values.tolist(), columns = data.columns.tolist(),
                           countries = countries, years = years, cancers = cancer_types)


@app.route('/plot', methods=["GET", "POST"])
def plot():
    ...
    plot_pic = None
    return render_template('plot.html', country_columns = countries, year_columns = years, cancer_columns = cancer_types, xaxis = xaxis, yaxis = yaxis, plot_pic=plot_pic)


if __name__ == '__main__':
    app.run(debug=True)
