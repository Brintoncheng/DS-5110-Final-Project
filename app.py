import matplotlib
matplotlib.use('Agg')
from flask import Flask, request, render_template, redirect, url_for, send_from_directory
import pandas as pd
import numpy as np
from initialization import import_data, clean_data, initialize_variables, filter_data
from constants import FILENAME
from visualizations import *
import os

app = Flask(__name__)


# Import data
df = import_data(FILENAME)

# Clean Data
df = clean_data(df)

# Initialize Variables
cols, countries, years, cancer_types, country, year, cancer, new_cols, default_column = initialize_variables(df)
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
    if request.method == "POST":
        country = request.form.get('country')
        year = request.form.get('year')
        cancer = request.form.get('cancer')
        country = None if country == '' else country
        year = None if year == '' else int(year) if year.isdigit() else None
        cancer = None if cancer == '' else cancer
        print(country, year, cancer)

        plot_object = choose_plot(data=data, country=country, year=year, cancer=cancer)
        plot_png = save_plot_to_png(plot_object, 'plot_png')
        return render_template('plot.html', country_columns=countries, year_columns=years, cancer_columns=cancer_types, plot_png=plot_png)

    else:
        return render_template('plot.html', country_columns = countries, year_columns = years, cancer_columns = cancer_types)

@app.route('/static/plots/<filename>')
def plot_png(filename):
    return send_from_directory('static/plots', filename)

@app.route('/thanks')
def thanks():
    return render_template('thanks.html')

if __name__ == '__main__':
    if not os.path.exists('static/plots'):
        os.makedirs('static/plots')
    app.run(debug=True)
