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
cols = np.array([col.strip() for col in cols])
cancer_types = cols[3:]
countries = df["Country"].unique()
codes = df["Code"].unique()
years = df["Year"].unique()
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
    return render_template('plot.html', xaxis = xaxis, yaxis = yaxis, plot=plot)


if __name__ == '__main__':
    app.run(debug=True)
