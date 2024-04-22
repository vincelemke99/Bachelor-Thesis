import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff

#pd.set_option('display.max_rows', None)  # Replace None with a number if you want to limit to a specific count
pd.set_option('display.max_columns', None)  # Adjusts how many columns are shown
pd.set_option('display.expand_frame_repr', False)  # Prevent DataFrame from being split across multiple pages
pd.set_option('display.width', None)  # Use None to automatically adjust to your screen width

df = pd.read_csv('trainings_data.csv')


#df.head
print(df.head)
print(df.shape)

# create a new column, date_parsed, with the parsed dates
df['date_parsed'] = pd.to_datetime(df['date'], format="%Y-%m-%d")
print(df['date_parsed'].head())
df.drop(['date'], axis=1, inplace=True)


df.drop(['name', 'email', 'produkt_code_pl', 'lead_id', 'kontakt_id', 'produkt_id'], axis="columns", inplace=True)

df.rename(columns={'date_parsed' : 'Datum', 'name' : 'Name', 'geschlecht' : 'Geschlecht', 'produkt_zeitraum_c' : 'Studientyp',
       'produkt_art_der_ausbildung_c' : 'Studiengangsart', 'produkt_standort' : 'Studienort',
       'produkt_fachbereich' : 'Fachbereich', 'produkt_name': 'Studiengang', 'studium_beginn' : 'Studien Beginn',
       'product_interest_type' : 'Conversion Type', 'is_converted' : 'Konvertiert', 'has_contract' : 'Vertragsabschluss'},inplace=True)

# Regression for all variables vs Selling_Price in plotly dark theme



# get the number of missing data points per column
missing_values_count = df.isnull().sum() # returns a pandas Series with the count of NaN (Not a Number) values in each column.
print(missing_values_count)

# how many total missing values do we have?
total_cells = np.product(df.shape) # df.shape returns a tuple representing the dimensions of the DataFrame df, which is the number of rows and columns, np.product() takes an iterable as an argument and multiplies its elements together. So here, it multiplies the number of rows by the number of columns to give the total number of cells (or elements) in the DataFrame.
total_missing = missing_values_count.sum() # ich adds up all the individual column missing value counts to get the total number of missing values across the entire DataFrame.
print(total_missing)
print(total_cells)

# percent of data that is missing
percent_missing = (total_missing/total_cells) * 100
print(percent_missing)



print(df.head())

df['Geschlecht'] = df['Geschlecht'].fillna('Unspecified')
columns_to_fill = ['Fachbereich', 'Studiengang', 'Studien Beginn', 'Conversion Type']
df[columns_to_fill] = df[columns_to_fill].fillna('Unknown')


print(df['Datum'])
print(df.nunique())