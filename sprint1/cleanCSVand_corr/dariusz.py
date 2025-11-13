from unicodedata import numeric

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


#Reading Dataset with List of unwanted values
df = pd.read_csv('Data.csv', na_values=['', ' ', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '-', '--', '...', 'Missing', 'missing', 'NA', 'na'], keep_default_na=True)

#Series and Countries columns list
series_column = 'Series Name'
countries_column = 'Country Name'
series_list = list(df[series_column].drop_duplicates())
countries_list = list(df[countries_column].drop_duplicates())

#Creating year column
all_columns = df.columns.tolist()
year_columns = all_columns[2:]

#Print list contain
def print_list(col_list):
    for l in col_list:
        print(l)

#Missing values in year columns
def missing_values_in_years():
    missing_values_years = df[year_columns].isnull().sum()
    for c, count in missing_values_years.items():
        if count > 0:
            print(f'{c}: {count} missing')

# Calculate missing fields percentage and Show missing percentages for each series/country
# with more than 50% missing data
def missing_values_in_rows():
    df['missing_count'] = df[year_columns].isnull().sum(axis=1)
    df['missing_percentage'] = (df['missing_count'] / len(year_columns)) * 100
    high_missing_data = df[df['missing_percentage'] > 50]
    return high_missing_data

def clean_dataset(high_missing_data):
    # Create cleaned dataset by removing  missing data combinations
    df_cleaned = df[df['missing_percentage'] <= 50]
    # Save cleaned dataset
    df_cleaned.to_csv('Data_Cleaned.csv', index=False)
    print(f"Cleaned dataset created: 'Data_Cleaned.csv'")
    print(f"Removed {len(high_missing_data)} rows")
    print(f"Original: {len(df)} rows")
    print(f"Cleaned: {len(df_cleaned)} rows")
    return df_cleaned

def plot_trend(series_name, country_names, year_columns):
    df_cleaned = pd.read_csv('Data_Cleaned.csv')
    plt.figure(figsize=(12, 6))

    #Converting year columns to numeric for plotting
    years = [int(year) for year in year_columns if year.replace(' ', '').isdigit()]

    for country in country_names:
        #getting data for series and country
        data = df_cleaned[(df_cleaned[series_column] == series_name) & (df_cleaned[countries_column] == country)]
        if not data.empty:
            #Extracting values for all years
            values = data[year_columns].iloc[0].values

            #Converting to numeric/handling missing values
            numeric_values = pd.to_numeric(values, errors='coerce')

            #Plot only non  NaN values
            valid_indices = ~np.isnan(numeric_values)
            valid_years = np.array(years)[valid_indices]
            valid_values = numeric_values[valid_indices]
            if len(valid_values) > 0:
                plt.plot(valid_years, valid_values, marker='o', label=country, linewidth=2)
    plt.title(f'{series_name} - Trend Analysis', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(series_name, fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()



if __name__ == '__main__':
    print(f'Series: {len(series_list)}')
    print(f'Countries: {len(countries_list)}')
    print(f'Years: {len(year_columns)}')
    high_missing_data = missing_values_in_rows()
    df_clean = clean_dataset(high_missing_data)
    comparison_country_list = ['Angola', 'Canada', 'Kazakhstan']
    plot_trend('Adolescent fertility rate (births per 1,000 women ages 15-19)', comparison_country_list, year_columns)



