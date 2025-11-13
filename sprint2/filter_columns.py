from unicodedata import numeric
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


"""
# Series and Countries columns list
series_column = 'Series Name'
countries_column = 'Country Name'

def load_dataset():
    # Reading Dataset with List of unwanted values
    df = pd.read_csv('Data.csv',
                     na_values=['', ' ', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '-', '--', '...',
                                'Missing', 'missing', 'NA', 'na'], keep_default_na=True)
    return df


#series_list = list(df[series_column].drop_duplicates())
#countries_list = list(df[countries_column].drop_duplicates())
"""


# Reading Dataset with List of unwanted values
df = pd.read_csv('Data.csv',
                 na_values=['', ' ', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '-', '--', '...',
                            'Missing', 'missing', 'NA', 'na'], keep_default_na=True)

# Series and Countries columns list
series_column = 'Series Name'
countries_column = 'Country Name'
series_list = list(df[series_column].drop_duplicates())
countries_list = list(df[countries_column].drop_duplicates())

# Creating year column
all_columns = df.columns.tolist()
year_columns = all_columns[2:]


# Missing functions that need to be added
def missing_values_in_rows(threshold=0.7):
    """
    Identify rows with high percentage of missing values
    """
    print("Analyzing missing values...")
    #df = load_dataset()

    # Creating year column
    #all_columns = df.columns.tolist()
    #year_columns = all_columns[2:]

    # Calculate missing percentage for each row
    missing_percentages = []
    for idx, row in df.iterrows():
        total_cells = len(year_columns)
        missing_cells = row[year_columns].isna().sum()
        missing_percentage = missing_cells / total_cells
        missing_percentages.append(missing_percentage)

    # Identify rows with missing values above threshold
    high_missing_indices = [i for i, perc in enumerate(missing_percentages) if perc > threshold]

    print(f"Found {len(high_missing_indices)} rows with more than {threshold * 100}% missing data")
    return high_missing_indices


def clean_dataset(high_missing_indices):
    """
    Clean the dataset by removing rows with high missing values
    """
    print("Cleaning dataset...")
    #Load dataset
    #df = load_dataset()
    # Remove rows with high missing values
    df_clean = df.drop(high_missing_indices).reset_index(drop=True)
    # Creating year column
    all_columns = df.columns.tolist()
    year_columns = all_columns[2:]

    # Fill remaining missing values with forward and backward fill
    df_clean[year_columns] = df_clean[year_columns].apply(pd.to_numeric, errors='coerce')
    df_clean[year_columns] = df_clean[year_columns].ffill(axis=1).bfill(axis=1)
    # Save cleaned dataset
    df_clean.to_csv('Data_Cleaned.csv', index=False)
    print("Cleaned dataset saved as 'Data_Cleaned.csv'")

    return df_clean


def manual_selection_interface():
    """Simple manual selection interface"""
    print("\n" + "=" * 60)
    print("MANUAL SELECTION INTERFACE")
    print("=" * 60)

    # Step 1: Select Series
    print("\nSTEP 1: Select a Series")
    print("-" * 30)
    for i, series in enumerate(series_list[:20], 1):  # Show first 20
        print(f"{i}. {series}")
    print("... and more")

    while True:
        try:
            series_choice = input(f"\nEnter series number (1-{len(series_list)}) or name: ").strip()
            if series_choice.isdigit():
                selected_series = series_list[int(series_choice) - 1]
            else:
                # Search for series by name
                matching_series = [s for s in series_list if series_choice.lower() in s.lower()]
                if matching_series:
                    print("Matching series:")
                    for i, s in enumerate(matching_series[:10], 1):
                        print(f"{i}. {s}")
                    choice = input("Select number: ").strip()
                    if choice.isdigit() and 1 <= int(choice) <= len(matching_series):
                        selected_series = matching_series[int(choice) - 1]
                    else:
                        selected_series = matching_series[0]
                else:
                    print("Series not found. Please try again.")
                    continue
            break
        except (ValueError, IndexError):
            print("Invalid selection. Please try again.")

    print(f"✓ Selected series: {selected_series}")

    # Step 2: Show available countries for selected series
    available_countries = df[df[series_column] == selected_series][countries_column].unique()
    print(f"\nSTEP 2: Select Countries (found {len(available_countries)} countries)")
    print("-" * 30)

    # Show first 15 countries
    for i, country in enumerate(available_countries[:15], 1):
        print(f"{i}. {country}")
    if len(available_countries) > 15:
        print(f"... and {len(available_countries) - 15} more")

    selected_countries = []
    while True:
        country_input = input("\nEnter country numbers (comma-separated) or names (comma-separated): ").strip()

        if country_input:
            if country_input.replace(',', '').replace(' ', '').isdigit():
                # User entered numbers
                numbers = [int(x.strip()) for x in country_input.split(',')]
                selected_countries = [available_countries[i - 1] for i in numbers if 1 <= i <= len(available_countries)]
            else:
                # User entered names
                country_names = [name.strip() for name in country_input.split(',')]
                selected_countries = []
                for name in country_names:
                    matching = [c for c in available_countries if name.lower() in c.lower()]
                    if matching:
                        selected_countries.extend(matching[:1])  # Take first match
                    else:
                        print(f"Country '{name}' not found. Skipping.")

            if selected_countries:
                print(f"✓ Selected countries: {', '.join(selected_countries)}")
                break
            else:
                print("No valid countries selected. Please try again.")
        else:
            print("Please enter at least one country.")

    # Step 3: Select years to compare
    print(f"\nSTEP 3: Select Years (available: {len(year_columns)} years)")
    print("-" * 30)
    print("Available years:", ', '.join(year_columns[:10]), "...")

    while True:
        year_input = input("\nEnter years to compare (comma-separated, 'all', or range like '2000-2010'): ").strip()

        if year_input.lower() == 'all':
            selected_years = year_columns
            break
        elif '-' in year_input and year_input.replace('-', '').isdigit():
            # Range input like "2000-2010"
            start, end = map(int, year_input.split('-'))
            selected_years = [year for year in year_columns if start <= int(year) <= end]
            if selected_years:
                break
            else:
                print("No years found in that range.")
        else:
            # Specific years
            year_choices = [y.strip() for y in year_input.split(',')]
            selected_years = [year for year in year_choices if year in year_columns]
            if selected_years:
                break
            else:
                print("No valid years found. Please try again.")

    print(f"✓ Selected years: {len(selected_years)} years")

    return selected_series, selected_countries, selected_years


def plot_trend(series_name, country_names, year_columns):
    """
    Plot trend comparison for selected series, countries, and years
    """
    try:
        df_cleaned = pd.read_csv('Data_Cleaned.csv')
    except FileNotFoundError:
        print("Cleaned dataset not found. Using original dataset.")
        df_cleaned = df

    plt.figure(figsize=(12, 6))

    # Convert year columns to numeric for plotting
    years = [int(year) for year in year_columns if str(year).replace(' ', '').isdigit()]

    for country in country_names:
        # Get data for series and country
        data = df_cleaned[(df_cleaned[series_column] == series_name) & (df_cleaned[countries_column] == country)]
        if not data.empty:
            # Extract values for selected years
            values = data[year_columns].iloc[0].values

            # Convert to numeric/handling missing values
            numeric_values = pd.to_numeric(values, errors='coerce')

            # Plot only non-NaN values
            valid_indices = ~np.isnan(numeric_values)
            valid_years = np.array(years)[valid_indices]
            valid_values = numeric_values[valid_indices]

            if len(valid_values) > 0:
                plt.plot(valid_years, valid_values, marker='o', label=country, linewidth=2)
            else:
                print(f"No valid data for {country}")
        else:
            print(f"No data found for {country}")

    plt.title(f'{series_name}\nTrend Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(series_name, fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()

def main():
    print(f'Series: {len(series_list)}')
    print(f'Countries: {len(countries_list)}')
    print(f'Years: {len(year_columns)}')

    # Clean the dataset first
    try:
        high_missing_data = missing_values_in_rows()
        df_clean = clean_dataset(high_missing_data)
    except Exception as e:
        print(f"Error during data cleaning: {e}")
        print("Continuing with original dataset...")

    # Use manual selection
    try:
        series, countries, years = manual_selection_interface()

        print(f"\n" + "=" * 60)
        print("GENERATING VISUALIZATION...")
        print(f"Series: {series}")
        print(f"Countries: {', '.join(countries)}")
        print(f"Years: {len(years)} selected")
        print("=" * 60)

        plot_trend(series, countries, years)

        # Option to run again
        while True:
            again = input("\nWould you like to create another comparison? (y/n): ").strip().lower()
            if again in ['y', 'yes']:
                series, countries, years = manual_selection_interface()
                plot_trend(series, countries, years)
            else:
                print("Goodbye!")
                break

    except KeyboardInterrupt:
        print("\n\nProgram interrupted by user. Goodbye!")
    except Exception as e:
        print(f"An error occurred: {e}")



if __name__ == '__main__':
    main()