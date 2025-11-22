from unicodedata import numeric
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from openai import OpenAI
import time

# Series and Countries columns list
series_column = 'Series Name'
countries_column = 'Country Name'

# Global variables
df = None
series_list = []
countries_list = []
year_columns = []

# Helper method to add file to directory
def add_file_to_directory(file_path):
    if not os.path.exists(file_path):
        print("File not found.")
        return False
    if file_path.endswith('.csv'):
        try:
            df = pd.read_csv(file_path)
            df.to_csv('Data.csv', index=False)
            print("File successfully saved")
            return True
        except Exception as e:
            print(f"Error reading file: {e}")
            return False
    else:
        print("Incorrect file type")
        return False

# Scan for CSV files and handle dataset loading
def scan_for_csv():
    all_files = os.listdir('.')
    csv_files = [file for file in all_files if file.endswith('.csv')]

    if len(csv_files) == 0:
        print("Dataset not found. Please add a file to directory.")
        file_path = input("Enter the path to your CSV file: ").strip().strip('"\'')
        if add_file_to_directory(file_path):
            return scan_for_csv()
        return None
    else:
        if "Data.csv" not in csv_files:
            print("Dataset not found. Please add a file to directory.")
            file_path = input("Enter the path to your CSV file: ").strip().strip('"\'')
            if add_file_to_directory(file_path):
                return scan_for_csv()
            return None
        else:
            if "Data_Cleaned.csv" not in csv_files:
                print("No cleaned data found")
                print("Starting cleaning process...")
                df = pd.read_csv('Data.csv',
                                 na_values=['', ' ', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan', 'None', 'none', '-',
                                            '--', '...', 'Missing', 'missing', 'NA', 'na'], keep_default_na=True)
                high_missing_data = missing_values_in_rows(df)
                cleaned_df = clean_dataset(high_missing_data, df)
                print("Dataset found and cleaned")
                return cleaned_df
            else:
                print("Dataset found")
                return pd.read_csv('Data_Cleaned.csv')

# Main dataset loading function
def load_dataset():
    global df, series_list, countries_list, year_columns

    print("Loading dataset...")
    df = scan_for_csv()

    if df is not None:
        series_list = list(df[series_column].drop_duplicates())
        countries_list = list(df[countries_column].drop_duplicates())
        all_columns = df.columns.tolist()
        year_columns = all_columns[2:]

        print(
            f"Dataset loaded successfully: {len(series_list)} series, {len(countries_list)} countries, {len(year_columns)} years")
        return True
    else:
        print("Failed to load dataset.")
        return False

# Identify rows with high percentage of missing values
def missing_values_in_rows(df, threshold=0.7):
    print("Analyzing missing values...")

    all_columns = df.columns.tolist()
    year_columns = all_columns[2:]

    missing_percentages = []
    for idx, row in df.iterrows():
        total_cells = len(year_columns)
        missing_cells = row[year_columns].isna().sum()
        missing_percentage = missing_cells / total_cells
        missing_percentages.append(missing_percentage)

    high_missing_indices = [i for i, perc in enumerate(missing_percentages) if perc > threshold]

    print(f"Found {len(high_missing_indices)} rows with more than {threshold * 100}% missing data")
    return high_missing_indices

# Clean the dataset by removing rows with high missing values
def clean_dataset(high_missing_indices, df):
    print("Cleaning dataset...")

    df_clean = df.drop(high_missing_indices).reset_index(drop=True)
    all_columns = df.columns.tolist()
    year_columns = all_columns[2:]

    df_clean[year_columns] = df_clean[year_columns].apply(pd.to_numeric, errors='coerce')
    df_clean[year_columns] = df_clean[year_columns].ffill(axis=1).bfill(axis=1)

    df_clean.to_csv('Data_Cleaned.csv', index=False)
    print("Cleaned dataset saved as 'Data_Cleaned.csv'")

    return df_clean

# Setup Ollama LLM
def setup_llm():
    try:
        client = OpenAI(
            base_url='http://localhost:11434/v1',
            api_key='ollama',
            timeout=30.0
        )
        return client
    except Exception as e:
        print(f"Ollama not available: {e}")
        return None

# Get the available models - auto-detect correct names
def get_available_model(client):
    try:
        # Listing installed models
        models = client.models.list()
        model_names = [model.id for model in models]


        # LLM Models Register
        print(f"Available models: {model_names}")
        model_patterns = [
            "tinyllama", "tinyllama:latest", "tinyllama:1.1b",
            "llama2", "llama2:latest", "llama2:7b",
            "codellama", "codellama:latest", "codellama:7b",
            "mistral", "mistral:latest", "mistral:7b",
            "gemma", "gemma:latest", "gemma:7b"
        ]

        # Extracting current LLM model
        for pattern in model_patterns:
            if pattern in model_names:
                print(f"Using model: {pattern}")
                return pattern

        for model_name in model_names:
            for pattern in model_patterns:
                if pattern in model_name.lower():
                    print(f"Using model: {model_name}")
                    return model_name

        # If nothing matches, use the first available model
        if model_names:
            print(f"Using first available model: {model_names[0]}")
            return model_names[0]
        else:
            print("No models found in Ollama")
            return None

    except Exception as e:
        print(f"Error getting models: {e}")
        return None

# Generate simple AI insights
def get_llm_insight_report(series_name, country_data, years):
    client = setup_llm()
    if not client:
        return "LLM not available for insights. Install Ollama from https://ollama.ai"

    try:
        model_name = get_available_model(client)
        if not model_name:
            return "No LLM models available."

        print(f"Generating simple insights with {model_name}...")

        data_summary = f"Data for {series_name}:\n"

        for country, data in country_data.items():
            if len(data['values']) > 1:
                first_val = data['values'][0]
                last_val = data['values'][-1]
                change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                trend = "decreasing" if change < 0 else "increasing" if change > 0 else "stable"
                data_summary += f"{country}: {trend} from {first_val:.0f} to {last_val:.0f}\n"

        prompt = f"""
        {data_summary}

        Give 2-3 simple sentences about these trends.
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You give very short simple analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150,
            timeout=30
        )

        insight = response.choices[0].message.content.strip()
        print("Insight generated!")
        return insight

    except Exception as e:
        return f"Insight failed: {str(e)}"

# Generate trend predictions
def get_llm_trend_prediction(series_name, country_data, prediction_years):
    client = setup_llm()
    if not client:
        return get_mathematical_trend_prediction(series_name, country_data, prediction_years)

    try:
        model_name = get_available_model(client)
        if not model_name:
            return get_mathematical_trend_prediction(series_name, country_data, prediction_years)

        print(f"Generating predictions with {model_name}...")

        # Build better data summary with trends
        data_summary = "Historical data:\n"
        for country, data in country_data.items():
            if len(data['values']) > 0:
                values = data['values']
                years = data['years']
                if len(values) > 1:
                    trend = "increasing" if values[-1] > values[0] else "decreasing" if values[-1] < values[
                        0] else "stable"
                    data_summary += f"{country}: {trend} from {values[0]:.3f} to {values[-1]:.3f} over {len(years)} years\n"
                else:
                    data_summary += f"{country}: current value {values[-1]:.3f}\n"

        # Much more specific prompt
        prompt = f"""
        You are a data analysis assistant. Based on the historical data below, predict future values.

        {data_summary}

        Series: {series_name}
        Prediction years: {prediction_years}

        Return ONLY valid JSON in this exact format:
        {{
            "predictions": {{
                "Country Name 1": [0.1, 0.2, 0.3, ...],
                "Country Name 2": [0.4, 0.5, 0.6, ...]
            }}
        }}

        Rules:
        - Return exactly {prediction_years} numbers per country
        - Make predictions realistic based on the trend
        - Keep values in the same range as historical data
        - Do NOT add any explanation, just the JSON
        """

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system",
                 "content": "You are a data analyst. You always return ONLY valid JSON without any additional text."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.3,  # Slightly higher for more creative but consistent predictions
            max_tokens=500,
            timeout=30
        )

        prediction_text = response.choices[0].message.content.strip()
        print(f"DEBUG: Raw LLM response: {prediction_text}")

        # Better JSON extraction
        import re
        json_match = re.search(r'\{.*\}', prediction_text, re.DOTALL)
        if json_match:
            prediction_text = json_match.group()

        try:
            predictions = json.loads(prediction_text)

            # Validate the structure
            if "predictions" in predictions and isinstance(predictions["predictions"], dict):
                # Ensure each country has the right number of predictions
                for country in predictions["predictions"]:
                    if len(predictions["predictions"][country]) != prediction_years:
                        print(f"Warning: {country} has wrong number of predictions. Using mathematical fallback.")
                        return get_mathematical_trend_prediction(series_name, country_data, prediction_years)

                print("LLM prediction ready!")
                return predictions
            else:
                print("Invalid prediction structure. Using mathematical fallback.")
                return get_mathematical_trend_prediction(series_name, country_data, prediction_years)

        except json.JSONDecodeError as e:
            print(f"JSON parsing failed: {e}")
            # Try to parse the incorrect format and convert it
            return parse_alternative_format(prediction_text, country_data, prediction_years)

    except Exception as e:
        print(f"LLM prediction failed: {e}")
        return get_mathematical_trend_prediction(series_name, country_data, prediction_years)

# Try to parse alternative JSON formats the LLM might return
def parse_alternative_format(prediction_text, country_data, prediction_years):
    try:
        data = json.loads(prediction_text)
        # Handle array format: [{"Country": "X", "Number": 0.1}, ...]
        if isinstance(data, list):
            predictions = {"predictions": {}}
            for item in data:
                if "Country" in item and "Number" in item:
                    country = item["Country"]
                    # Repeat the single number for all prediction years
                    predictions["predictions"][country] = [item["Number"]] * prediction_years

            if predictions["predictions"]:
                print("Converted alternative format to expected structure")
                return predictions

        # Handle other unexpected formats
        print("Could not parse alternative format. Using mathematical fallback.")
        return get_mathematical_trend_prediction(None, country_data, prediction_years)

    except:
        print("Failed to parse any format. Using mathematical fallback.")
        return get_mathematical_trend_prediction(None, country_data, prediction_years)

# Create manual predictions when LLM fails
def create_manual_predictions(country_data, prediction_years):
    predictions = {"predictions": {}}

    for country, data in country_data.items():
        if len(data['values']) > 0:
            values = data['values']
            last_value = values[-1]

            # Simple trend detection
            if len(values) >= 3:
                recent_trend = np.polyfit(range(len(values[-3:])), values[-3:], 1)[0]
                # Small adjustment based on recent trend
                adjustment = recent_trend * 0.5  # Dampen the trend
            else:
                adjustment = 0

            # Create predictions with slight trend continuation
            future_values = []
            for i in range(prediction_years):
                pred_value = last_value + (adjustment * (i + 1))
                # Ensure reasonable bounds
                if any(keyword in country_data.get('series_name', '').lower() for keyword in ['hiv', 'rate']):
                    pred_value = max(0, pred_value)  # Can't go negative
                future_values.append(round(pred_value, 3))

            predictions["predictions"][country] = future_values

    print("Created enhanced manual predictions")
    return predictions

# Mathematical fallback for trend predictions
def get_mathematical_trend_prediction(series_name, country_data, prediction_years):
    predictions = {"predictions": {}}

    for country, data in country_data.items():
        if len(data['values']) >= 2:
            x = np.array(data['years'])
            y = np.array(data['values'])

            # Remove any NaN values
            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(y_clean) >= 2:
                # Use appropriate degree based on data points
                degree = min(2, len(y_clean) - 1)

                try:
                    z = np.polyfit(x_clean, y_clean, degree)
                    p = np.poly1d(z)

                    # Calculate future values with trend
                    last_year = max(x_clean)
                    future_values = []

                    for i in range(1, prediction_years + 1):
                        pred = p(last_year + i)

                        # Apply constraints based on data type
                        if any(keyword in series_name.lower() for keyword in ['hiv', 'mortality', 'rate', 'incidence']):
                            pred = max(0, pred)  # These can't be negative
                            # Cap at reasonable maximum (e.g., 2x the max historical value)
                            max_historical = max(y_clean)
                            pred = min(pred, max_historical * 2)

                        future_values.append(round(pred, 3))

                    predictions["predictions"][country] = future_values

                except (np.linalg.LinAlgError, ValueError):
                    # If polynomial fit fails, use simple linear extrapolation
                    if len(y_clean) >= 2:
                        slope = (y_clean[-1] - y_clean[0]) / (x_clean[-1] - x_clean[0])
                        last_value = y_clean[-1]
                        future_values = [round(last_value + slope * i, 3) for i in range(1, prediction_years + 1)]
                        predictions["predictions"][country] = future_values
                    else:
                        # Fallback to repeating last value
                        predictions["predictions"][country] = [round(y_clean[-1], 3)] * prediction_years
            else:
                # Not enough clean data
                last_value = data['values'][-1] if len(data['values']) > 0 else 0
                predictions["predictions"][country] = [round(last_value, 3)] * prediction_years
        else:
            # Single data point or no data
            last_value = data['values'][-1] if len(data['values']) > 0 else 0
            predictions["predictions"][country] = [round(last_value, 3)] * prediction_years

    return predictions

# Plot line chart for trend comparison
def plot_line_chart(series_name, country_data, years):
    plt.figure(figsize=(12, 6))

    for country, data in country_data.items():
        plt.plot(data['years'], data['values'], marker='o', label=country, linewidth=2)

    plt.title(f'{series_name}\nTrend Comparison', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(series_name, fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(years, rotation=45)
    plt.tight_layout()
    plt.show()

# Plot bar chart for country comparison
def plot_bar_chart(series_name, country_data, years=None):
    plt.figure(figsize=(10, 6))
    countries_with_data = list(country_data.keys())
    latest_values = [country_data[country]['latest_value'] for country in countries_with_data]

    bars = plt.bar(countries_with_data, latest_values,
                   color=plt.cm.Set3(np.linspace(0, 1, len(countries_with_data))))
    plt.title(f'{series_name}\nCountry Comparison (Latest Values)', fontsize=14, fontweight='bold')
    plt.xlabel('Country', fontsize=12)
    plt.ylabel(f'{series_name}', fontsize=10)
    plt.xticks(rotation=45)

    for bar, value in zip(bars, latest_values):
        if value is not None:
            plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f'{value:.1f}', ha='center', va='bottom', fontweight='bold')

    plt.tight_layout()
    plt.show()

# Plot heatmap for value distribution
def plot_heatmap(series_name, country_data, years):
    if len(country_data) < 2:
        print("Heatmap requires at least 2 countries for meaningful comparison.")
        return

    plt.figure(figsize=(12, 6))
    heatmap_data = []
    heatmap_countries = []

    for country, data in country_data.items():
        country_series = []
        for year in years:
            # Convert year to integer for comparison
            year_int = int(year)

            # Check if this year exists in the country's data
            if year_int in data['years']:
                # Find the index of the year in the years list
                try:
                    year_index = data['years'].index(year_int)
                    country_series.append(data['values'][year_index])
                except (ValueError, IndexError):
                    country_series.append(np.nan)
            else:
                country_series.append(np.nan)

        heatmap_data.append(country_series)
        heatmap_countries.append(country)

    heatmap_data = np.array(heatmap_data)

    # Create heatmap
    im = plt.imshow(heatmap_data, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    plt.title(f'{series_name}\nValue Heatmap Across Years', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel('Country', fontsize=10)
    plt.xticks(range(len(years)), years, rotation=45)
    plt.yticks(range(len(heatmap_countries)), heatmap_countries)

    # Add colorbar
    cbar = plt.colorbar(im)
    cbar.set_label(series_name, rotation=270, labelpad=15)

    # Add value annotations
    for i in range(len(heatmap_countries)):
        for j in range(len(years)):
            if not np.isnan(heatmap_data[i, j]):
                plt.text(j, i, f'{heatmap_data[i, j]:.0f}',
                         ha='center', va='center', fontsize=8,
                         color='black' if heatmap_data[i, j] < np.nanmax(heatmap_data) * 0.7 else 'white')

    plt.tight_layout()
    plt.show()

# Plot trend predictions as a line chart with future projections
def plot_trend_prediction(series_name, country_data, years, prediction_years, predictions):

    plt.figure(figsize=(14, 8))

    # Convert years to integers for calculations
    years_int = [int(year) for year in years]

    # Plot historical data and predictions for each country
    for country, data in country_data.items():
        if country in predictions["predictions"]:
            # Historical data (convert years to int for plotting)
            historical_years = [int(year) for year in data['years']]
            plt.plot(historical_years, data['values'], marker='o', label=f'{country} (Historical)', linewidth=2,
                     markersize=6)

            # Future predictions
            future_values = predictions["predictions"][country]
            last_historical_year = max(historical_years)
            future_years = list(range(last_historical_year + 1, last_historical_year + 1 + len(future_values)))

            # Plot prediction line (dashed)
            plt.plot(future_years, future_values, '--', marker='x',
                     label=f'{country} (Predicted)', linewidth=2, markersize=8,
                     color=plt.gca().lines[-1].get_color())

            # Add prediction labels
            for i, (year, value) in enumerate(zip(future_years, future_values)):
                plt.annotate(f'{value:.1f}', (year, value),
                             textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8,
                             bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7))

    # Add vertical line separating historical and predicted data
    if country_data:
        first_country = list(country_data.keys())[0]
        historical_years_first = [int(year) for year in country_data[first_country]['years']]
        last_historical_year = max(historical_years_first)
        plt.axvline(x=last_historical_year, color='red', linestyle=':', alpha=0.7, label='Prediction Start')

        plt.text(last_historical_year + 0.5, plt.ylim()[1] * 0.9,
                 f'Future Prediction\n{prediction_years} years',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    plt.title(f'{series_name}\nHistorical Data + {prediction_years}-Year Prediction', fontsize=14, fontweight='bold')
    plt.xlabel('Year', fontsize=12)
    plt.ylabel(series_name, fontsize=10)
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks to include both historical and future years
    all_years = years_int + list(range(max(years_int) + 1, max(years_int) + prediction_years + 1))
    plt.xticks(all_years, rotation=45)

    plt.tight_layout()
    plt.show()

# Manual selection interface
def manual_selection_interface():
    global df, series_list, countries_list, year_columns

    if df is None:
        print("Dataset not loaded. Please load dataset first.")
        return None, None, None, None

    print("\n" + "=" * 60)
    print("MANUAL SELECTION INTERFACE")
    print("=" * 60)

    print("\nSTEP 1: Select a Series")
    print("-" * 30)
    for i, series in enumerate(series_list[:20], 1):
        print(f"{i}. {series}")
    print("... and more")

    while True:
        try:
            series_choice = input(f"\nEnter series number (1-{len(series_list)}) or name: ").strip()
            if series_choice.isdigit():
                selected_series = series_list[int(series_choice) - 1]
            else:
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

    print(f"Selected series: {selected_series}")

    available_countries = df[df[series_column] == selected_series][countries_column].unique()
    print(f"\nSTEP 2: Select Countries (found {len(available_countries)} countries)")
    print("-" * 30)

    for i, country in enumerate(available_countries[:15], 1):
        print(f"{i}. {country}")
    if len(available_countries) > 15:
        print(f"... and {len(available_countries) - 15} more")

    selected_countries = []
    while True:
        country_input = input("\nEnter country numbers (comma-separated) or names (comma-separated): ").strip()

        if country_input:
            if country_input.replace(',', '').replace(' ', '').isdigit():
                numbers = [int(x.strip()) for x in country_input.split(',')]
                selected_countries = [available_countries[i - 1] for i in numbers if 1 <= i <= len(available_countries)]
            else:
                country_names = [name.strip() for name in country_input.split(',')]
                selected_countries = []
                for name in country_names:
                    matching = [c for c in available_countries if name.lower() in c.lower()]
                    if matching:
                        selected_countries.extend(matching[:1])
                    else:
                        print(f"Country '{name}' not found. Skipping.")

            if selected_countries:
                print(f"Selected countries: {', '.join(selected_countries)}")
                break
            else:
                print("No valid countries selected. Please try again.")
        else:
            print("Please enter at least one country.")

    print(f"\nSTEP 3: Select Years (available: {len(year_columns)} years)")
    print("-" * 30)
    print("Available years:", ', '.join(year_columns[:10]), "...")

    while True:
        year_input = input("\nEnter years to compare (comma-separated, 'all', or range like '2000-2010'): ").strip()

        if year_input.lower() == 'all':
            selected_years = year_columns
            break
        elif '-' in year_input and year_input.replace('-', '').isdigit():
            start, end = map(int, year_input.split('-'))
            selected_years = [year for year in year_columns if start <= int(year) <= end]
            if selected_years:
                break
            else:
                print("No years found in that range.")
        else:
            year_choices = [y.strip() for y in year_input.split(',')]
            selected_years = [year for year in year_choices if year in year_columns]
            if selected_years:
                break
            else:
                print("No valid years found. Please try again.")

    print(f"Selected years: {len(selected_years)} years")

    print(f"\nSTEP 4: Select Chart Type")
    print("-" * 30)
    print("Available chart types:")
    print("1. Line Chart - Trend comparison over time")
    print("2. Bar Chart - Side-by-side country comparison")
    print("3. Heatmap - Value distribution across years")

    chart_types = {
        '1': 'line',
        '2': 'bar',
        '3': 'heatmap'
    }

    while True:
        chart_choice = input("\nEnter chart type number (1-3): ").strip()
        if chart_choice in chart_types:
            selected_chart = chart_types[chart_choice]
            break
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")

    chart_names = {
        'line': 'Line Chart',
        'bar': 'Bar Chart',
        'heatmap': 'Heatmap'
    }
    print(f"Selected chart: {chart_names[selected_chart]}")

    return selected_series, selected_countries, selected_years, selected_chart

# Interface for AI analysis options
def analysis_selection_interface(series_name, countries, years):
    print(f"\n" + "=" * 60)
    print("AI ANALYSIS OPTIONS")
    print("=" * 60)
    print(f"Series: {series_name}")
    print(f"Countries: {', '.join(countries)}")
    print(f"Years: {len(years)} selected")
    print("-" * 60)

    print("\nSelect analysis type:")
    print("1. Simple Insight Report (Fast, basic analysis)")
    print("2. Simple Trend Prediction (Fast, basic forecast)")
    print("3. View Chart Only (no additional analysis)")

    while True:
        analysis_choice = input("\nEnter choice (1-3): ").strip()
        if analysis_choice in ['1', '2', '3']:
            break
        else:
            print("Invalid selection. Please enter 1, 2, or 3.")

    prediction_years = 0
    if analysis_choice == '2':
        print(f"\nSelect prediction timeframe:")
        print("1. 5-year prediction")
        print("2. 10-year prediction")
        print("3. 15-year prediction")

        while True:
            years_choice = input("\nEnter choice (1-3): ").strip()
            if years_choice == '1':
                prediction_years = 5
                break
            elif years_choice == '2':
                prediction_years = 10
                break
            elif years_choice == '3':
                prediction_years = 15
                break
            else:
                print("Invalid selection. Please enter 1, 2, or 3.")

    return analysis_choice, prediction_years

# Generate and display AI insights
def generate_insight_report(series_name, country_data, years):
    print(f"\n" + "=" * 70)
    print("AI INSIGHT REPORT")
    print("=" * 70)

    print(f"\nAnalyzing {series_name}...")
    insight_report = get_llm_insight_report(series_name, country_data, years)

    print(insight_report)
    print("\n" + "=" * 70)

    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"insight_report_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write(f"AI Insight Report - {series_name}\n")
        f.write("=" * 50 + "\n\n")
        f.write(insight_report)

    print(f"Report saved as: {filename}")

# Plot a dedicated line chart for trend predictions
def plot_trend_prediction_chart(series_name, enhanced_country_data, years, prediction_years):
    plt.figure(figsize=(14, 8))

    # Convert years to integers for calculations
    years_int = [int(year) for year in years]

    # Plot historical data and predictions for each country
    for country, data in enhanced_country_data.items():
        # Plot historical data
        historical_years = [int(year) for year in data['years']]
        plt.plot(historical_years, data['values'], marker='o', label=f'{country} (Historical)',
                 linewidth=2, markersize=6, alpha=0.8)

        # Plot future predictions if available
        if 'future_values' in data and 'future_years' in data:
            future_years = data['future_years']
            future_values = data['future_values']

            # Plot prediction line (dashed)
            plt.plot(future_years, future_values, '--', marker='x',
                     label=f'{country} ({prediction_years}yr Prediction)', linewidth=2,
                     markersize=8, alpha=0.8, color=plt.gca().lines[-1].get_color())

            # Add prediction value labels
            for i, (year, value) in enumerate(zip(future_years, future_values)):
                plt.annotate(f'{value:.1f}', (year, value),
                             textcoords="offset points", xytext=(0, 8), ha='center',
                             fontsize=8, fontweight='bold',
                             bbox=dict(boxstyle="round,pad=0.2", facecolor="yellow", alpha=0.7))

    # Add vertical line and prediction zone
    if enhanced_country_data:
        first_country = list(enhanced_country_data.keys())[0]
        historical_years_first = [int(year) for year in enhanced_country_data[first_country]['years']]
        last_historical_year = max(historical_years_first)

        # Vertical line at prediction start
        plt.axvline(x=last_historical_year, color='red', linestyle=':', alpha=0.7,
                    label='Prediction Start', linewidth=2)

        # Shaded area for prediction period
        plt.axvspan(last_historical_year, last_historical_year + prediction_years,
                    alpha=0.1, color='red', label='Prediction Period')

        # Prediction label
        plt.text(last_historical_year + prediction_years / 2, plt.ylim()[1] * 0.95,
                 f'AI Prediction\nNext {prediction_years} Years',
                 ha='center', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral", alpha=0.8))

    plt.title(f'{series_name}\nHistorical Trends + {prediction_years}-Year AI Prediction',
              fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Year', fontsize=12, fontweight='bold')
    plt.ylabel(series_name, fontsize=12, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)

    # Set x-axis ticks
    all_years = years_int + list(range(max(years_int) + 1, max(years_int) + prediction_years + 1))
    plt.xticks(all_years, rotation=45)

    plt.tight_layout()
    plt.show()

# Generate and display long-term trend predictions with visual charts
def generate_trend_prediction(series_name, country_data, years, prediction_years):
    print(f"\n" + "=" * 70)
    print(f"{prediction_years}-YEAR TREND PREDICTION")
    print("=" * 70)

    print(f"\nGenerating {prediction_years}-year predictions for {series_name}...")
    predictions = get_llm_trend_prediction(series_name, country_data, prediction_years)

    # Debug: Print what we actually got from the LLM
    print(f"DEBUG: Predictions type: {type(predictions)}")
    print(f"DEBUG: Predictions content: {predictions}")

    # Handle different response formats
    if isinstance(predictions, list):
        print("LLM returned list instead of dictionary. Using mathematical fallback.")
        predictions = get_mathematical_trend_prediction(series_name, country_data, prediction_years)
    elif "predictions" not in predictions:
        print("LLM response missing 'predictions' key. Using mathematical fallback.")
        predictions = get_mathematical_trend_prediction(series_name, country_data, prediction_years)

    print(f"\nPREDICTION SUMMARY:")
    print("-" * 50)

    # Safe iteration with error handling
    if "predictions" in predictions and isinstance(predictions["predictions"], dict):
        for country, future_values in predictions["predictions"].items():
            if country in country_data and len(country_data[country]['values']) > 0:
                last_actual = country_data[country]['values'][-1]
                if isinstance(future_values, list) and len(future_values) > 0:
                    first_predicted = future_values[0]
                    if last_actual != 0:
                        predicted_change = ((first_predicted - last_actual) / last_actual * 100)
                        trend = "Increase" if predicted_change > 0 else "Decrease" if predicted_change < 0 else "Stable"
                        print(
                            f"{trend} {country}: {last_actual:.1f} → {first_predicted:.1f} ({predicted_change:+.1f}%)")
                else:
                    print(f"⚠️  {country}: Invalid prediction format")
    else:
        print("No valid predictions generated. Using mathematical fallback.")
        predictions = get_mathematical_trend_prediction(series_name, country_data, prediction_years)
        # Try again with mathematical predictions
        if "predictions" in predictions:
            for country, future_values in predictions["predictions"].items():
                if country in country_data and len(country_data[country]['values']) > 0:
                    last_actual = country_data[country]['values'][-1]
                    if isinstance(future_values, list) and len(future_values) > 0:
                        first_predicted = future_values[0]
                        if last_actual != 0:
                            predicted_change = ((first_predicted - last_actual) / last_actual * 100)
                            trend = "Increase" if predicted_change > 0 else "Decrease" if predicted_change < 0 else "Stable"
                            print(
                                f"{trend} {country}: {last_actual:.1f} → {first_predicted:.1f} ({predicted_change:+.1f}%)")

    print("\n" + "=" * 70)

    # ALWAYS show a line chart for trend predictions
    print(f"\nGenerating {prediction_years}-year prediction chart...")

    # Create enhanced data for the prediction chart
    enhanced_country_data = country_data.copy()

    # Add prediction data to the country_data for the chart
    if "predictions" in predictions and isinstance(predictions["predictions"], dict):
        for country, future_values in predictions["predictions"].items():
            if country in enhanced_country_data:
                enhanced_country_data[country]['future_values'] = future_values
                enhanced_country_data[country]['future_years'] = list(range(
                    max(enhanced_country_data[country]['years']) + 1,
                    max(enhanced_country_data[country]['years']) + 1 + len(future_values)
                ))

    # Plot the trend prediction chart
    plot_trend_prediction_chart(series_name, enhanced_country_data, years, prediction_years)

    # Save predictions
    timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
    filename = f"trend_prediction_{prediction_years}yr_{timestamp}.txt"

    with open(filename, 'w') as f:
        f.write(f"{prediction_years}-Year Trend Prediction - {series_name}\n")
        f.write("=" * 50 + "\n\n")
        if "predictions" in predictions:
            for country, future_values in predictions["predictions"].items():
                f.write(f"{country} predictions: {future_values}\n")
        else:
            f.write("No valid predictions generated\n")

    print(f"Prediction data saved as: {filename}")

# Plot selected visualization
def plot_trend(series_name, country_names, year_columns, chart_type='line'):

    global df

    try:
        df_cleaned = pd.read_csv('Data_Cleaned.csv')
    except FileNotFoundError:
        print("Cleaned dataset not found. Using original dataset.")
        df_cleaned = df

    # Convert year columns to numeric for plotting
    years = [int(year) for year in year_columns if str(year).replace(' ', '').isdigit()]

    # Collect data for all countries
    country_data = {}

    for country in country_names:
        data = df_cleaned[(df_cleaned[series_column] == series_name) & (df_cleaned[countries_column] == country)]
        if not data.empty:
            values = data[year_columns].iloc[0].values
            numeric_values = pd.to_numeric(values, errors='coerce')

            valid_indices = ~np.isnan(numeric_values)
            valid_years = np.array(years)[valid_indices]  # Now using integer years
            valid_values = numeric_values[valid_indices]

            if len(valid_values) > 0:
                country_data[country] = {
                    'years': valid_years.tolist(),  # Store as list of integers
                    'values': valid_values.tolist(),
                    'latest_value': float(valid_values[-1]) if len(valid_values) > 0 else None,
                    'first_value': float(valid_values[0]) if len(valid_values) > 0 else None
                }
            else:
                print(f"No valid data for {country}")
        else:
            print(f"No data found for {country}")

    if not country_data:
        print("No valid data found for any selected country.")
        return None

    chart_functions = {
        'line': plot_line_chart,
        'bar': plot_bar_chart,
        'heatmap': plot_heatmap
    }

    if chart_type in chart_functions:
        chart_functions[chart_type](series_name, country_data, years)
    else:
        print(f"Unknown chart type: {chart_type}")

    return country_data


def main():
    if not load_dataset():
        return

    print(f'Dataset: {len(series_list)} series, {len(countries_list)} countries, {len(year_columns)} years')

    if "Data_Cleaned.csv" not in os.listdir('.'):
        try:
            high_missing_data = missing_values_in_rows(df)
            df_clean = clean_dataset(high_missing_data, df)
        except Exception as e:
            print(f"Error during data cleaning: {e}")
            print("Continuing with original dataset...")

    while True:
        series, countries, years, chart_type = manual_selection_interface()

        if series is None:
            return

        print(f"\n" + "=" * 60)
        print("GENERATING VISUALIZATION...")
        print(f"Series: {series}")
        print(f"Countries: {', '.join(countries)}")
        print(f"Years: {len(years)} selected")
        print(f"Chart Type: {chart_type}")
        print("=" * 60)

        country_data = plot_trend(series, countries, years, chart_type)

        if country_data is None:
            continue

        analysis_choice, prediction_years = analysis_selection_interface(series, countries, years)

        if analysis_choice == '1':
            generate_insight_report(series, country_data, years)
        elif analysis_choice == '2':
            generate_trend_prediction(series, country_data, years, prediction_years)
        else:
            print("\nChart display completed.")

        again = input("\nWould you like to analyze another series? (y/n): ").strip().lower()
        if again not in ['y', 'yes']:
            print("Goodbye!")
            break


if __name__ == '__main__':
    main()