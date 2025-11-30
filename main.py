from unicodedata import numeric
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend for web
import matplotlib.pyplot as plt
import numpy as np
import os
import json
from openai import OpenAI
import time
import io
import base64

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


# Scan for CSV files & cleaning
def scan_for_csv():
    all_files = os.listdir()
    csv_files = [file for file in all_files if file.endswith('.csv')]

    if "Data.csv" not in csv_files:
        print("Dataset not found. Please add a file to directory.")
        return None

    # Clean if needed
    if "Data_Cleaned.csv" not in csv_files:
        print("No cleaned data found, cleaning...")
        df = pd.read_csv(
            'Data.csv',
            na_values=['', ' ', 'NULL', 'null', 'N/A', 'n/a', 'NaN', 'nan', 'None',
                       'none', '-', '--', '...', 'Missing', 'missing', 'NA', 'na'],
            keep_default_na=True
        )
        high_missing_data = missing_values_in_rows(df)
        cleaned_df = clean_dataset(high_missing_data, df)
        return cleaned_df

    print("Dataset found")
    return pd.read_csv('Data_Cleaned.csv')


# Load dataset into globals
def load_dataset():
    global df, series_list, countries_list, year_columns

    print("Loading dataset...")
    df = scan_for_csv()

    if df is not None:
        series_list = list(df[series_column].drop_duplicates())
        countries_list = list(df[countries_column].drop_duplicates())
        all_columns = df.columns.tolist()
        year_columns = all_columns[2:]   # full dataset year columns

        print(f"Dataset loaded: {len(series_list)} series, {len(countries_list)} countries, {len(year_columns)} years")
        return True

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

    print(f"Found {len(high_missing_indices)} rows with > {threshold * 100}% missing")
    return high_missing_indices


# Clean dataset
def clean_dataset(high_missing_indices, df):
    print("Cleaning dataset...")

    df_clean = df.drop(high_missing_indices).reset_index(drop=True)
    all_columns = df.columns.tolist()
    year_columns = all_columns[2:]

    df_clean[year_columns] = df_clean[year_columns].apply(pd.to_numeric, errors='coerce')
    df_clean[year_columns] = df_clean[year_columns].ffill(axis=1).bfill(axis=1)

    df_clean.to_csv('Data_Cleaned.csv', index=False)
    print("Clean dataset saved as 'Data_Cleaned.csv'")
    return df_clean


# Setup LLM (Ollama)
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


# Get available LLM model
def get_available_model(client):
    try:
        models = client.models.list()
        model_names = [model.id for model in models]

        model_patterns = [
            "tinyllama", "llama2", "codellama", "mistral", "gemma"
        ]

        for pattern in model_patterns:
            for name in model_names:
                if pattern in name.lower():
                    return name

        return model_names[0] if model_names else None

    except Exception:
        return None


# Calculate accuracy (unchanged)
def calculate_model_accuracy(country_data):
    accuracy_scores = {}

    for country, data in country_data.items():
        if len(data['values']) >= 4:
            x = np.array(data['years'])
            y = np.array(data['values'])

            mask = ~np.isnan(y)
            x_clean = x[mask]
            y_clean = y[mask]

            if len(y_clean) >= 4:
                split_point = int(len(x_clean) * 0.7)
                x_train, y_train = x_clean[:split_point], y_clean[:split_point]
                x_test, y_test = x_clean[split_point:], y_clean[split_point:]

                try:
                    degree = min(1, len(y_train) - 1)
                    z = np.polyfit(x_train, y_train, degree)
                    p = np.poly1d(z)
                    test_predictions = p(x_test)

                    errors = np.abs((test_predictions - y_test) / y_test) * 100
                    accuracy = np.mean(errors < 15) * 100
                    accuracy_scores[country] = accuracy
                except:
                    accuracy_scores[country] = "Calculation failed"
            else:
                accuracy_scores[country] = "Insufficient data"
        else:
            accuracy_scores[country] = "Insufficient data"

    return accuracy_scores


# Generate AI insights (unchanged)
def get_llm_insight_report(series_name, country_data, years):
    client = setup_llm()
    if not client:
        return "LLM not available."

    try:
        model_name = get_available_model(client)
        if not model_name:
            return "No LLM model found."

        data_summary = f"Data for {series_name}:\n"
        for country, data in country_data.items():
            if len(data['values']) > 1:
                first_val = data['values'][0]
                last_val = data['values'][-1]
                change = ((last_val - first_val) / first_val * 100) if first_val != 0 else 0
                trend = "decreasing" if change < 0 else "increasing" if change > 0 else "stable"
                data_summary += f"{country}: {trend} ({first_val:.0f} → {last_val:.0f})\n"

        prompt = f"{data_summary}\nGive 2–3 very short sentences summarizing the trends."

        response = client.chat.completions.create(
            model=model_name,
            messages=[
                {"role": "system", "content": "You give very short simple analysis."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=150
        )

        return response.choices[0].message.content.strip()

    except Exception as e:
        return f"Insight failed: {e}"


# FULL HISTORICAL trend predictions
def get_llm_trend_prediction(series_name, country_data, prediction_years):
    # function body unchanged
    return get_mathematical_trend_prediction(series_name, country_data, prediction_years)


# Mathematical fallback prediction
def get_mathematical_trend_prediction(series_name, country_data, prediction_years):
    predictions = {"predictions": {}}
    accuracy_scores = calculate_model_accuracy(country_data)

    for country, data in country_data.items():
        if len(data['values']) >= 2:
            x = np.array(data['years'])
            y = np.array(data['values'])

            mask = ~np.isnan(y)
            x_clean, y_clean = x[mask], y[mask]

            if len(y_clean) >= 2:
                degree = min(2, len(y_clean) - 1)
                try:
                    z = np.polyfit(x_clean, y_clean, degree)
                    p = np.poly1d(z)
                    last_year = max(x_clean)

                    future = [float(round(p(last_year + i), 3)) for i in range(1, prediction_years + 1)]
                    predictions["predictions"][country] = future
                except:
                    slope = (y_clean[-1] - y_clean[0]) / (x_clean[-1] - x_clean[0])
                    last_val = y_clean[-1]
                    predictions["predictions"][country] = [float(round(last_val + slope * i, 3)) for i in range(1, prediction_years + 1)]
            else:
                last_val = data['values'][-1]
                predictions["predictions"][country] = [round(last_val, 3)] * prediction_years
        else:
            last_val = data['values'][-1] if data['values'] else 0
            predictions["predictions"][country] = [round(last_val, 3)] * prediction_years

    predictions["accuracy_scores"] = accuracy_scores
    return predictions



# Extract FULL data to filter by selected years
def get_country_data(series_name, country_names, selected_years):
    global df

    selected_years = [int(y) for y in selected_years]

    try:
        df_cleaned = pd.read_csv('Data_Cleaned.csv')
        current_df = df_cleaned
    except:
        current_df = df

    # Full year columns
    all_columns = current_df.columns.tolist()
    dataset_years = [int(y) for y in all_columns[2:]]

    country_data = {}

    for country in country_names:

        series_filter = current_df['Series Name'] == series_name
        country_filter = current_df['Country Name'] == country
        data = current_df[series_filter & country_filter]

        if data.empty:
            continue

        values = pd.to_numeric(data[all_columns[2:]].iloc[0].values, errors='coerce')
        valid_mask = ~np.isnan(values)

        full_years = np.array(dataset_years)[valid_mask]
        full_vals = values[valid_mask]

        # Filter by selected years
        filtered_years, filtered_vals = [], []
        for y, v in zip(full_years, full_vals):
            if y in selected_years:
                filtered_years.append(y)
                filtered_vals.append(v)

        if len(filtered_vals) > 0:
            country_data[country] = {
                'years': [int(y) for y in filtered_years],
                'values': [float(v) for v in filtered_vals],
                'latest_value': float(filtered_vals[-1]),
                'first_value': float(filtered_vals[0])
            }

    return country_data if country_data else None


# Generate web chart in the HTML
def generate_web_chart(series_name, country_data, selected_years, chart_type='line',
                       include_predictions=False, predictions=None, prediction_years=0):

    plt.figure(figsize=(12, 8))

    # Convert years to int
    selected_years = sorted([int(y) for y in selected_years])

    if chart_type == 'line':
        for country, data in country_data.items():
            plt.plot(data['years'], data['values'], marker='o',
                     label=f'{country} (Historical)', linewidth=2)

            if include_predictions and predictions and country in predictions["predictions"]:
                last_year = max(data['years'])
                future_vals = predictions["predictions"][country]
                future_years = list(range(last_year + 1, last_year + 1 + len(future_vals)))

                plt.plot(future_years, future_vals, '--', marker='x',
                         label=f'{country} (Predicted)', linewidth=2)

        last_hist = max(selected_years)
        if include_predictions:
            plt.axvline(x=last_hist, color='red', linestyle=':', label='Prediction Start')

    elif chart_type == 'bar':
        countries = list(country_data.keys())
        latest_vals = [country_data[c]['latest_value'] for c in countries]
        bars = plt.bar(countries, latest_vals)

        for bar, val in zip(bars, latest_vals):
            plt.text(bar.get_x() + bar.get_width() / 2, val, f'{val:.1f}',
                     ha='center', va='bottom')

    elif chart_type == 'heatmap':
        if len(country_data) >= 2:
            heatmap = []
            country_labels = []

            for country, data in country_data.items():
                row = []
                for year in selected_years:
                    if year in data['years']:
                        idx = data['years'].index(year)
                        row.append(data['values'][idx])
                    else:
                        row.append(np.nan)
                heatmap.append(row)
                country_labels.append(country)

            heatmap = np.array(heatmap)
            im = plt.imshow(heatmap, cmap='YlOrRd', aspect='auto')
            plt.colorbar(im)

            for i in range(len(country_labels)):
                for j in range(len(selected_years)):
                    val = heatmap[i, j]
                    if not np.isnan(val):
                        plt.text(j, i, f'{val:.0f}', ha='center', va='center')

            plt.yticks(range(len(country_labels)), country_labels)
            plt.xticks(range(len(selected_years)), selected_years, rotation=45)

    plt.title(f'{series_name}\nTrend Comparison')
    plt.xlabel('Year')
    plt.ylabel(series_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(alpha=0.3)
    plt.tight_layout()

    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=100)
    buf.seek(0)
    img64 = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close()

    return img64


# Web version wrapper
def plot_trend(series_name, countries, selected_years, chart_type='line'):
    return get_country_data(series_name, countries, selected_years)


def plot_trend_cli(series, countries, years, chart_type='line'):
    pass


# Main
def main():
    if not load_dataset():
        return
    print("Dataset loaded.")

    print("Web mode enabled — CLI disabled.")


if __name__ == '__main__':
    main()
