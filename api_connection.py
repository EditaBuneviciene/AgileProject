from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import sys
import os
import matplotlib
from werkzeug.utils import secure_filename

matplotlib.use('Agg')

# Add app to Python path
sys.path.append('sprint3')

# Import backend functions
from main import (
    load_dataset, plot_trend, get_llm_insight_report,
    get_llm_trend_prediction, add_file_to_directory
)

# app = Flask(__name__)
app = Flask(__name__, static_folder='web', static_url_path='/web')
CORS(app)

# Globals
series_list = []
countries_list = []
year_columns = []
df_global = None


# Initialize dataset on startup
def initialize_data():
    print("Initializing dataset...")
    try:
        import main

        success = load_dataset()
        if success:
            global series_list, countries_list, year_columns, df_global
            series_list = main.series_list
            countries_list = main.countries_list
            year_columns = main.year_columns   # full dataset years
            df_global = main.df

            print("Dataset loaded successfully")
            print(f"Series: {len(series_list)}, Countries: {len(countries_list)}, Years: {len(year_columns)}")
        else:
            print("Failed to load dataset")
        return success

    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


# Initialize dataset once
initialize_data()


# Debug Info
@app.route('/api/debug', methods=['GET'])
def debug_info():
    try:
        return jsonify({
            'status': 'success',
            'series_count': len(series_list),
            'countries_count': len(countries_list),
            'years_count': len(year_columns),
            'df_loaded': df_global is not None,
            'series_sample': series_list[:5],
            'countries_sample': countries_list[:5]
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# Load dataset info for frontend
@app.route('/api/load-data', methods=['GET'])
def api_load_data():
    try:
        if not series_list or not countries_list:
            print("Data not loaded, reloading...")
            success = initialize_data()
            if not success:
                return jsonify({"status": "error", "message": "Failed to load dataset"})

        # Build dropdown options
        series_options = [{"name": s, "value": s} for s in series_list]
        country_options = [{"name": c, "value": c} for c in countries_list]
        year_options = [{"name": y, "value": y} for y in year_columns]

        return jsonify({
            "status": "success",
            "series": series_options,
            "countries": country_options,
            "years": year_options
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# API to main to generate chart
@app.route('/api/generate-chart', methods=['POST'])
def api_generate_chart():
    try:
        data = request.json
        series = data['series']
        countries = data['countries']
        selected_years = data['years']
        chart_type = data.get('chart_type', 'line')

        include_predictions = data.get('include_predictions', False)
        predictions = data.get('predictions')
        prediction_years = data.get('prediction_years', 0)

        print(f"Generate chart request: {series}, {countries}, {selected_years}, {chart_type}")

        country_data = plot_trend(series, countries, selected_years, chart_type)

        if country_data is None or len(country_data) == 0:
            return jsonify({"status": "error", "message": "No data available for selection"})

        from main import generate_web_chart
        chart_image = generate_web_chart(
            series_name=series,
            country_data=country_data,
            selected_years=selected_years,
            chart_type=chart_type,
            include_predictions=include_predictions,
            predictions=predictions,
            prediction_years=prediction_years
        )

        return jsonify({
            "status": "success",
            "chart_image": f"data:image/png;base64,{chart_image}",
            "country_data": country_data
        })

    except Exception as e:
        print(f"Chart generation error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})



# Generate insight Report
@app.route('/api/get-insights', methods=['POST'])
def api_get_insights():
    try:
        data = request.json
        series = data['series']
        countries = data['countries']
        selected_years = data['years']

        print(f"Generating insights for: {series}, {countries}")

        country_data = plot_trend(series, countries, selected_years, 'line')

        if not country_data:
            return jsonify({"status": "error", "message": "No data available for selected criteria"})

        insights = get_llm_insight_report(series, country_data, selected_years)

        return jsonify({
            "status": "success",
            "insights": insights
        })

    except Exception as e:
        print(f"Insights error: {e}")
        return jsonify({"status": "error", "message": str(e)})


# Trend Predictions

@app.route('/api/get-predictions', methods=['POST'])
def api_get_predictions():
    try:
        data = request.json
        series = data['series']
        countries = data['countries']
        selected_years = data['years']
        prediction_years = int(data['prediction_years'])

        print(f"Generating {prediction_years}-year predictions for: {series}, {countries}")

        country_data = plot_trend(series, countries, selected_years, 'line')

        if not country_data:
            return jsonify({"status": "error", "message": "No data available for selected criteria"})

        predictions = get_llm_trend_prediction(series, country_data, prediction_years)

        return jsonify({
            "status": "success",
            "predictions": predictions
        })

    except Exception as e:
        print(f"Prediction error: {e}")
        import traceback; traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)})


# Check if CSV exists
@app.route('/api/check-data-status', methods=['GET'])
def check_data_status():
    try:
        data_available = os.path.exists('Data_Cleaned.csv') or os.path.exists('Data.csv')

        return jsonify({
            'status': 'success',
            'data_available': data_available,
            'files': {
                'data_csv': os.path.exists('Data.csv'),
                'data_cleaned_csv': os.path.exists('Data_Cleaned.csv')
            }
        })

    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# Upload csv and clean it
@app.route('/api/upload-csv', methods=['POST'])
def upload_csv():
    try:
        if 'file' not in request.files:
            return jsonify({'status': 'error', 'message': 'No file provided'})

        file = request.files['file']
        if file.filename == '':
            return jsonify({'status': 'error', 'message': 'No file selected'})

        if not file.filename.lower().endswith('.csv'):
            return jsonify({'status': 'error', 'message': 'File must be a CSV'})

        filename = secure_filename(file.filename)
        file.save('Data.csv')

        success = add_file_to_directory('Data.csv')

        if success:
            initialize_data()
            return jsonify({
                'status': 'success',
                'message': 'File uploaded & processed successfully.'
            })

        return jsonify({'status': 'error', 'message': 'File processing failed'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Upload failed: {e}'})


# Start API connection
if __name__ == '__main__':
    print("Flask running at http://localhost:5000")
    app.run(debug=True, port=5000)

