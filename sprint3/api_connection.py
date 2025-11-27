from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os
import pandas as pd
import matplotlib
import os
from werkzeug.utils import secure_filename

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64

# Add app to Python path
sys.path.append('.')

# Import existing functions
from main import load_dataset, plot_trend, generate_insight_report, generate_trend_prediction, add_file_to_directory

app = Flask(__name__)
CORS(app)

# Global variables
series_list = []
countries_list = []
year_columns = []
df_global = None


# Initialize data on startup
def initialize_data():
    print("Initializing dataset...")
    try:
        # Import the module to access updated globals
        import main

        success = load_dataset()
        if success:
            # Get the updated values from the main module
            global series_list, countries_list, year_columns, df_global
            series_list = main.series_list
            countries_list = main.countries_list
            year_columns = main.year_columns
            df_global = main.df

            print("Dataset loaded successfully")
            print(f"Series: {len(series_list)}, Countries: {len(countries_list)}, Years: {len(year_columns)}")
        else:
            print("Failed to load dataset")
        return success
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return False


# Initialising
initialize_data()


# Check what data is available
@app.route('/api/debug', methods=['GET'])
def debug_info():
    try:
        return jsonify({
            'status': 'success',
            'series_count': len(series_list),
            'countries_count': len(countries_list),
            'years_count': len(year_columns),
            'df_loaded': df_global is not None,
            'series_sample': series_list[:5] if series_list else [],
            'countries_sample': countries_list[:5] if countries_list else []
        })
    except Exception as e:
        return jsonify({'status': 'error', 'message': str(e)})


# Load the dataset
@app.route('/api/load-data', methods=['GET'])
def api_load_data():
    try:
        # Ensure data is loaded
        if not series_list or not countries_list:
            print("📥 Data not loaded, loading now...")
            success = initialize_data()  # Use our initialize function
            if not success:
                return jsonify({"status": "error", "message": "Failed to load dataset"})

        # Convert to format frontend expects
        series_options = [{"name": series, "value": series} for series in series_list[:100]]
        country_options = [{"name": country, "value": country} for country in countries_list[:200]]
        year_options = [{"name": year, "value": year} for year in year_columns]

        return jsonify({
            "status": "success",
            "series": series_options,
            "countries": country_options,
            "years": year_options
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# Generate charts using existing plot_trend function
@app.route('/api/generate-chart', methods=['POST'])
def api_generate_chart():
    try:
        data = request.json
        series = data['series']
        countries = data['countries']
        years = data['years']
        chart_type = data.get('chart_type', 'line')

        print(f"📊 Generating chart: {series}, {countries}, {years}, {chart_type}")

        # Call your existing function
        country_data = plot_trend(series, countries, years, chart_type)

        if country_data is None:
            return jsonify({"status": "error", "message": "No data available for selected criteria"})

        # Convert matplotlib chart to image
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        buf.seek(0)
        image_base64 = base64.b64encode(buf.getvalue()).decode('utf-8')
        plt.close()

        return jsonify({
            "status": "success",
            "chart_image": f"data:image/png;base64,{image_base64}",
            "country_data": country_data
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


# Get AI insights using existing function
@app.route('/api/get-insights', methods=['POST'])
def api_get_insights():
    try:
        data = request.json
        series = data['series']
        country_data = data['country_data']
        years = data['years']

        print(f"🤖 Generating insights for: {series}")

        insights = generate_insight_report(series, country_data, years)

        return jsonify({
            "status": "success",
            "insights": insights
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Check if data files exist
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


# Handle CSV file upload
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

        # Save the uploaded file
        filename = secure_filename(file.filename)
        file.save('Data.csv')

        # Use existing function to process the file
        success = add_file_to_directory('Data.csv')

        if success:
            # Reload the dataset after upload
            initialize_data()
            return jsonify({
                'status': 'success',
                'message': 'File uploaded and processed successfully! Data is now ready for analysis.'
            })
        else:
            return jsonify({'status': 'error', 'message': 'File processing failed'})

    except Exception as e:
        return jsonify({'status': 'error', 'message': f'Upload failed: {str(e)}'})


if __name__ == '__main__':
    print("Connecting to: http://localhost:5000")
    app.run(debug=True, port=5000)