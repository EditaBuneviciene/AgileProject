from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import sys
import os
import pandas as pd
import matplotlib

matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import io
import base64

# Add app to Python path
sys.path.append('.')

# Import existing functions
from main import load_dataset, plot_trend, generate_insight_report, generate_trend_prediction

app = Flask(__name__)
CORS(app)

# Global variables to match the app
df = None
series_list = []
countries_list = []
year_columns = []

# Load dataset
@app.route('/api/load-data', methods=['GET'])
def api_load_data():
    try:
        global df, series_list, countries_list, year_columns

        success = load_dataset()

        if success:
            # Get available series and countries
            series_options = [{"name": series, "value": series} for series in series_list[:50]]  # Limit for dropdown
            country_options = [{"name": country, "value": country} for country in countries_list[:100]]
            year_options = [{"name": year, "value": year} for year in year_columns]

            return jsonify({
                "status": "success",
                "series": series_options,
                "countries": country_options,
                "years": year_options
            })
        else:
            return jsonify({"status": "error", "message": "Failed to load dataset"})

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Generating charts using existing plot_trend function
@app.route('/api/generate-chart', methods=['POST'])
def api_generate_chart():
    try:
        data = request.json
        series = data['series']
        countries = data['countries']
        years = data['years']
        chart_type = data.get('chart_type', 'line')

        # Calling existing function
        country_data = plot_trend(series, countries, years, chart_type)

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

# Get AI insights
@app.route('/api/get-insights', methods=['POST'])
def api_get_insights():
    try:
        data = request.json
        series = data['series']
        country_data = data['country_data']
        years = data['years']

        # Call your existing function
        insights = generate_insight_report(series, country_data, years)

        return jsonify({
            "status": "success",
            "insights": insights
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# Get predictions
@app.route('/api/get-predictions', methods=['POST'])
def api_get_predictions():
    try:
        data = request.json
        series = data['series']
        country_data = data['country_data']
        years = data['years']
        prediction_years = data['prediction_years']

        # Calling existing function
        predictions = generate_trend_prediction(series, country_data, years, prediction_years)

        return jsonify({
            "status": "success",
            "predictions": predictions
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


if __name__ == '__main__':
    print("Connecting to: http://localhost:5000")
    app.run(debug=True, port=5000)