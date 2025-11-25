import pytest
import pandas as pd
import numpy as np
import os
import sys

# Add the parent directory to Python path to import main.py
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import (
    missing_values_in_rows,
    clean_dataset,
    calculate_model_accuracy,
    get_mathematical_trend_prediction,
    plot_trend
)


# Create a sample DataFrame for testing
class TestDataAnalysis:
    def setup_method(self):
        self.sample_data = pd.DataFrame({
            'Country Name': ['CountryA', 'CountryA', 'CountryB', 'CountryB'],
            'Series Name': ['Series1', 'Series2', 'Series1', 'Series2'],
            '2000': [1.0, 2.0, 3.0, 4.0],
            '2001': [1.5, 2.5, 3.5, 4.5],
            '2002': [2.0, 3.0, 4.0, 5.0],
            '2003': [np.nan, 3.5, np.nan, 5.5],
            '2004': [2.5, 4.0, 4.5, 6.0]
        })

        # Create test data with high missing values
        self.high_missing_data = pd.DataFrame({
            'Country Name': ['CountryC', 'CountryD'],
            'Series Name': ['Series1', 'Series2'],
            '2000': [np.nan, 1.0],
            '2001': [np.nan, 2.0],
            '2002': [np.nan, 3.0],
            '2003': [np.nan, 4.0],
            '2004': [np.nan, 5.0]
        })
    # Test identification of rows with high missing values
    def test_missing_values_in_rows(self):
        # Test with high missing data
        high_missing_indices = missing_values_in_rows(self.high_missing_data, threshold=0.7)
        assert len(high_missing_indices) == 1  # Only CountryC should be flagged
        assert high_missing_indices[0] == 0  # First row (CountryC)

        # Test with normal data
        normal_indices = missing_values_in_rows(self.sample_data, threshold=0.7)
        assert len(normal_indices) == 0  # No rows should be flagged
    # Test dataset cleaning functionality
    def test_clean_dataset(self):
        high_missing_indices = [0]  # Index of CountryC
        cleaned_df = clean_dataset(high_missing_indices, self.high_missing_data)

        # Check that high-missing row was removed
        assert len(cleaned_df) == 1
        assert cleaned_df.iloc[0]['Country Name'] == 'CountryD'

        # Check that data types are numeric
        year_columns = ['2000', '2001', '2002', '2003', '2004']
        for col in year_columns:
            assert pd.api.types.is_numeric_dtype(cleaned_df[col])
    # Test accuracy calculation with train-test split
    def test_calculate_model_accuracy(self):
        # Create sample country data
        country_data = {
            'CountryA': {
                'years': [2000, 2001, 2002, 2003, 2004],
                'values': [1.0, 1.5, 2.0, 2.5, 3.0]
            },
            'CountryB': {
                'years': [2000, 2001, 2002],
                'values': [2.0, 2.2, 2.4]
            }
        }

        accuracy_scores = calculate_model_accuracy(country_data)

        # Check that accuracy scores are calculated
        assert 'CountryA' in accuracy_scores
        assert 'CountryB' in accuracy_scores

        # CountryA should have numeric accuracy, CountryB might have "Insufficient data"
        assert isinstance(accuracy_scores['CountryA'], (int, float)) or accuracy_scores[
            'CountryA'] == "Insufficient data"

    # Test mathematical trend prediction
    def test_mathematical_trend_prediction(self):
        country_data = {
            'CountryA': {
                'years': [2000, 2001, 2002],
                'values': [1.0, 1.5, 2.0]
            }
        }

        predictions = get_mathematical_trend_prediction('Test Series', country_data, 3)

        # Check structure of predictions
        assert 'predictions' in predictions
        assert 'accuracy_scores' in predictions
        assert 'CountryA' in predictions['predictions']

        # Check prediction values
        country_predictions = predictions['predictions']['CountryA']
        assert len(country_predictions) == 3
        assert all(isinstance(val, (int, float)) for val in country_predictions)
    # Test data processing in plot_trend function (without actual plotting)
    def test_plot_trend_data_processing(self):
        # Mock the plotting functions to avoid actually displaying plots
        import matplotlib
        matplotlib.use('Agg')  # Use non-interactive backend

        # Create a temporary cleaned data file for testing
        test_df = pd.DataFrame({
            'Country Name': ['CountryA'],
            'Series Name': ['Series1'],
            '2000': [1.0],
            '2001': [1.5],
            '2002': [2.0]
        })
        test_df.to_csv('Data_Cleaned.csv', index=False)

        try:
            # Test with valid data
            country_data = plot_trend('Series1', ['CountryA'], ['2000', '2001', '2002'], 'line')

            assert country_data is not None
            assert 'CountryA' in country_data
            assert 'values' in country_data['CountryA']
            assert 'years' in country_data['CountryA']
        finally:
            # Clean up
            if os.path.exists('Data_Cleaned.csv'):
                os.remove('Data_Cleaned.csv')
    # Test edge cases and error conditions
    def test_edge_cases(self):
        # Test with empty data
        empty_country_data = {}
        predictions = get_mathematical_trend_prediction('Test Series', empty_country_data, 3)
        assert predictions['predictions'] == {}

        # Test with single data point
        single_point_data = {
            'CountryA': {
                'years': [2000],
                'values': [1.0]
            }
        }
        predictions = get_mathematical_trend_prediction('Test Series', single_point_data, 2)
        assert len(predictions['predictions']['CountryA']) == 2
    # Test that LLM-related functions exist and have correct signatures
    def test_mock_llm_functions(self):

        # These are just signature tests - we don't actually call LLM in tests
        from main import get_llm_trend_prediction, get_llm_insight_report

        # Test that functions exist and are callable
        assert callable(get_llm_trend_prediction)
        assert callable(get_llm_insight_report)


if __name__ == '__main__':
    pytest.main([__file__, "-v"])