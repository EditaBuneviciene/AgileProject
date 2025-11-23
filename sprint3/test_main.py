import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys
import os

# Import app functions
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import all functions from main
from main import (
    missing_values_in_rows,
    clean_dataset,
    get_mathematical_trend_prediction,
    create_manual_predictions,
    parse_alternative_format
)

# Unit test
class TestDataAnalysis:
# Test missing values detection with sample data
    def test_missing_values_detection(self):
        test_data = {
            'Country Name': ['Country_A', 'Country_B', 'Country_C'],
            'Series Name': ['Series_X', 'Series_X', 'Series_X'],
            '2000': [1.0, None, 3.0],
            '2001': [None, None, 4.0],
            '2002': [3.0, None, 5.0]
        }
        df = pd.DataFrame(test_data)
        result = missing_values_in_rows(df, threshold=0.5)

        # Country_B has 2/3 missing (66%) - should be detected
        assert len(result) == 1
        assert 1 in result
# Test that clean_dataset removes specified rows
    def test_clean_dataset_removes_rows(self):

        test_data = {
            'Country Name': ['A', 'B', 'C'],
            'Series Name': ['X', 'X', 'X'],
            '2000': [1, 2, 3],
            '2001': [4, 5, 6]
        }
        df = pd.DataFrame(test_data)
        cleaned = clean_dataset([1], df)  # Remove index 1
        assert len(cleaned) == 2
        assert 'B' not in cleaned['Country Name'].values
# Test mathematical trend prediction
    def test_mathematical_prediction_basic(self):

        country_data = {
            'TestCountry': {
                'years': [2010, 2011, 2012],
                'values': [100, 120, 140]
            }
        }

        predictions = get_mathematical_trend_prediction(
            'Test Series',
            country_data,
            prediction_years=2
        )
        assert 'predictions' in predictions
        assert 'TestCountry' in predictions['predictions']
        assert len(predictions['predictions']['TestCountry']) == 2

#  Test manual predictions
    def test_manual_predictions_creation(self):

        country_data = {
            'Country_A': {
                'years': [2020],
                'values': [100.5]
            }
        }

        predictions = create_manual_predictions(country_data, 3)
        assert 'predictions' in predictions
        assert 'Country_A' in predictions['predictions']
        assert len(predictions['predictions']['Country_A']) == 3

# Test parsing alternative JSON formats from LLM
    def test_parse_alternative_format(self):
        import json
        alternative_json = [
            {"Country": "Test1", "Number": 0.5},
            {"Country": "Test2", "Number": 0.6}
        ]

        country_data = {
            'Test1': {'years': [2020], 'values': [0.5]},
            'Test2': {'years': [2020], 'values': [0.6]}
        }

        result = parse_alternative_format(json.dumps(alternative_json), country_data, 3)
        assert 'predictions' in result
        assert 'Test1' in result['predictions']

    @patch('main.OpenAI')
    @patch('main.get_available_model')

# Test LLM insight report generation with mock data
    def test_llm_insight_report(self, mock_get_model, mock_openai):

        mock_get_model.return_value = "test-model"
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Test insight analysis"
        mock_client.chat.completions.create.return_value = mock_response
        mock_openai.return_value = mock_client

        from main import get_llm_insight_report
        country_data = {
            'Country_A': {
                'years': [2020, 2021],
                'values': [100, 120]
            }
        }

        insight = get_llm_insight_report('Test Series', country_data, [2020, 2021])
        assert "Test insight analysis" in insight

if __name__ == "__main__":
    pytest.main([__file__, "-v"])