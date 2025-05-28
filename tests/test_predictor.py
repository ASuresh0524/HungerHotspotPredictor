import pytest
import pandas as pd
import numpy as np
from src.food_insecurity.predict_food_insecurity import FoodInsecurityPredictor

@pytest.fixture
def sample_data():
    """Create sample data for testing"""
    data = {
        'county_fips': ['06037', '36061'],
        'county_name': ['Los Angeles', 'New York'],
        'state': ['CA', 'NY'],
        'snap_allocation': [1250000000, 980000000],
        'wic_allocation': [180000000, 150000000],
        'rural_development': [25000000, 12000000],
        'conservation_programs': [15000000, 8000000],
        'food_insecurity_rate': [14.2, 12.8],
        'median_household_income': [72150, 89620],
        'unemployment_rate': [5.8, 4.9],
        'poverty_rate': [14.2, 12.8],
        'cost_of_living_index': [146.6, 187.2],
        'annual_rainfall_inches': [14.2, 46.8],
        'drought_severity_index': [-2.1, 1.2],
        'crop_yield_index': [82.5, 88.9],
        'natural_disaster_count': [3, 2],
        'year': [2023, 2023]
    }
    return pd.DataFrame(data)

def test_prepare_features(sample_data):
    """Test feature preparation"""
    predictor = FoodInsecurityPredictor()
    X, y, feature_columns = predictor.prepare_features(sample_data)
    
    assert X.shape[1] == 12  # Check number of features
    assert len(y) == len(sample_data)
    assert isinstance(X, np.ndarray)
    assert isinstance(y, pd.Series)

def test_train_model(sample_data):
    """Test model training"""
    predictor = FoodInsecurityPredictor()
    X, y, _ = predictor.prepare_features(sample_data)
    predictor.train_model(X, y)
    
    assert predictor.model is not None

def test_evaluate_model(sample_data):
    """Test model evaluation"""
    predictor = FoodInsecurityPredictor()
    X, y, _ = predictor.prepare_features(sample_data)
    predictor.train_model(X, y)
    
    mse, r2 = predictor.evaluate_model(X, y)
    assert isinstance(mse, float)
    assert isinstance(r2, float)
    assert 0 <= r2 <= 1  # R-squared should be between 0 and 1

def test_identify_high_risk_counties(sample_data):
    """Test high-risk county identification"""
    predictor = FoodInsecurityPredictor()
    X, y, _ = predictor.prepare_features(sample_data)
    predictor.train_model(X, y)
    
    high_risk = predictor.identify_high_risk_counties(sample_data, X, threshold_percentile=50)
    assert isinstance(high_risk, pd.DataFrame)
    assert len(high_risk) <= len(sample_data)  # Should return subset of counties 