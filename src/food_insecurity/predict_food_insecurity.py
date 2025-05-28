import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FoodInsecurityPredictor:
    def __init__(self, data_dir='data'):
        self.data_dir = data_dir
        self.model = None
        self.scaler = StandardScaler()
        
    def load_data(self):
        """Load and merge all relevant datasets"""
        try:
            # Load individual datasets
            farm_bill = pd.read_csv(f'{self.data_dir}/raw/farm_bill_allocations.csv')
            food_insecurity = pd.read_csv(f'{self.data_dir}/raw/food_insecurity_by_county.csv')
            economic = pd.read_csv(f'{self.data_dir}/raw/county_economic_indicators.csv')
            climate = pd.read_csv(f'{self.data_dir}/raw/climate_indicators_by_county.csv')
            
            # Merge all datasets on county_fips and year
            merged_data = farm_bill.merge(food_insecurity, on=['county_fips', 'year'])
            merged_data = merged_data.merge(economic, on=['county_fips', 'year'])
            merged_data = merged_data.merge(climate, on=['county_fips', 'year'])
            
            logger.info(f"Successfully loaded and merged {len(merged_data)} county records")
            return merged_data
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def prepare_features(self, data):
        """Prepare feature matrix for modeling"""
        feature_columns = [
            'snap_allocation', 'wic_allocation', 'rural_development', 'conservation_programs',
            'median_household_income', 'unemployment_rate', 'poverty_rate', 'cost_of_living_index',
            'annual_rainfall_inches', 'drought_severity_index', 'crop_yield_index', 'natural_disaster_count'
        ]
        
        X = data[feature_columns]
        y = data['food_insecurity_rate']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        return X_scaled, y, feature_columns
    
    def train_model(self, X, y):
        """Train the random forest model"""
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.model.fit(X, y)
        logger.info("Model training completed")
    
    def evaluate_model(self, X, y):
        """Evaluate model performance"""
        predictions = self.model.predict(X)
        mse = mean_squared_error(y, predictions)
        r2 = r2_score(y, predictions)
        
        logger.info(f"Model Performance Metrics:")
        logger.info(f"Mean Squared Error: {mse:.4f}")
        logger.info(f"R-squared Score: {r2:.4f}")
        
        return mse, r2
    
    def identify_high_risk_counties(self, data, X, threshold_percentile=90):
        """Identify counties with highest predicted food insecurity rates"""
        predictions = self.model.predict(X)
        
        # Add predictions to dataframe
        results_df = data[['county_fips', 'county_name', 'state']].copy()
        results_df['predicted_food_insecurity_rate'] = predictions
        results_df['actual_food_insecurity_rate'] = data['food_insecurity_rate']
        results_df['prediction_error'] = abs(results_df['predicted_food_insecurity_rate'] - 
                                          results_df['actual_food_insecurity_rate'])
        
        # Identify high-risk counties
        threshold = np.percentile(predictions, threshold_percentile)
        high_risk_counties = results_df[results_df['predicted_food_insecurity_rate'] >= threshold]
        
        logger.info(f"\nHigh Risk Counties (above {threshold_percentile}th percentile):")
        for _, county in high_risk_counties.iterrows():
            logger.info(f"{county['county_name']}, {county['state']}: {county['predicted_food_insecurity_rate']:.2f}%")
        
        return high_risk_counties
    
    def save_model(self, output_dir='models'):
        """Save the trained model and scaler"""
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        joblib.dump(self.model, f'{output_dir}/food_insecurity_model.pkl')
        joblib.dump(self.scaler, f'{output_dir}/scaler.pkl')
        logger.info(f"Model and scaler saved to {output_dir}")

def main():
    # Initialize predictor
    predictor = FoodInsecurityPredictor()
    
    # Load and prepare data
    data = predictor.load_data()
    X, y, feature_columns = predictor.prepare_features(data)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    predictor.train_model(X_train, y_train)
    
    # Evaluate model
    train_mse, train_r2 = predictor.evaluate_model(X_train, y_train)
    test_mse, test_r2 = predictor.evaluate_model(X_test, y_test)
    
    # Identify high-risk counties
    high_risk_counties = predictor.identify_high_risk_counties(data, X)
    
    # Save model
    predictor.save_model()

if __name__ == "__main__":
    main() 