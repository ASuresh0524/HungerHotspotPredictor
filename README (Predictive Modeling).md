# BFW-ML-AI-Projects
Overview
This project develops a machine learning model to predict county-level food insecurity rates based on factors like Farm Bill funding allocations, economic indicators, and climate/crop data. The goal is to identify counties most at risk for high food insecurity in order to guide Farm Bill funding and interventions.

Data
The model uses the following datasets:

farm_bill_allocations.csv: Farm Bill spending amounts by county and program
food_insecurity_by_county.csv: Food insecurity rates by county
county_economic_indicators.csv: County-level economic data like median income, unemployment rate
climate_indicators_by_county.csv: County-level climate and crop yield data
Modeling
The predict_food_insecurity.py script executes the following steps:

Loads and merges the datasets into one DataFrame
Selects key features for modeling
Splits data into training and test sets
Trains a linear regression model
Evaluates model performance on test data
Makes predictions and calculates error metrics
Identifies high error counties to prioritize
Saves model to file for operationalization
Key metrics are coefficient of determination (R-squared) to evaluate model fit and mean squared error to quantify prediction error.

Usage
To run the model:

Copy code

python predict_food_insecurity.py
This will output model evaluation results and a list of high priority counties.

The trained model object is saved to food_insecurity_model.pkl for making predictions on new data.

Next Steps
Possible ways to improve the model:

Try different ML algorithms like random forests
Tune model hyperparameters
Incorporate additional data like demographics
Develop ensemble methods to combine multiple models
