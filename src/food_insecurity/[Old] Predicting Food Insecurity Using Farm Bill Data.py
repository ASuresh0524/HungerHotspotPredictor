# Import libraries
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import pickle

# Load datasets
farm_bill_data = pd.read_csv('farm_bill_allocations.csv')
food_insecurity_data = pd.read_csv('food_insecurity_by_county.csv')
economic_data = pd.read_csv('county_economic_indicators.csv')
climate_data = pd.read_csv('climate_indicators_by_county.csv')
volunteer_data = pd.read_csv('volunteer_availability.csv')

# Merge datasets
data = pd.merge(farm_bill_data, food_insecurity_data, on='County')
data = pd.merge(data, economic_data, on='County')
data = pd.merge(data, climate_data, on='County')

# Select features
features = ['Farm_bill_allocation', 'Median_income', 'Unemployment_rate',
            'Drought_severity', 'Crop_yield', 'Population_density',
            'Poverty_rate', 'Food_desert_score', 'Average_temperature']

X = data[features]
y = data['Food_insecurity_rate']

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test) 

# Train models
lr_model = LinearRegression()
lr_model.fit(X_train_scaled, y_train)

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled, y_train)

# Evaluate models
def evaluate_model(model, X, y, model_name):
    predictions = model.predict(X)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)
    print(f"{model_name} - MSE: {mse:.4f}, R2: {r2:.4f}")
    return predictions

lr_predictions = evaluate_model(lr_model, X_test_scaled, y_test, "Linear Regression")
rf_predictions = evaluate_model(rf_model, X_test_scaled, y_test, "Random Forest")

# Feature importance (for Random Forest)
feature_importance = pd.DataFrame({
    'feature': features,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("Feature Importance:")
print(feature_importance)

# Spatial Analysis
# Assuming we have a GeoDataFrame with county geometries
counties_gdf = gpd.read_file('county_boundaries.geojson')
data_with_predictions = pd.concat([data, pd.Series(rf_predictions, name='Predicted_insecurity')], axis=1)
counties_with_data = counties_gdf.merge(data_with_predictions, on='County')
plt.savefig('spatial_analysis.png')

# Plot food insecurity map
fig, ax = plt.subplots(figsize=(15, 10))
counties_with_data.plot(column='Food_insecurity_rate', cmap='YlOrRd', linewidth=0.8, edgecolor='0.8', ax=ax, legend=True)
ax.set_title('Food Insecurity Rate by County')
plt.savefig('food_insecurity_map.png')

# Identify high-need areas
high_need_counties = counties_with_data[counties_with_data['Food_insecurity_rate'] > counties_with_data['Food_insecurity_rate'].quantile(0.9)]
print("High-need counties:")
print(high_need_counties[['County', 'Food_insecurity_rate']])

# Demand Forecasting
# Simple time series forecasting (assuming we have historical data)
from statsmodels.tsa.arima.model import ARIMA

# Assuming 'Date' column exists in the data
data['Date'] = pd.to_datetime(data['Date'])
data.set_index('Date', inplace=True)

# Forecast food insecurity for the next 12 months for a specific county
county_data = data[data['County'] == 'Example County']['Food_insecurity_rate']
model = ARIMA(county_data, order=(1,1,1))
results = model.fit()
forecast = results.forecast(steps=12)
print("12-month food insecurity forecast for Example County:")
print(forecast)

# Volunteer Coordination
# Simple volunteer availability prediction
from sklearn.linear_model import LogisticRegression

X_volunteer = volunteer_data[['Day_of_week', 'Is_weekend', 'Is_holiday']]
y_volunteer = volunteer_data['Is_available']

volunteer_model = LogisticRegression()
volunteer_model.fit(X_volunteer, y_volunteer)

# Predict volunteer availability for next week
next_week = pd.DataFrame({
    'Day_of_week': range(7),
    'Is_weekend': [0, 0, 0, 0, 0, 1, 1],
    'Is_holiday': [0, 0, 0, 0, 0, 0, 0]
})
volunteer_predictions = volunteer_model.predict_proba(next_week)[:, 1]
print("Predicted volunteer availability for next week:")
print(pd.DataFrame({'Day': ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
                    'Availability_prob': volunteer_predictions}))

# Save models
with open('food_insecurity_rf_model.pkl', 'wb') as file:
    pickle.dump(rf_model, file)

with open('volunteer_availability_model.pkl', 'wb') as file:
    pickle.dump(volunteer_model, file)

print("Models saved successfully.")
