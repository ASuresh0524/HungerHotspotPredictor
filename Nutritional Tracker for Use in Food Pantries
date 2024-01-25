# Import XGBoost library
import xgboost as xgb 

# Load nutrition data CSV into pandas DataFrame
nutrition_data = pd.read_csv('nutrition_data.csv')

# Separate features from labels 
X = nutrition_data.drop(['nutrition_label'], axis=1) 
y = nutrition_data['nutrition_label']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42) 

# Define XGBoost parameters
params = {'eta': 0.1, 
          'max_depth': 3,
          'objective': 'multi:softmax',  
          'num_class': 10}

# Initialize XGBoost classifier 
nutrition_model = xgb.XGBClassifier(**params)

# Train nutrition model on training data
nutrition_model.fit(X_train, y_train)  

# Make predictions on validation data
val_predictions = nutrition_model.predict(X_val)

# Calculate and print accuracy
accuracy = float(np.sum(val_predictions==y_val)) / len(y_val)
print(f'Nutrition Model Accuracy: {accuracy:.2%}') 

# Save trained model to file
nutrition_model.save_model('nutrition_tracker.model')
