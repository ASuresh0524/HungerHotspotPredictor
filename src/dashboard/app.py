from flask import Flask, render_template, jsonify, request
import pandas as pd
import json
from datetime import datetime
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from food_insecurity.predict_food_insecurity import FoodInsecurityPredictor

app = Flask(__name__)
predictor = FoodInsecurityPredictor()

@app.route('/')
def index():
    """Render main dashboard page"""
    return render_template('index.html')

@app.route('/api/high-risk-counties')
def high_risk_counties():
    """Get high risk counties data"""
    try:
        # Load and prepare data
        data = predictor.load_data()
        X, y, _ = predictor.prepare_features(data)
        
        # Train model if not already trained
        if predictor.model is None:
            predictor.train_model(X, y)
        
        # Get high risk counties
        high_risk = predictor.identify_high_risk_counties(data, X)
        
        # Convert to dictionary for JSON response
        result = high_risk.to_dict(orient='records')
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/county-stats')
def county_stats():
    """Get general statistics about counties"""
    try:
        data = predictor.load_data()
        stats = {
            'total_counties': len(data),
            'states_covered': len(data['state'].unique()),
            'avg_food_insecurity': data['food_insecurity_rate'].mean(),
            'highest_risk_state': data.groupby('state')['food_insecurity_rate'].mean().idxmax()
        }
        return jsonify({'success': True, 'data': stats})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

@app.route('/api/funding-impact')
def funding_impact():
    """Analyze impact of funding on food insecurity"""
    try:
        data = predictor.load_data()
        
        # Calculate correlation between funding and food insecurity
        funding_impact = {
            'snap_correlation': data['snap_allocation'].corr(data['food_insecurity_rate']),
            'wic_correlation': data['wic_allocation'].corr(data['food_insecurity_rate']),
            'rural_dev_correlation': data['rural_development'].corr(data['food_insecurity_rate'])
        }
        
        # Get top funded counties
        top_funded = data.nlargest(5, 'snap_allocation')[
            ['county_name', 'state', 'snap_allocation', 'food_insecurity_rate']
        ].to_dict(orient='records')
        
        return jsonify({
            'success': True,
            'correlations': funding_impact,
            'top_funded_counties': top_funded
        })
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True) 