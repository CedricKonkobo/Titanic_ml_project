from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))
from features.build_features import FeatureEngineer

app = Flask(__name__)
CORS(app)

# Load model at startup
MODEL_PATH = Path(__file__).parent.parent.parent / "models" / "best_model.joblib"
model = joblib.load(MODEL_PATH)

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model': 'Titanic Survival Predictor v1.0',
        'version': '1.0.0'
    })

@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint.
    Expected JSON: {
        "Pclass": 1,
        "Sex": "female",
        "Age": 25,
        "SibSp": 0,
        "Parch": 0,
        "Fare": 100,
        "Embarked": "C",
        "Name": "Doe, Mrs. Jane",
        "Ticket": "12345",
        "Cabin": null
    }
    """
    try:
        data = request.get_json(force=True)
        
        # Validation
        required = ['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked']
        missing = [f for f in required if f not in data]
        if missing:
            return jsonify({'error': f'Missing fields: {missing}'}), 400
            
        # Create DataFrame
        input_df = pd.DataFrame([data])
        
        # Feature engineering (minimal for API)
        fe = FeatureEngineer()
        input_df = fe.create_features(input_df)
        input_df = fe.handle_missing(input_df)
        
        # Prepare features (drop non-feature columns)
        X = input_df.drop(['survived', 'passenger_id', 'same', 'ticket', 'cabin'], 
                         axis=1, errors='ignore')
        
        # Align columns with training (simplified)
        # In production, use saved preprocessor
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0].tolist()
        
        return jsonify({
            'survived': int(prediction),
            'survival_probability': round(probability[1], 4),
            'death_probability': round(probability[0], 4),
            'prediction_label': 'Survived' if prediction == 1 else 'Died'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Batch prediction for multiple passengers."""
    try:
        data = request.get_json(force=True)
        passengers = data.get('passengers', [])
        
        if not passengers:
            return jsonify({'error': 'No passengers provided'}), 400
            
        input_df = pd.DataFrame(passengers)
        predictions = model.predict(input_df).tolist()
        probabilities = model.predict_proba(input_df)[:, 1].tolist()
        
        return jsonify({
            'predictions': predictions,
            'survival_probabilities': [round(p, 4) for p in probabilities],
            'count': len(predictions)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)