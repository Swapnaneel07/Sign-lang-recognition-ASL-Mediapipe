import pickle
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
import os

# --- 1. CONFIGURATION ---
MODEL_FILE = 'asl_keypoint_model.pkl'

# --- 2. LOAD MODEL ---
if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(f"Model file not found: {MODEL_FILE}")

with open(MODEL_FILE, 'rb') as f:
    clf = pickle.load(f)

# Classes from your training script (ensure this matches the order used during training)
CLASSES = clf.classes_ # Directly pull classes from the loaded sklearn object

app = Flask(__name__)
# Enable CORS for the frontend (index.html) running in the browser to talk to this server
CORS(app) 

# --- 3. API ENDPOINT ---

@app.route('/predict', methods=['POST'])
def predict():
    """
    Receives normalized keypoint features (63 elements) from the frontend,
    runs prediction, and returns the predicted class (letter) and confidence.
    """
    try:
        data = request.json
        features = np.array(data['features'], dtype=np.float32).reshape(1, -1)
        
        # 1. Get class probabilities for confidence calculation
        probas = clf.predict_proba(features)[0]
        
        # 2. Get the final predicted class index and value
        pred_index = np.argmax(probas)
        prediction = CLASSES[pred_index]
        confidence = probas[pred_index]
        
        return jsonify({
            'prediction': prediction,
            'confidence': round(confidence * 100, 2), # Return as percentage
            'success': True
        })
    
    except Exception as e:
        print(f"Prediction Error: {e}")
        return jsonify({'error': str(e), 'success': False}), 500

# We don't use app.run() here because Render uses gunicorn (Start Command)
# But we keep this structure for completeness.