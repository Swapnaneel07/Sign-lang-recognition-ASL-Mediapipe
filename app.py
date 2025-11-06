import pickle
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app) 

# --- Model Loading ---
try:
    with open('asl_keypoint_model.pkl', 'rb') as f:
        model = pickle.load(f)
    print("Model 'asl_keypoint_model.pkl' loaded successfully!")
except FileNotFoundError:
    print("CRITICAL ERROR: 'asl_keypoint_model.pkl' not found. Make sure it's in the same directory.")
    model = None

# --- API Endpoint ---
@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.json
        features = data['features']

        if len(features) != 63:
            return jsonify({"error": "Invalid feature length. Expected 63."}), 400

        prediction = model.predict([features])
        confidence = model.predict_proba([features])
        
        result = {
            "prediction": prediction[0],
            "confidence": np.max(confidence)
        }
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# --- NEW: Serve the index.html file ---
@app.route('/')
def serve_index():
    """Serves the index.html file from the current directory."""
    return send_from_directory('.', 'index.html')

if __name__ == '__main__':
    print("\nModel loaded. Open this URL in your browser:")
    print("http://127.0.0.1:5000")
    print("Press CTRL+C to quit.")
    app.run(port=5000, debug=True)