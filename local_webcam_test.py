import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
import pandas as pd
import os

# --- CONFIGURATION ---
OUTPUT_MODEL_FILE = 'asl_keypoint_model.pkl'
VIDEO_SOURCE = 0 # Default webcam. Change to 1 or 2 if needed.

# --- 1. Load the Model and Classes ---
try:
    with open(OUTPUT_MODEL_FILE, 'rb') as file:
        classifier = pickle.load(file)
    print(f"✅ Classifier loaded successfully from {OUTPUT_MODEL_FILE}")
except FileNotFoundError:
    print(f"❌ ERROR: Model file '{OUTPUT_MODEL_FILE}' not found. Run train_classifier.py first.")
    exit()

# Extract class names (labels) from the classifier model
# The classes are stored in the model's structure
class_names = list(classifier.classes_)
print(f"Loaded {len(class_names)} ASL classes.")


# --- 2. Initialize MediaPipe Hands ---
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False, # Use video mode for higher tracking speed
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils


# --- 3. Keypoint Normalization Function (Same as in data_collector.py) ---
def process_landmarks_for_prediction(hand_landmarks):
    flat_coords = []
    for landmark in hand_landmarks.landmark:
        flat_coords.extend([landmark.x, landmark.y, landmark.z])
    
    coords = np.array(flat_coords).reshape(-1, 3)

    # Translation Normalization (Shift origin to wrist/root)
    normalized_coords = coords - coords[0]
    
    # Scale Normalization
    scale_factor = np.linalg.norm(normalized_coords[9]) 
    if scale_factor == 0: return None
    
    normalized_coords /= scale_factor
    
    # Return as a pandas DataFrame row for classifier input
    return pd.DataFrame([normalized_coords.flatten().tolist()])


# --- 4. Main Webcam Loop ---

cap = cv2.VideoCapture(VIDEO_SOURCE)
if not cap.isOpened():
    print(f"❌ ERROR: Could not open webcam source {VIDEO_SOURCE}. Please check device index.")
    exit()

print("\nPress 'q' to exit the application. Start signing!")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        continue
    
    # Flip the image horizontally for a mirror view
    frame = cv2.flip(frame, 1)
    
    # Process the frame
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)
    
    prediction_text = "--- NO HAND DETECTED ---"
    
    # --- Classification Logic ---
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            
            # 1. Draw the Keypoints and connections on the hand (for visualization)
            mp_drawing.draw_landmarks(
                frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # 2. Extract and Normalize Keypoints
            normalized_features = process_landmarks_for_prediction(hand_landmarks)
            
            if normalized_features is not None:
                # 3. Predict the sign using the trained model
                
                # Predict the class index (0-25)
                predicted_class_index = classifier.predict(normalized_features)[0]
                
                # Get the class label (A, B, C, ...)
                predicted_label = predicted_class_index
                
                # Get prediction probabilities for confidence score
                probabilities = classifier.predict_proba(normalized_features)[0]
                confidence = np.max(probabilities)

                prediction_text = f"SIGN: {predicted_label} ({confidence:.2f})"
            
            # --- Display Prediction Text on Screen ---
            cv2.rectangle(frame, (10, 10), (450, 60), (0, 0, 0), -1) # Black background
            cv2.putText(frame, prediction_text, (20, 45), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA) # White text

    # Show the final output frame
    cv2.imshow('ASL Keypoint Classifier - 99% Accuracy!', frame)
    
    # Exit condition
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

# Cleanup
hands.close()
cap.release()
cv2.destroyAllWindows()