import cv2
import mediapipe as mp
import pandas as pd
import numpy as np
import os
import glob
import time

# --- CONFIGURATION ---
# Path to the downloaded and unzipped Kaggle training data
# NOTE: Ensure this path is correct based on your local download location
DATASET_ROOT_DIR = r'E:\data_raw\Train_Alphabet' 
OUTPUT_CSV_FILE = 'asl_keypoint_data.csv'

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.7)

# --- UTILITY: Normalize Keypoints ---
# This is crucial: it standardizes hand size and position (rotation/translation)
# so the model only learns finger shape, not where the hand is on screen.
def process_landmarks(hand_landmarks):
    # Extract coordinates into a single flat list
    flat_coords = []
    
    # Get all 21 keypoints
    for landmark in hand_landmarks.landmark:
        flat_coords.extend([landmark.x, landmark.y, landmark.z])
    
    # Convert to numpy array
    coords = np.array(flat_coords).reshape(-1, 3) # 21x3 array

    # 1. Translation Normalization (Shift origin to wrist/root)
    # Subtract wrist coordinates (landmark 0) from all other landmarks
    root_x, root_y, root_z = coords[0]
    normalized_coords = coords - coords[0]
    
    # 2. Scale Normalization
    # Calculate distance from wrist (0) to middle finger base (9) or index finger base (5) 
    # to find a scaling factor.
    scale_factor = np.linalg.norm(normalized_coords[9]) # Using middle finger base as scale reference
    if scale_factor == 0: return None # Avoid division by zero
    
    # Divide all coordinates by the scale factor
    normalized_coords /= scale_factor
    
    # Flatten back into a list (63 values: x1, y1, z1, x2, y2, z2, ...)
    return normalized_coords.flatten().tolist()


# --- MAIN EXTRACTION LOOP ---

def extract_data():
    data_list = []
    
    # Get all subfolders (A, B, C, Blank, etc.)
    all_folders = sorted([d for d in os.listdir(DATASET_ROOT_DIR) if os.path.isdir(os.path.join(DATASET_ROOT_DIR, d))])
    
    print(f"Found {len(all_folders)} folders. Starting keypoint extraction...")
    
    for class_name in all_folders:
        if class_name == 'background': continue # Skip any non-ASL folders
        
        image_dir = os.path.join(DATASET_ROOT_DIR, class_name)
        image_paths = glob.glob(os.path.join(image_dir, '*.png'))
        
        start_time = time.time()
        
        for i, img_path in enumerate(image_paths):
            if i > 0 and i % 500 == 0:
                print(f"  Processed {i}/{len(image_paths)} images for class '{class_name}'.")

            # Load the image
            image = cv2.imread(img_path)
            if image is None: continue

            # Convert BGR image to RGB
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Process the image with MediaPipe
            results = hands.process(image_rgb)
            
            if results.multi_hand_landmarks:
                # Assuming only one hand is present and detected (max_num_hands=1)
                landmarks = results.multi_hand_landmarks[0]
                
                # Get the normalized coordinates (63 values)
                normalized_coords = process_landmarks(landmarks)
                
                if normalized_coords:
                    # Prepend the class name
                    row = [class_name] + normalized_coords
                    data_list.append(row)

        end_time = time.time()
        print(f"Class '{class_name}' finished. Time taken: {end_time - start_time:.2f}s")


    # --- Save to CSV ---
    print("\nExtraction complete. Saving to CSV...")
    
    # Create column headers: 'label', x1, y1, z1, x2, y2, z2, ..., x21, y21, z21
    headers = ['label'] + [f'L{j}_{axis}' for j in range(21) for axis in ['x', 'y', 'z']]
    
    df = pd.DataFrame(data_list, columns=headers)

    # Filter out the 'Blank' samples (these are our true negative samples)
    df_signs = df[df['label'] != 'Blank']
    
    # Save the final classification dataset
    df_signs.to_csv(OUTPUT_CSV_FILE, index=False)
    print(f"✅ Success! Keypoint data saved to {OUTPUT_CSV_FILE} with {len(df_signs)} samples.")


if __name__ == '__main__':
    extract_data()