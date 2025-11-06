import pickle
import json
import os

# --- CONFIGURATION ---
INPUT_MODEL_FILE = 'asl_keypoint_model.pkl'
OUTPUT_JSON_FILE = 'asl_classifier_model.json'

def export_model():
    if not os.path.exists(INPUT_MODEL_FILE):
        print(f"Error: Input model '{INPUT_MODEL_FILE}' not found. Run train_classifier.py first.")
        return

    print(f"Loading model from {INPUT_MODEL_FILE}...")
    with open(INPUT_MODEL_FILE, 'rb') as file:
        classifier = pickle.load(file)

    # We are using a Random Forest (an ensemble of Decision Trees)
    trees = []
    
    # Extract the classes (labels) for deployment
    classes = list(classifier.classes_)

    # Iterate through every decision tree in the forest
    for estimator in classifier.estimators_:
        tree = estimator.tree_
        
        # We need to extract the structure (children, features, thresholds)
        # into a format JavaScript can easily parse.
        tree_structure = {
            'children_left': tree.children_left.tolist(),
            'children_right': tree.children_right.tolist(),
            'feature': tree.feature.tolist(),
            'threshold': tree.threshold.tolist(),
            'value': tree.value.tolist() # The class counts at each leaf node
        }
        trees.append(tree_structure)

    # Package the final data structure
    model_json = {
        'classes': classes,
        'n_features': classifier.n_features_in_,
        'n_classes': len(classes),
        'trees': trees
    }

    # Save the JSON file
    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(model_json, f)

    print(f"✅ Success! Model exported to {OUTPUT_JSON_FILE}. Ready for deployment.")

if __name__ == '__main__':
    export_model()