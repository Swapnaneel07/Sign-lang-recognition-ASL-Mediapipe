import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import pickle
import os

# --- CONFIGURATION ---
INPUT_CSV = 'asl_keypoint_data.csv'
OUTPUT_MODEL_FILE = 'asl_keypoint_model.pkl'

def train_model():
    if not os.path.exists(INPUT_CSV):
        print(f"Error: Input file '{INPUT_CSV}' not found. Run data_collector.py first.")
        return

    # 1. Load Data
    df = pd.read_csv(INPUT_CSV)
    
    # Drop samples where MediaPipe failed to extract keypoints (optional, but clean)
    df.dropna(inplace=True) 

    X = df.drop('label', axis=1) # Features (63 coordinates)
    y = df['label']             # Target (A, B, C, etc.)

    # 2. Split Data
    # The synthetic data is pre-split, but we'll use a local validation split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    print(f"Training on {len(X_train)} samples across {len(y.unique())} classes...")

    # 3. Initialize and Train the Classifier (Fast and effective)
    model = RandomForestClassifier(n_estimators=100, max_depth=15, min_samples_leaf=1, random_state=42, verbose=1, n_jobs=-1)
    model.fit(X_train, y_train)

    # 4. Evaluate
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\nTraining Complete.")
    print(f"Validation Accuracy: {accuracy * 100:.2f}%")

    # --- Create reports directory and save evaluation artifacts ---
    reports_dir = 'reports'
    os.makedirs(reports_dir, exist_ok=True)

    # 1) Text report with accuracy and classification report
    cls_report = classification_report(y_test, y_pred)
    report_text = (
        f"Validation Accuracy: {accuracy * 100:.2f}%\n\n"
        f"Classification Report:\n{cls_report}\n"
    )
    report_path = os.path.join(reports_dir, 'evaluation_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report_text)

    # 2) Confusion matrix image
    cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, cmap='Blues', colorbar=True)
    plt.title('Confusion Matrix')
    cm_path = os.path.join(reports_dir, 'confusion_matrix.png')
    fig.savefig(cm_path, bbox_inches='tight')
    plt.close(fig)

    # 3) Save raw predictions for further analysis
    preds_df = X_test.copy()
    preds_df['y_true'] = y_test.values
    preds_df['y_pred'] = y_pred
    preds_csv = os.path.join(reports_dir, 'predictions.csv')
    preds_df.to_csv(preds_csv, index=False)

    print(f"Saved evaluation report to {report_path}")
    print(f"Saved confusion matrix to {cm_path}")
    print(f"Saved predictions CSV to {preds_csv}")

    # 5. Save the Model
    with open(OUTPUT_MODEL_FILE, 'wb') as file:
        pickle.dump(model, file)
    
    print(f"✅ Classification model saved to {OUTPUT_MODEL_FILE}")


if __name__ == '__main__':
    train_model()