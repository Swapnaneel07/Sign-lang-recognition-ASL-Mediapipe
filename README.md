# Sign-lang-recognition-ASL-Mediapipe

An accessible, lightweight pipeline for recognizing American Sign Language (ASL) signs from webcam input using MediaPipe for keypoint extraction and a small classifier exported to JSON.

## Table of contents

- [Project purpose](#project-purpose)
- [Repository structure](#repository-structure)
- [High-level methodology](#high-level-methodology)
- [Data collection](#data-collection)
- [Preprocessing & features](#preprocessing--features)
- [Training the classifier](#training-the-classifier)
- [Exporting and using the model](#exporting-and-using-the-model)
- [Run locally (quickstart)](#run-locally-quickstart)
- [Tips for better accuracy](#tips-for-better-accuracy)
- [Evaluation](#evaluation)
- [Troubleshooting](#troubleshooting)
- [Contributing & license](#contributing--license)

## Project purpose

This repository demonstrates a simple, end-to-end approach to ASL sign recognition using MediaPipe to extract hand (and optionally pose) keypoints from a webcam stream, a small classifier trained on those keypoints, and utilities to collect data, train, export, and demo the model in Python and web (JSON) formats.

It is intended for learning, prototyping, and small demos — not for production-level sign recognition systems.

## Repository structure

Key files you'll see here:

- `app.py` - a demo app that loads the exported model and runs inference (desktop Python).
- `local_webcam_test.py` - quick webcam test utility (MediaPipe + model inference).
- `data_collector.py` - collects keypoints from MediaPipe and saves rows to `asl_keypoint_data.csv`.
- `asl_keypoint_data.csv` - the collected dataset of labeled keypoint vectors (if present).
- `train_classifier.py` - script to train the classifier from the CSV dataset.
- `export_model_to_json.py` - exports the trained classifier to a JSON format consumable by web demos.
- `asl_classifier_model.json` - an exported JSON model artifact (example output).
- `index.html` - minimal web demo that can consume the JSON model.
- `requirements.txt` - Python dependencies for running scripts.

## High-level methodology

The pipeline follows these steps:

1. Capture frames from a webcam and run MediaPipe's hand/pose detectors.
2. Extract relevant keypoints (x, y, z and/or visibility) for hands and optionally pose.
3. Flatten and normalize those keypoints into a feature vector per frame and label them.
4. Save labeled vectors to `asl_keypoint_data.csv` (data collection stage).
5. Train a small classifier (e.g., an MLP) on the CSV data with `train_classifier.py`.
6. Export the trained model to a JSON format with `export_model_to_json.py` for browser demos or light-weight loading.
7. Load the model in `app.py` or `index.html` and run live inference on new webcam frames.

Notes and assumptions:

- The repo includes an exported JSON model (`asl_classifier_model.json`) and a CSV dataset example. If your setup differs, follow the data collection steps below.
- The classifier is intentionally small (suitable for real-time on CPU). Typical implementations use a shallow MLP (TensorFlow/Keras). If you change the model type, adjust `export_model_to_json.py` accordingly.

## Data collection

Use `data_collector.py` to record labeled examples.

- Start the script, pick a label (e.g., the sign name), and record several seconds of footage while performing the sign.
- Aim for 50–500 examples per sign across different lighting, speeds, and hand positions for better generalization.

Typical usage (from project root):

```powershell
python -m pip install -r requirements.txt
python data_collector.py
```

The script appends one feature vector per frame to `asl_keypoint_data.csv` along with a label column.

## Preprocessing & features

- Keypoints are normalized to the frame size and can be zero-centered on a reference joint (for translation invariance).
- You can include raw (x,y) coordinates, z-depth, and visibility scores. The feature vector is usually a flattened list of floats.
- `train_classifier.py` expects the CSV to have a final column named `label` (or adapt the script if your CSV uses another column name).

## Training the classifier

Train a classifier from the collected CSV data. Example:

```powershell
python train_classifier.py --data asl_keypoint_data.csv --epochs 40 --batch-size 32
```

What `train_classifier.py` typically does:

- Loads `asl_keypoint_data.csv` as features X and labels y.
- Encodes labels, splits data into train/val (and optionally test) sets.
- Trains a small neural network (MLP) using TensorFlow/Keras.
- Saves the best model to disk and/or produces an exportable JSON representation.

If you want reproducible training, set a random seed in the script or pass one via CLI arguments.

## Exporting and using the model

To export a trained model to JSON (for the browser or a lightweight loader):

```powershell
python export_model_to_json.py --model-dir ./saved_model --out asl_classifier_model.json
```

- `export_model_to_json.py` should convert the weights and topology to a JSON format that `index.html` (or a small JS loader) understands. Adjust the exporter to match whatever inference loader you use.

To run the local Python demo:

```powershell
python app.py
```

To test webcam inference quickly (no UI):

```powershell
python local_webcam_test.py
```

## Tips for better accuracy

- Collect varied data: multiple speeds, dominant/non-dominant hand, lighting, backgrounds.
- Keep the camera stable and ensure the hand is mostly visible in the center of the frame.
- Normalize keypoints: use a root joint or bounding box normalization to remove scale/translation variance.
- Augment data if you're short on samples: jitter keypoints slightly or simulate small rotations.

## Evaluation

Add an evaluation step in `train_classifier.py` or as a separate script to compute accuracy, confusion matrix, and per-class precision/recall.

Quick example (conceptual):

```python
# after training
preds = model.predict(X_test)
# compute accuracy / confusion matrix using sklearn.metrics
```

Aim to hold out a test set (10–20%) and evaluate on unseen users if possible.

## Reports

When you run `train_classifier.py`, the script now writes a `reports/` directory containing:

- `evaluation_report.txt` — validation accuracy and a full classification report (precision/recall/f1) for each class.
- `confusion_matrix.png` — a plotted confusion matrix (PNG image) showing true vs predicted labels.
- `predictions.csv` — the test features with two extra columns: `y_true` and `y_pred` for per-sample analysis.

There is a simple viewer at `reports/index.html` that will display the confusion matrix image and load the text report (works when served or opened in a browser that allows file access). Use this to quickly inspect results after training.

## Troubleshooting

- "No hand detected": Improve lighting; increase camera resolution, move hand closer to camera.
- "Model not loading": Ensure JSON/export format matches the loader expected by `app.py` or `index.html`.
- "Slow inference": Use a smaller model, reduce input frequency (process every Nth frame), or run on a machine with a CPU that supports vectorized operations.

## Contributing

Contributions are welcome. Good first contributions:

- Add more labeled sample data (CSV rows) for new signs.
- Improve normalization and preprocessing in `data_collector.py`.
- Add unit tests for preprocessing and model export scripts.

Please open issues or PRs describing your change.

## License & Acknowledgements

This project is provided for educational purposes. Check repository metadata for the license or add one if you plan to publish.

This project uses MediaPipe (Google) for keypoint detection and (optionally) TensorFlow/Keras for training. See their respective licenses for deployment considerations.

## References

- MediaPipe: https://mediapipe.dev
- Keras: https://keras.io

---

If you'd like, I can also:

- add run scripts for Windows PowerShell to simplify commands,
- add a short demo GIF or screenshots, or
- extend `train_classifier.py` with a small evaluation report and save it to `reports/`.

Tell me which addition you'd prefer and I will update the repo accordingly.
