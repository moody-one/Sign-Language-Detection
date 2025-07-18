# 🤟 Sign Language Detection with Mediapipe & TensorFlow

A real-time sign language recognition system using hand landmarks via Mediapipe and a custom-trained neural network. This project helps detect and classify static hand gestures (signs) using live webcam input.

## 📁 Files Overview
- `collect_sign_data.py` — Capture and save hand landmarks with labels.
- `train_sign_model.py` — Train the classification model using the collected data.
- `sign_language_model.h5` — Trained Keras model file.
- `label_map.pkl` — Mapping of gesture labels to class indices.
- `landmark_data.csv` — Collected landmark dataset.
- `test_sign_model.py` — Run the trained model in real time using your webcam.

## 🛠️ Tech Stack
- Python
- Mediapipe
- OpenCV
- TensorFlow / Keras

## 🔧 How to Run
1. Run `collect_sign_data.py` to gather sign samples.
2. Train using `train_sign_model.py`.
3. Test live with `test_sign_model.py`.

## 📌 Notes
Make sure your webcam is connected. Use a plain background and good lighting for best results.

---

Contributions and feedback are welcome!
