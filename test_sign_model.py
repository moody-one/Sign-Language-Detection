# test_sign_model.py

import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pickle

# Load Trained Model
model = tf.keras.models.load_model('sign_language_model.h5')

# Load Label Encoder
with open('label_map.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            x_list = []
            y_list = []

            for lm in hand_landmarks.landmark:
                x_list.append(lm.x)
                y_list.append(lm.y)
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            landmarks_np = np.array(landmarks).reshape(1, -1)

            probs = model.predict(landmarks_np)
            predicted_class = np.argmax(probs)
            predicted_label = label_encoder.inverse_transform([predicted_class])[0]

            confidence = round(np.max(probs) * 100, 2)

            # Draw floating square (bounding box)
            h, w, _ = frame.shape
            min_x = int(min(x_list) * w) - 20
            min_y = int(min(y_list) * h) - 20
            max_x = int(max(x_list) * w) + 20
            max_y = int(max(y_list) * h) + 20

            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (255, 0, 0), 2)

            # Display label and confidence
            cv2.putText(frame, f'{predicted_label} ({confidence}%)', (min_x, min_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)

    cv2.imshow('Sign Language Detection', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()