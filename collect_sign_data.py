# collect_sign_data.py

import cv2
import mediapipe as mp
import csv
import os

label = input("Enter the sign label (Example: A): ")

if not os.path.exists('landmark_data.csv'):
    with open('landmark_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['label'] + [f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)])

cap = cv2.VideoCapture(0)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.x)
            for lm in hand_landmarks.landmark:
                landmarks.append(lm.y)

            with open('landmark_data.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([label] + landmarks)

    cv2.putText(frame, f'Label: {label}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Data Collection - Press Q to Stop', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()