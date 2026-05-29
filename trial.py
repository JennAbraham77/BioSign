import cv2
import mediapipe as mp
import numpy as np
import pickle
import time
from collections import deque
from keras.models import load_model

model = load_model("gesture_lstm_model.h5")

with open("gesture_lstm_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

with open("gesture_label_encoder.pkl", "rb") as f:
    encoder = pickle.load(f)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

sequence_length = 30

frame_buffer = deque(maxlen=sequence_length)
prediction_buffer = deque(maxlen=10)

prev_landmarks = None
prev_time = None

def get_normalized_landmarks(hand_landmarks):
    raw = []

    for lm in hand_landmarks.landmark:
        raw.append((lm.x, lm.y, lm.z))

    wrist = raw[0]

    max_distance = 0

    for x, y, z in raw:
        distance = np.sqrt(
            (x - wrist[0]) ** 2 +
            (y - wrist[1]) ** 2 +
            (z - wrist[2]) ** 2
        )
        max_distance = max(max_distance, distance)

    if max_distance == 0:
        max_distance = 1

    normalized = []

    for x, y, z in raw:
        normalized.append((
            (x - wrist[0]) / max_distance,
            (y - wrist[1]) / max_distance,
            (z - wrist[2]) / max_distance
        ))

    return normalized

def get_position_features(current):
    row = []

    for x, y, z in current:
        row.append(x)

    for x, y, z in current:
        row.append(y)

    for x, y, z in current:
        row.append(z)

    return row

def compute_velocity(current, prev, dt):
    if prev is None:
        return [0.0] * 63

    row = []

    for i in range(21):
        row.append((current[i][0] - prev[i][0]) / dt)

    for i in range(21):
        row.append((current[i][1] - prev[i][1]) / dt)

    for i in range(21):
        row.append((current[i][2] - prev[i][2]) / dt)

    return row

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()

    if not ret:
        break

    current_time = time.time()
    dt = max((current_time - prev_time), 1e-3) if prev_time else 1 / 30
    prev_time = current_time

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]

        mp_drawing.draw_landmarks(
            frame,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS
        )

        current = get_normalized_landmarks(hand_landmarks)

        position_features = get_position_features(current)
        velocity_features = compute_velocity(current, prev_landmarks, dt)

        features = np.array(position_features + velocity_features).reshape(1, -1)

        features_scaled = scaler.transform(features)[0]

        frame_buffer.append(features_scaled)

        prev_landmarks = current

        if len(frame_buffer) == sequence_length:
            sequence = np.array(frame_buffer).reshape(1, sequence_length, -1)

            probabilities = model.predict(sequence, verbose=0)[0]

            pred_index = np.argmax(probabilities)
            confidence = probabilities[pred_index] * 100

            predicted_user = encoder.inverse_transform([pred_index])[0]

            prediction_buffer.append(predicted_user)

            stable_user = max(
                set(prediction_buffer),
                key=list(prediction_buffer).count
            )

            cv2.putText(
                frame,
                f"User: {stable_user} ({confidence:.1f}%)",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

    else:
        frame_buffer.clear()
        prev_landmarks = None

    cv2.imshow("Gesture Authentication - LSTM", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
