import cv2
import mediapipe as mp
import numpy as np
import pickle
import time

# ----------------------------
# 1️⃣ Load trained model and scaler
# ----------------------------
with open("gesture_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("gesture_scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

# ----------------------------
# 2️⃣ Initialize MediaPipe Hands
# ----------------------------
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ----------------------------
# 3️⃣ Helper functions
# ----------------------------
prev_landmarks = None
prev_time = None

def preprocess_landmarks(landmarks):
    """Return flattened x,y,z positions of all 21 landmarks"""
    data = []
    for lm in landmarks.landmark:
        data.extend([lm.x, lm.y, lm.z])
    return data  # 63 features

def compute_velocity(current_landmarks, prev_landmarks, dt):
    """Compute velocity (vx, vy, vz) for each landmark"""
    data = []
    if prev_landmarks is None:
        # First frame: zero velocity
        for _ in range(21):
            data.extend([0.0, 0.0, 0.0])
    else:
        for lm_curr, lm_prev in zip(current_landmarks.landmark, prev_landmarks.landmark):
            vx = (lm_curr.x - lm_prev.x)/dt
            vy = (lm_curr.y - lm_prev.y)/dt
            vz = (lm_curr.z - lm_prev.z)/dt
            data.extend([vx, vy, vz])
    return data  # 63 features

# ----------------------------
# 4️⃣ Start webcam capture
# ----------------------------
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    current_time = time.time()
    dt = (current_time - prev_time) if prev_time else 1/30  # fallback 1/30s
    prev_time = current_time

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(frame_rgb)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Draw skeleton
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Get features
            features_pos = preprocess_landmarks(hand_landmarks)          # 63 features
            features_vel = compute_velocity(hand_landmarks, prev_landmarks, dt)  # 63 features
            features = np.array(features_pos + features_vel).reshape(1, -1)  # 126 features

            # Scale features
            features_scaled = scaler.transform(features)

            # Predict user
            predicted_user = model.predict(features_scaled)[0]
            confidence = np.max(model.predict_proba(features_scaled)) * 100

            # Display prediction
            cv2.putText(frame, f"User: {predicted_user} ({confidence:.1f}%)",
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Update previous landmarks
            prev_landmarks = hand_landmarks

    cv2.imshow("Gesture Authentication", frame)

    # Exit on 'q' key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()