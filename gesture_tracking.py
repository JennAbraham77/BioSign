# gesture_tracking.py
import cv2
import mediapipe as mp
import csv
import os

# Create output folder
os.makedirs("gesture_data", exist_ok=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# Ask for user ID and gesture label
user_id = input("Enter your user ID: ")
gesture_label = input("Enter gesture label (e.g., 0=fist, 1=thumbs_up): ")

# Prepare CSV file
csv_file = f"gesture_data/user_{user_id}_gesture_{gesture_label}.csv"
csv_columns = [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["gesture_label"]

with open(csv_file, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_columns)

    # Start webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam")
        exit()

    print("Press ESC to exit, collecting landmarks...")

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]  # first hand
            row = []
            for lm in hand_landmarks.landmark:
                row.append(lm.x)
            for lm in hand_landmarks.landmark:
                row.append(lm.y)
            row.append(gesture_label)
            writer.writerow(row)

            # Draw landmarks
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Tracking", img)

        if cv2.waitKey(1) & 0xFF == 27:  # ESC
            break

cap.release()
cv2.destroyAllWindows()
print(f"Data saved to {csv_file}")
