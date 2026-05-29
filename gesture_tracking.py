import cv2
import mediapipe as mp
import csv
import os
import math

os.makedirs("gesture_data", exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

user_id = input("Enter user ID: ")
gesture_label = input("Enter gesture label: ")

csv_file = "gesture_data/gesture_dataset.csv"

columns = (
    [f"x{i}" for i in range(21)] +
    [f"y{i}" for i in range(21)] +
    [f"z{i}" for i in range(21)] +
    [f"vx{i}" for i in range(21)] +
    [f"vy{i}" for i in range(21)] +
    [f"vz{i}" for i in range(21)] +
    ["user_id", "gesture_label"]
)

if not os.path.exists(csv_file):
    with open(csv_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

cap = cv2.VideoCapture(0)

prev_landmarks = None

def normalize_landmarks(landmarks):
    wrist = landmarks[0]
    max_distance = 0

    for x, y, z in landmarks:
        dist = math.sqrt(
            (x - wrist[0]) ** 2 +
            (y - wrist[1]) ** 2 +
            (z - wrist[2]) ** 2
        )
        max_distance = max(max_distance, dist)

    if max_distance == 0:
        max_distance = 1

    normalized = []

    for x, y, z in landmarks:
        normalized.append((
            (x - wrist[0]) / max_distance,
            (y - wrist[1]) / max_distance,
            (z - wrist[2]) / max_distance
        ))

    return normalized

print("Press ESC to stop collecting...")

with open(csv_file, "a", newline="") as f:
    writer = csv.writer(f)

    while True:
        ret, frame = cap.read()

        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            current = []

            for lm in hand_landmarks.landmark:
                current.append((lm.x, lm.y, lm.z))

            current = normalize_landmarks(current)

            row = []

            for x, y, z in current:
                row.append(x)

            for x, y, z in current:
                row.append(y)

            for x, y, z in current:
                row.append(z)

            if prev_landmarks:
                for i in range(21):
                    row.append(current[i][0] - prev_landmarks[i][0])

                for i in range(21):
                    row.append(current[i][1] - prev_landmarks[i][1])

                for i in range(21):
                    row.append(current[i][2] - prev_landmarks[i][2])
            else:
                row.extend([0] * 63)

            row.append(user_id)
            row.append(gesture_label)

            if len(row) == 128:
                writer.writerow(row)
            else:
                print("Skipped bad row with length:", len(row))

            prev_landmarks = current

            mp_draw.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

        cv2.imshow("Gesture Collection", frame)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

print("Dataset collection complete!")
