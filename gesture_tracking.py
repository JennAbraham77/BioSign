import cv2
import mediapipe as mp
import csv
import os

# Create output folder
os.makedirs("gesture_data", exist_ok=True)

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
mp_draw = mp.solutions.drawing_utils

# User input
user_id = input("Enter your user ID: ")
gesture_label = input("Enter gesture label: ")

# CSV file
master_csv = "gesture_data/gesture_dataset.csv"

# Columns (x,y,z + velocity)
csv_columns = (
    [f"x{i}" for i in range(21)] +
    [f"y{i}" for i in range(21)] +
    [f"z{i}" for i in range(21)] +
    [f"vx{i}" for i in range(21)] +
    [f"vy{i}" for i in range(21)] +
    [f"vz{i}" for i in range(21)] +
    ["user_id", "gesture_label"]
)

# Create file if not exists
if not os.path.exists(master_csv):
    with open(master_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

# Webcam
cap = cv2.VideoCapture(0)

print("Press ESC to stop...")

prev_landmarks = None  # store previous frame

with open(master_csv, mode='a', newline='') as f:
    writer = csv.writer(f)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]

            current = []
            for lm in hand_landmarks.landmark:
                current.append((lm.x, lm.y, lm.z))

            row = []

            # positions
            for (x, y, z) in current:
                row.append(x)
            for (x, y, z) in current:
                row.append(y)
            for (x, y, z) in current:
                row.append(z)

            # velocities
            if prev_landmarks:
                for i in range(21):
                    vx = current[i][0] - prev_landmarks[i][0]
                    vy = current[i][1] - prev_landmarks[i][1]
                    vz = current[i][2] - prev_landmarks[i][2]
                    row.append(vx)
                for i in range(21):
                    row.append(current[i][1] - prev_landmarks[i][1])
                for i in range(21):
                    row.append(current[i][2] - prev_landmarks[i][2])
            else:
                # first frame → zero velocity
                row.extend([0]*63)

            # labels
            row.append(user_id)
            row.append(gesture_label)

            writer.writerow(row)

            prev_landmarks = current

            # Draw
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

        cv2.imshow("Gesture Tracking", img)

        if cv2.waitKey(1) & 0xFF == 27:
            break

cap.release()
cv2.destroyAllWindows()

print("Dataset saved!")
