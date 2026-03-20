import cv2
import mediapipe as mp
import csv
import os
import glob

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

# Define master CSV
master_csv = "gesture_data/gesture_dataset.csv"

# Define columns
csv_columns = ["user_id"] + [f"x{i}" for i in range(21)] + [f"y{i}" for i in range(21)] + ["gesture_label"]

# If master CSV doesn't exist, create it with header
if not os.path.exists(master_csv):
    with open(master_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_columns)

# Function to combine old CSVs (if any)
existing_files = glob.glob(f"gesture_data/user_*_gesture_*.csv")
for file in existing_files:
    with open(file, 'r') as f:
        reader = csv.reader(f)
        next(reader)  # skip header
        rows = list(reader)
    with open(master_csv, 'a', newline='') as f_master:
        writer = csv.writer(f_master)
        writer.writerows(rows)
    os.remove(file)  # optional: remove old file after merging

# Start webcam to collect new data
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press ESC to exit, collecting landmarks...")

with open(master_csv, mode='a', newline='') as f:
    writer = csv.writer(f)

    while True:
        success, img = cap.read()
        if not success:
            print("Failed to grab frame")
            break

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(img_rgb)

        if result.multi_hand_landmarks:
            hand_landmarks = result.multi_hand_landmarks[0]  # first hand
            row = [user_id]  # Add user_id as first column
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
print(f"All data saved to {master_csv}")
