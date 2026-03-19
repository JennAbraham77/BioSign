import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarker, HandLandmarkerOptions

# Path to the downloaded model
MODEL_PATH = "hand_landmarker.task"

# Create the hand landmarker task options
BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=MODEL_PATH),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=2
)

# Initialize the hand landmark detector
landmarker = HandLandmarker.create_from_options(options)

# Open webcam
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam")
    exit()

print("Press ESC to exit")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        break

    # Convert BGR image to MediaPipe Image type
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

    # Get integer timestamp
    timestamp_ms = int(cap.get(cv2.CAP_PROP_POS_MSEC))

    # Perform detection
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            print("New hand detected:")
        for i, landmark in enumerate(hand_landmarks):
            print(f"Landmark {i}: x={landmark.x:.3f}, y={landmark.y:.3f}, z={landmark.z:.3f}")

    # Draw landmarks on the frame
    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for i, landmark in enumerate(hand_landmarks):
                x, y, z = landmark.x, landmark.y, landmark.z
                cv2.circle(frame, (int(x * frame.shape[1]), int(y * frame.shape[0])), 5, (0, 255, 0), -1)
                cv2.putText(frame, str(i), (int(x * frame.shape[1]), int(y * frame.shape[0]) - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

    # Show the frame
    cv2.imshow("Hand Landmark Detection", frame)
    

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()