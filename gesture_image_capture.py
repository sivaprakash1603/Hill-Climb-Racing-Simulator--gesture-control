import cv2
import mediapipe as mp
import numpy as np
import os
import csv

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

# Directory to save the dataset
dataset_dir = 'hand_gesture_dataset'
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

# CSV file to store landmark points and labels
csv_file = os.path.join(dataset_dir, 'hand_gesture_data.csv')

# Initialize the CSV file for landmark storage if it doesn't exist
if not os.path.exists(csv_file):
    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        # Write header (21 points x 2 coordinates + 1 label = 43 columns)
        writer.writerow([f'x{i}' for i in range(21)] + [f'y{i}' for i in range(21)] + ['label'])

print("Press 0, 1, 2, etc. to label gestures. Press 'q' to quit.")

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Convert the image from BGR to RGB
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image_rgb)

    # Draw landmarks and connections on the image
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            # Extract landmark points
            h, w, _ = image.shape
            hand_landmark_array = np.array([[point.x * w, point.y * h] for point in hand_landmarks.landmark])

            # Draw the keypoints on the image
            for landmark in hand_landmark_array:
                cv2.circle(image, tuple(map(int, landmark)), 5, (255, 0, 0), -1)

            # Wait for key press to label the gesture
            key = cv2.waitKey(1) & 0xFF

            # If a number key is pressed, save the landmarks with the corresponding label
            if ord('0') <= key <= ord('9'):
                label = chr(key)  # Convert to string ('0' to '9')
                flattened_landmarks = hand_landmark_array.flatten().tolist()

                # Save landmarks and label to CSV
                with open(csv_file, mode='a', newline='') as file:
                    writer = csv.writer(file)
                    writer.writerow(flattened_landmarks + [label])

                print(f"Gesture labeled as {label} and saved to dataset.")

    # Show the processed image with landmarks
    cv2.imshow('Hand Tracking', image)

    # Break loop with 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
