import cv2
import numpy as np
import tensorflow as tf
import mediapipe as mp
import pyautogui
import joblib

# Load the trained model and the scaler
model = tf.keras.models.load_model('gesture_recognition_model.h5')
scaler = joblib.load('scaler.pkl')

# Initialize MediaPipe hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initialize hand detection model
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.5)

# Gesture mapping based on the labels (update this if you have different labels)
gesture_map = {0: 'label_1', 1: 'label_2', 2: 'label_3'}

# Initialize previous gesture state
prev_gesture = None

# Open webcam
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    
    if not ret:
        print("Failed to capture frame from webcam.")
        break

    # Flip the frame horizontally for a mirror view


    # Convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    results = hands.process(rgb_frame)

    # If hand landmarks are detected
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw the hand landmarks on the frame
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Extract the landmark coordinates (x, y)
            h, w, _ = frame.shape
            hand_landmark_array = np.array([[point.x * w, point.y * h] for point in hand_landmarks.landmark])

            # Flatten the array for model input
            flattened_landmarks = hand_landmark_array.flatten()

            # Normalize the real-time landmarks using the saved scaler
            flattened_landmarks = scaler.transform([flattened_landmarks])

            # Predict using the trained model
            prediction = model.predict(flattened_landmarks)
            predicted_class = np.argmax(prediction[0])  # Get the predicted class
            gesture = gesture_map[predicted_class]

            print(f'Predicted Gesture: {gesture}')  # Print the detected gesture

            # Simulate key press based on the detected gesture
            if gesture == 'label_1':  # No key press for label_1
                if prev_gesture != 'label_1':
                    print("Detected label_1, releasing all keys.")
                    pyautogui.keyUp('right')
                    pyautogui.keyUp('left')
                    prev_gesture = 'label_1'

            elif gesture == 'label_3':  # Left key press for label_3
                if prev_gesture != 'label_3':
                    print("Detected label_3, pressing and holding left key.")
                    pyautogui.keyUp('right')  # Ensure the right key is not pressed
                    pyautogui.keyDown('left')
                    prev_gesture = 'label_3'

            elif gesture == 'label_2':  # Right key press for label_2
                if prev_gesture != 'label_2':
                    print("Detected label_2, pressing and holding right key.")
                    pyautogui.keyUp('left')  # Ensure the left key is not pressed
                    pyautogui.keyDown('right')
                    prev_gesture = 'label_2'

            # Display the predicted gesture on the frame
            cv2.putText(frame, f'Gesture: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    
    # Display the frame
    cv2.imshow('Hand Gesture Recognition', frame)

    # Break the loop on 'q' key press
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()

# Ensure all keys are released after the program ends
pyautogui.keyUp('left')
pyautogui.keyUp('right')
