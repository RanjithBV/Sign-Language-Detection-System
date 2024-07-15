import cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model
import mediapipe as mp

# Load the trained model
model = tf.keras.models.load_model('./model.h5')

# Define the labels
labels = ['A', 'B', 'C', 'D', 'del', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'Nothing', 'Space']

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)

# Initialize MediaPipe Drawing
mp_drawing = mp.solutions.drawing_utils

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Flip the frame horizontally for a later selfie-view display
    frame = cv2.flip(frame, 1)

    # Convert the BGR image to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame to detect hands
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            # Get bounding box of the hand
            h, w, c = frame.shape
            x_min = min([landmark.x for landmark in hand_landmarks.landmark]) * w
            y_min = min([landmark.y for landmark in hand_landmarks.landmark]) * h
            x_max = max([landmark.x for landmark in hand_landmarks.landmark]) * w
            y_max = max([landmark.y for landmark in hand_landmarks.landmark]) * h

            # Define the region of interest (ROI) for hand placement
            x, y, w, h = int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)

            # Extract the ROI
            roi = frame[y:y+h, x:x+w]

            if roi.size != 0:
                # Resize the ROI to the size required by the model
                resized_frame = cv2.resize(roi, (128, 128))

                # Normalize the ROI
                normalized_frame = resized_frame / 255.0

                # Reshape the frame for the model
                reshaped_frame = np.reshape(normalized_frame, (1, 128, 128, 3))

                # Make predictions
                predictions = model.predict(reshaped_frame)
                predicted_label = labels[np.argmax(predictions)]

                # Display the predictions on the frame
                cv2.putText(frame, "Prediction: " + predicted_label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # Draw the hand landmarks
            mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Show the frame with the prediction
    cv2.imshow('Sign Language Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
