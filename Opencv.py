import Opencv as cv2
import numpy as np
import tensorflow as tf
from keras.models import load_model

# Load the trained model
model = tf.keras.models.load_model('path/to/your/saved_model.h5')

# Define the labels
labels = ['L', 'K', 'space', 'Q', 'H', 'S', 'B', 'N', 'P', 'nothing', 'V', 'R', 'U', 'O', 'Z', 'T', 'I', 'W', 'X', 'Y', 'C', 'del', 'E', 'G', 'J', 'M', 'D', 'A', 'F']

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

    # Preprocess the frame
    resized_frame = cv2.resize(frame, (128, 128))
    normalized_frame = resized_frame / 255.0
    reshaped_frame = np.reshape(normalized_frame, (1, 128, 128, 3))

    # Make predictions
    predictions = model.predict(reshaped_frame)
    predicted_label = labels[np.argmax(predictions)]

    # Display the predictions on the frame
    cv2.putText(frame, "Prediction: " + predicted_label, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Show the frame with the prediction
    cv2.imshow('Sign Language Prediction', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
