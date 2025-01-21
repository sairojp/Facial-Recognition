import cv2
import streamlit as st
import pickle
from knn import KNNClassifier
import pandas as pd

# Load Haar cascade and pre-trained KNN model
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load face data and labels
with open('../FaceRec-KNN/data/faces.pkl', 'rb') as f:
    faces = pickle.load(f)

with open('../FaceRec-KNN/data/names.pkl', 'rb') as f:
    labels = pickle.load(f)

# Train the KNN classifier
KNN = KNNClassifier(n_neighbors=4)
KNN.fit(faces, labels)

# Streamlit App UI
st.title("Real-time Face Recognition")
st.write("This app displays the live video feed with face detection and recognition.")

# Webcam video feed
cap = cv2.VideoCapture(0)  # Initialize the webcam

# Control checkbox to start/stop video
run = st.checkbox("Start Video", key="start_video_checkbox")

# Placeholder for the video feed
frame_placeholder = st.empty()

# Placeholder for the detected names table
names_placeholder = st.empty()

# List to store detected names
detected_names = []

while run:
    # Read frames from the webcam
    success, frame = cap.read()
    if not success:
        st.error("Failed to access the webcam.")
        break

    # Convert frame to grayscale for face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

    # Loop through detected faces
    for (a, b, w, h) in face_coordinates:
        # Crop, resize, and prepare face for KNN prediction
        face_cropped = frame[b:b+h, a:a+w]
        face_resized = cv2.resize(face_cropped, (50, 50)).flatten().reshape(1, -1)
        try:
            # Predict the face name
            face_name = KNN.predict(face_resized)[0]  # Get the name as a string
            detected_names.append(face_name)  # Add the name to the list
            # Annotate the frame with the name and bounding box
            cv2.putText(frame, face_name, (a, b - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        except Exception as e:
            st.error(f"Error in prediction: {e}")
        cv2.rectangle(frame, (a, b), (a + w, b + h), (255, 255, 0), 2)

    # Update the video feed
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_placeholder.image(frame_rgb, channels="RGB")

    # Update the table of detected names
    if detected_names:
        # Remove duplicates and prepare DataFrame
        unique_names = pd.DataFrame({"Detected Names": list(set(detected_names))})
        names_placeholder.table(unique_names)

    # # Re-check if the "Start Video" checkbox is still checked
    # run = st.checkbox("Start Video", key="start_video_checkbox")

# Release the webcam when done
cap.release()
