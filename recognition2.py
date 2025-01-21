import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np

# Load Haar cascade for face detection
facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Load saved face data and labels
with open('data/faces.pkl', 'rb') as w:
    faces = pickle.load(w)
print("Number of faces in dataset:", len(faces))

with open('data/names.pkl', 'rb') as file:
    labels = pickle.load(file)

# Train the KNN classifier
print("Labels in dataset:", labels)
print('Shape of Faces matrix ===> ', faces.shape)
KNN = KNeighborsClassifier(n_neighbors=4)
KNN.fit(faces, labels)

# Start video capture
cap = cv2.VideoCapture(0)
success = True

# Set a threshold for recognizing faces
threshold = 0.6  # Adjust this value based on testing

while success:
    success, frame = cap.read()
    if success:
        # Convert the frame to grayscale (required by Haar cascade)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Detect faces using Haar cascade
        face_coordinates = facecascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=4)

        for (a, b, w, h) in face_coordinates:
            # Extract the face region
            frame_captured = frame[b:b + h, a:a + w, :]

            # Resize to match the training data size (50x50)
            frame_resized = cv2.resize(frame_captured, (50, 50)).flatten().reshape(1, -1)

            # Predict the name using KNN and get the probabilities
            face_name = KNN.predict(frame_resized)
            probabilities = KNN.predict_proba(frame_resized)

            # Get the confidence score for the predicted class
            class_index = KNN.classes_.tolist().index(face_name[0])
            confidence_score = probabilities[0][class_index]

            # Set display logic based on confidence
            if confidence_score < threshold:
                display_text = "Unknown"
                color = (0, 0, 255)  # Red for unknown faces
            else:
                display_text = f"{face_name[0]} ({confidence_score:.2f})"
                color = (255, 255, 0)  # Cyan for recognized faces

            # Draw rectangle and put text on the frame
            cv2.putText(frame, display_text, (a, b - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
            cv2.rectangle(frame, (a, b), (a + w, b + h), color, 2)  # Cyan box

        # Display the frame
        cv2.imshow('Realtime Face Recognition', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Error reading frame")
        break

# Cleanup
cv2.destroyAllWindows()
cap.release()
