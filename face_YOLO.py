import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np
from ultralytics import YOLO

# Load the YOLO model
model = YOLO("yolov8n-face.pt")  # Load the YOLOv8 model (use your specific model)

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
cap = cv2.VideoCapture("AnjaliPandey.mp4")
success = True

# Set a threshold for recognizing faces
threshold = 0.6  # Adjust this value based on testing

while success:
    success, frame = cap.read()
    if success:
        # Get YOLOv8 model results (detection)
        results = model(frame)

        # Iterate through the results
        for result in results[0].boxes.data:  # Access the boxes data (bounding boxes)
            x1, y1, x2, y2, conf, cls = result  # Unpack bounding box info
            if conf >= threshold and cls == 0:  # Check for high confidence and person (face)
                # Crop the face from the frame
                face = frame[int(y1):int(y2), int(x1):int(x2)]

                # Resize to match the training data size (50x50)
                face_resized = cv2.resize(face, (50, 50)).flatten().reshape(1, -1)

                # Predict the name using KNN and get the probabilities
                face_name = KNN.predict(face_resized)
                probabilities = KNN.predict_proba(face_resized)

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
                cv2.putText(frame, display_text, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255), 1)
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)  # Cyan box

        # Display the frame
        cv2.imshow('Realtime Face Recognition', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("Error reading frame")
        break

# Cleanup
