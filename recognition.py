import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle
import numpy as np


facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

with open('data/faces.pkl', 'rb') as w:
    faces = pickle.load(w)
print(len(faces))
with open('data/names.pkl', 'rb') as file:
    labels = pickle.load(file)

cap = cv2.VideoCapture(0)
print(labels)
print('Shape of Faces matrix ===> ', faces.shape)
KNN = KNeighborsClassifier(n_neighbors = 4)
KNN.fit(faces, labels)

success = True

while success:
    success, frame = cap.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

        for (a,b, w, h) in face_coordinates:
            frameCaptured = frame[b:b+h, a:a+w, :]
            frameResized = cv2.resize(frameCaptured, (50,50)).flatten().reshape(1,-1)
            face_name = KNN.predict(frameResized)
            cv2.putText(frame, face_name[0], (a, b - 10), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 255, 255),
                        1)  # Smaller and thinner text
            cv2.rectangle(frame, (a, b), (a + w, b + w), (255, 255, 0), 2)  # Cyan box

        cv2.imshow('Realtime face Recognition', frame)
        if cv2.waitKey(1) == 27:
            break
    else:
        print("error")
        break

cv2.destroyAllWindows()
cap.release()