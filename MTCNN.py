import cv2
from sklearn.neighbors import KNeighborsClassifier
import pickle

from mtcnn import MTCNN

# pip install 'tensorflow[and-cuda]' Use this command to install tf for GPU Below code is to check if GPU available
# import tensorflow as tf
# print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

detector = MTCNN()


with open('data/faces.pkl', 'rb') as w:
    faces = pickle.load(w)

with open('data/names.pkl', 'rb') as file:
    labels = pickle.load(file)

cap = cv2.VideoCapture(0)

print('Shape of Faces matrix ===> ', faces.shape)
KNN = KNeighborsClassifier(n_neighbors = 4)
KNN.fit(faces, labels)

success = True

while success:
    success, frame = cap.read()
    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_coordinates = detector.detect_faces(frame)

        for face in face_coordinates:
            a,b,w,h = face['box']
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