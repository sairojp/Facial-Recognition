import cv2
import numpy as np
import os
import pickle

faces = []
i = 0
frame_num =-1
cap = cv2.VideoCapture(0)


facecascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

name = input("Enter your name")
success = True

while success:
    success, frame = cap.read()
    frame_num += 1

    if success:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        face_coordinates = facecascade.detectMultiScale(gray, 1.3, 4)

        for (a, b, w, h) in face_coordinates:
            face = frame[b:b+h, a:a+w, :]
            resized_faces = cv2.resize(face, (50,50))

            if i % 10 == 0 and len(faces) < 10:
                faces.append(resized_faces)

            cv2.rectangle(frame, (a,b), (a+w, b+h), (255,0,0), 2)
        i += 1

        cv2.imshow('frames', frame)

        if cv2.waitKey(1) == 27 or len(faces) >= 10:
            break

    else:
        print('error')
        break

cv2.destroyAllWindows()
cap.release()

faces = np.asarray(faces)
faces = faces.reshape(10, -1)

if 'names.pkl' not in os.listdir('data/'):
    names = [name]*10
    with open('data/names.pkl', 'wb') as file:
        pickle.dump(names, file)
else:
    with open('data/names.pkl', 'rb') as file:
        names = pickle.load(file)

    names = names + [name]*10
    with open('data/names.pkl', 'wb') as file:
        pickle.dump(names, file)


if 'faces.pkl' not in os.listdir('data/'):
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(faces, w)
else:
    with open('data/faces.pkl', 'rb') as w:
        existing_face = pickle.load(w)

    updated_face = np.append(existing_face, faces, axis=0)
    with open('data/faces.pkl', 'wb') as w:
        pickle.dump(updated_face, w)