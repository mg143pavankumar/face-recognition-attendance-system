import cv2
import cmake
import face_recognition
import os
import numpy as np
from datetime import datetime


# import image from the imagesAttendance folder
path = "ImagesAttendance"
images = []
classNames = []

# getting all the images from the folder
myList = os.listdir(path)

for cl in myList:
    curImg = cv2.imread(f'{path}/{cl}')
    images.append(curImg)
    classNames.append(os.path.splitext(cl)[0])

# print(images)
# print(classNames)


def findEncodings(images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]

        encodeList.append(encode)

    return encodeList


def markAttendance(name):
    with open('Attendance\Attendance.csv') as f:
        myDataList = f.readlines()
        nameList = []

        # print(myDataList)

        for line in myDataList:
            entry = line.split(",")
            nameList.append(entry[0])

        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')

            f.writelines(f'\n{name}, {dtString}')


markAttendance('Pavan')

encodeListForKnowFaces = findEncodings(images)
print("Encoding completed")
# print(len(encodeListForKnowFaces))


# Initialize webcam for taking images
webCam = cv2.VideoCapture(0)

while True:
    success, img = webCam.read()
    imgS = cv2.resize(img, (0, 0), None, fx=0.25, fy=0.25)
    imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

    facesCurrentFrame = face_recognition.face_locations(imgS)
    encodesCurrentFrame = face_recognition.face_encodings(
        imgS, facesCurrentFrame)

    for encodeFace, faceLocation in zip(encodesCurrentFrame, facesCurrentFrame):
        matches = face_recognition.compare_faces(
            encodeListForKnowFaces, encodeFace)

        faceDistance = face_recognition.face_distance(
            encodeListForKnowFaces, encodeFace)

        matchIndex = np.argmin(faceDistance)
        # print(matchIndex)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            print(name)

            y1, x2, y2, x1 = faceLocation
            y1, x2, y2, x1 = y1*4, x2*4, y2*4, x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2),
                          (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6),
                        cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 1)

    cv2.imshow('Webcam', img)
    cv2.waitKey(1)
