import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

'''path = 'imgs'
images = []
Names = []

for cls in os.listdir(path):
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    Names.append(os.path.splitext(cls)[0])

print(Names)

def findEncodings(Images):
    encodeList = []

    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList

encodeKnown = findEncodings(images)
print('Encoding complete')

def markAttend(name):
    with open('dump.csv','r+') as f:
        myDataList = f.readlines()
        names = []
        print(myDataList)

        for line in myDataList:
            entry = line.split(',')
            names.append(entry[0])

        if name not in names:
            now = datetime.now()
            dtStr = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{dtStr}')'''

def markPresence(index):
    with open('dump.csv', 'a') as f:
        now = datetime.now()
        dtStr = now.strftime('%H:%M:%S')
        f.writelines(f'\nperson{index},{dtStr}')

encodeKnown = []

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    #imgS = cv2.resize(img,(0,0),None,0.25,0.25)
    imgS = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    facesLoc = face_recognition.face_locations(imgS)
    encodes = face_recognition.face_encodings(imgS, facesLoc)

    for encodeFace, faceLoc in zip(encodes, facesLoc):
        if encodeKnown == []:
            encodeKnown.append(encodeFace)

        matches = face_recognition.compare_faces(encodeKnown,encodeFace)
        faceDis = face_recognition.face_distance(encodeKnown,encodeFace)

        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            print('in match condition')
            name = f'person {matchIndex}'
            y1,x2,y2,x1 = faceLoc
            #x1,x2,y1,y2 = x1*4,x2*4,y1*4,y2*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,0,255),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_TRIPLEX,1,(255,255,255),2)

            markPresence(matchIndex)
        else:
            print('New Person')
            encodeKnown.append(encodeFace)

            name = f'person {len(encodeKnown)}'
            y1, x2, y2, x1 = faceLoc
            #x1, x2, y1, y2 = x1 * 4, x2 * 4, y1 * 4, y2 * 4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 0, 255), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_TRIPLEX, 1, (255, 255, 255), 2)

            markPresence(len(encodeKnown))

    cv2.imshow('webcam',img)
    cv2.waitKey(1)