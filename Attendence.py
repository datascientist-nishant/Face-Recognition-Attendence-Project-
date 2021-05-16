import cv2
import numpy as np
import face_recognition
import os

path='ImagesBasic'
images=[]
classnames=[]
myList= os.listdir(path)
print(myList)

for cl in myList:
    curimage= cv2.imread(f'{path}/{cl}')
    images.append(curimage)
    classnames.append(os.path.splitext(cl)[0])
print(classnames)


def finalencodings(images):
    encodelist=[]
    for img in images:
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        encode = face_recognition.face_locations(img)[0]
        encodelist.append(encode)
    return encodelist

encodelistknown= finalencodings(images)
print(len(encodelistknown))

cap= cv2.VideoCapture(0)

while True:
    success, img= cap.read()
    imgs= cv2.resize(img,(0,0),None,0.25,0.25)
    imgs= cv2.cvtColor(imgs, cv2.COLOR_BGR2RGB)

    facesCurFrame = face_recognition.face_locations(imgs)
    encodeCurFrame = face_recognition.face_encodings(imgs,facesCurFrame)

    for encodeFace, faceloc in zip(encodeCurFrame, facesCurFrame):
        matches = face_recognition.compare_faces(encodelistknown,encodeFace)
        facedis= face_recognition.face_distance(encodelistknown,encodeFace)

        matchindex= np.argmin(facedis)


        if matches[matchindex]:
            name = classnames[matchindex].upper()
            print(name)


    cv2.imshow("webcam",img)
    cv2.waitKey(0)
    cv2.destroyWindow("webcam")










