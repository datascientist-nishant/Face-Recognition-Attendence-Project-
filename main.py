import cv2
import numpy as np
import face_recognition

imglisha= face_recognition.load_image_file('ImagesBasic/lisha.jpg')
imglisha=cv2.resize(imglisha,(500,600))
imglisha= cv2.cvtColor(imglisha,cv2.COLOR_BGR2RGB)

imgTest= face_recognition.load_image_file('ImagesBasic/lisha test.jpg')
imgTest= cv2.cvtColor(imgTest,cv2.COLOR_BGR2RGB)


face_loc= face_recognition.face_locations(imglisha)[0]
encodelisha=face_recognition.face_encodings(imglisha)[0]
encodelisha=[encodelisha]
#print(face_loc)
cv2.rectangle(imglisha,(face_loc[3], face_loc[0]),(face_loc[1],face_loc[2]),(255,0,255),2)


facelocTest= face_recognition.face_locations(imgTest)[0]
encodeTest=face_recognition.face_encodings(imgTest)[0]
#print(face_loc)
cv2.rectangle(imgTest,(facelocTest[3], facelocTest[0]),(facelocTest[1],facelocTest[2]),(255,0,255),2)



#results = face_recognition.compare_faces( [encodeTest,encodelisha], tolerance=0.7)
facedis=face_recognition.face_distance(encodelisha,encodeTest)
print(facedis)







cv2.imshow('lisha', imglisha)
cv2.imshow('lishaTest', imgTest)
cv2.waitKey(0)
