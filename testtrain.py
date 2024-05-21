import numpy as np
import cv2 as cv
har_cascade=cv.CascadeClassifier('harc_face.xml')
people=["Shah Rukh Khan", "Depika", "Narendra Modi", "Hritik Roshan"]
# features=np.load('features.npy')
# labels=np.load('labels.npy')
facerecongnization=cv.face.LBPHFaceRecognizer.create()
facerecongnization.read('face_read.yml')

img=cv.imread(r'C:\Users\Lenovo\PycharmProjects\opencv\photos\18.jpeg')
gray=cv.cvtColor(img,cv.COLOR_BGRA2GRAY)
cv.imshow('person',gray)
face_rect=har_cascade.detectMultiScale(gray,1.1,4)
for (x,y,w,h) in face_rect:
    face_roi=gray[y:y+h,x:x+h]

    label,confidence=facerecongnization.predict(face_roi)
    # print(f'lable={people[label]} with a confidence of {confidence}')
    cv.putText(img,str(people[label]),(20,20),cv.FONT_HERSHEY_COMPLEX,0.5,(0,255,255),thickness=2)
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,255),thickness=2)

cv.imshow('Dected',img)

cv.waitKey(0)