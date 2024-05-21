import os
import cv2 as cv
import numpy as np


people=["Shah Rukh Khan", "Depika", "Narendra Modi", "Hritik Roshan"]


DIR=r'C:\Users\Lenovo\Desktop\train'
har_cascade=cv.CascadeClassifier('harc_face.xml')
features=[]
labels=[]

def Creat_train():
    for person in people:
        path=os.path.join(DIR,person)
        label=people.index(person)
        for img in os.listdir(path):
            img_path=os.path.join(path,img)
            img_array=cv.imread(img_path)
            gray=cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)
            faces_rect=har_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=4)

            for (x,y,w,h) in faces_rect:
                face_roi=gray[y:y+h , x:x+w]
                features.append(face_roi)
                labels.append(label)
Creat_train()

print("Trained-------")
features=np.array(features, dtype='object')
labels=np.array(labels)

facerecongnization=cv.face.LBPHFaceRecognizer.create()
facerecongnization.train(features,labels)

facerecongnization.save('face_read.yml')
np.save('feature.npy', features)
np.save('labels.npy',labels)

















