import cv2 as cv
img=cv.imread(r'C:\Users\Lenovo\PycharmProjects\opencv\photos\cat.jpeg')
cv.imshow('Cat',img)
cv.waitKey(0)


capture=cv.VideoCapture(0)#here we can give path of existing video or integer for webcam like 0
while True:
    isTrue, frame=capture.read()
    cv.imshow('path')
    













