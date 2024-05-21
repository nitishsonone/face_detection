import cv2 as cv
import numpy as np

img=cv.imread(r'C:\Users\Lenovo\PycharmProjects\opencv\photos\download.jpeg')
cv.imshow('dowland',img)


blank=np.zeros((500,500,3),dtype='uint8')
cv.imshow('Black', blank)

# blank[:]=0,255,255
# cv.imshow('yellow',blank)


cv.rectangle(blank,(0,0),(250,250),(0,255,255),thickness=2)
cv.imshow('new',blank)

cv.waitKey(0)