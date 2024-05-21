import cv2 as cv


# img=cv.imread(r'C:\Users\Lenovo\PycharmProjects\opencv\photos\ddd.jpeg')
# cv.imshow('Face_Img', img)

grp_img=cv.imread(r'C:\Users\Lenovo\PycharmProjects\opencv\photos\grp.jpg')


def rescaleFrame(frame, scale=0.75):
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)

    dimensions = (800,500)

    return cv.resize(frame, dimensions, interpolation=cv.INTER_AREA)

resize=rescaleFrame(grp_img)
cv.imshow('resized',resize)

gray=cv.cvtColor(grp_img,cv.COLOR_BGR2GRAY)
# cv.imshow('gray_face',gray)


har_cascade=cv.CascadeClassifier('harc_face.xml')
faces=har_cascade.detectMultiScale(grp_img,scaleFactor=1.1,minNeighbors=4)
print(f"numberface found {len(faces)}")

for (x,y,w,h) in faces:
    cv.rectangle(resize,(x,y),(x+w,y+h),(0,50,50),thickness=2)
cv.imshow('Detected_image',resize)


cv.waitKey(0)