from pylab import *
import cv2

#load the video
cap = cv2.VideoCapture("GreenScreen(lowMP4).mp4")
cv2.namedWindow("input")

#Get the first frame and get the video size
f, backGround = cap.read()
(w,h,t)= shape(backGround)

#Load an image as the stationary backGround
backGround=cv2.imread("GreenScreen(lowMP4)_Green.jpg", cv2.COLOR_RGB2GRAY)
backGround=cv2.resize(backGround, (h,w))
backGround=cv2.cvtColor(backGround, cv2.COLOR_RGB2GRAY)


while(f):
    f, img = cap.read()
    img_backup=copy(img)
    if f==True:
        cv2.imshow("input", img)
        ch = cv2.waitKey(1)
        if ch == 32:#  Break by SPACE key
            break
#--------------------


