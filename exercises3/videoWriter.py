from pylab import *
import cv2
import cv

cap = cv2.VideoCapture("GreenScreen(lowMP4).mp4")
writer = cv2.VideoWriter('GreenScreen_Output.avi', cv.CV_FOURCC('D','I','V','3'), 15.0, (640,360), True) #Make a video writer for
cv2.namedWindow("input")
running=True
while(running):
    running,I = cap.read()
    if (running):
        #Do your code here resulting -> imgNew
        writer.write(I)#Write the image
        cv2.imshow("input", array(I))
        ch = cv2.waitKey(1)
        if ch == 32:#  Break by SPACE key
            break
writer.release()