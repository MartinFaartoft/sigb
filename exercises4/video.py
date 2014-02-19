import cv2
import Exercises4_Tools as tools
from pylab import *

cap = cv2.VideoCapture("video1.mp4")

template=None

cv2.namedWindow("input")
while(True):
    f,I = cap.read()
    if (template!=None):
        """
        Do the template matching here and show a rectangle around the best match
        """
        
    cv2.imshow("input", array(I))
    ch = cv2.waitKey(1)
    
    if ch == 32:#32 is space key.
        
        """
        set a template in the current frame after pressing the space key
        use SelectArea() and do slicing
        """

    

