from pylab import *
import cv2 

def showAll_pylab(**imgs):
    ##When a double-starred parameter is declared such as $**imgs$, 
    ##then all the keyword arguments from that point till the end are collected as a dictionary called $'imgs'$.
    figure()
    gray()
    
    for (counter, (k,v)) in enumerate(imgs.items()): ##See how to use the $enumerate()$ function for creating a counter in a loop.
        subplot(1,len(imgs)  ,counter);imshow(v);title(k)

    show()
    
#-------------Main body
#Loading the images  
I=cv2.imread("Eye1.jpg", cv2.COLOR_RGB2GRAY) 
I=cv2.cvtColor(I, cv2.COLOR_RGB2GRAY)

#Binary Thresholding
(thresh, im_bw1) = cv2.threshold( I, 100, 255,cv2.THRESH_BINARY)
#Otsu Thresholding
(thresh, im_bw2) = cv2.threshold( I, 128, 255, cv2.THRESH_OTSU)
#Adaptive Thresholding
im_bw3=cv2.adaptiveThreshold(I, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY ,113,30)
#Main difference of Adaptive Thresholding is that , Adaptive thresholding adapts  the threshold value on each pixel to the local image characteristics
#while global thresholding methods dont. As a result, in some conditions , such as bad lighting conditions , adaptive thresholding gives better results.


showAll_pylab(I1=im_bw1,I2=im_bw2,I3=im_bw3)

