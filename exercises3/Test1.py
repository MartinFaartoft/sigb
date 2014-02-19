
from pylab import *
import cv2 

#----------------------------Functions

def show1_OpenCV(image):
    ##This function define a window by namedWindow() and then show the image in that window
    image=array(image)
    cv2.namedWindow('By OpenCV', cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV'Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
    cv2.imshow('By OpenCV', image) ## show the image in 'By OpenCV' window
    cv2.waitKey(0) ## the window will be closed with a (any)key press
def show2_OpenCV(*image):
    ## Showing the image using OpenCV
    im=[]
    for i in image:
        im.append(array(i))
    
    cv2.namedWindow('1', cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV' Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
    cv2.imshow('1', im[0]) ## show the image in 'By OpenCV' window
    cv2.namedWindow('2', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('2', im[1]) 
    cv2.waitKey(0) ## the window will be closed with a (any)key press

          
def show1_pylab(image):
    ## Showing the image using pylab
    figure("By pylab")## Create a figure
    gray()
    title("1"); imshow(image)
    show()

def show2_pylab(*images):
    ## Showing the image using pylab
    figure("By pylab")## Create a figure
    gray()
    subplot(1,2,1);title("1"); imshow(images[0])## more about 'subplot()' : <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>
    subplot(1,2,2);title("2"); imshow(images[1])
    show()  


##------------------------------Main body

# Loading an image using openCV
I1=cv2.imread("children.tif") 






 
 
 
 
 
 
 