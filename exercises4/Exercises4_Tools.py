import numpy as np
import cv2
from pylab import *


#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#--------------------------Area Selection
help_message = '''This function returns the corners of the selected area as: [(UpLeftcorner),(DownRightCorner)]
Use the RightButton and LeftButton of mouse and 
Click on the image to set the area
Keys:
  Enter - OK
  ESC   - exit (Cancel)
'''


seed_Left_pt = None
seed_Right_pt = None
img=None

def update(dummy=None):
    if (seed_Left_pt is None) | (seed_Right_pt is None):
        cv2.imshow('SELECT AN AREA', img)
        return
    
    flooded = img.copy()
    cv2.rectangle(flooded, seed_Left_pt, seed_Right_pt,  (0, 0, 255),1)
    cv2.imshow('SELECT AN AREA', flooded)

    
    
def onmouse(event, x, y, flags, param):
    global seed_Left_pt
    global seed_Right_pt
    
    if  flags & cv2.EVENT_FLAG_LBUTTON:
        seed_Left_pt = x, y
#        print seed_Left_pt

    if  flags & cv2.EVENT_FLAG_RBUTTON: 
        seed_Right_pt = x, y
#        print seed_Right_pt
    
    update()
def setCorners():
    ponits=[]

    UpLeft=(min(seed_Left_pt[0],seed_Right_pt[0]),min(seed_Left_pt[1],seed_Right_pt[1]))
    DownRight=(max(seed_Left_pt[0],seed_Right_pt[0]),max(seed_Left_pt[1],seed_Right_pt[1]))
    ponits.append(UpLeft)
    ponits.append(DownRight)
    return ponits
    
            
def SelectArea(image):# This function returns the corners of the selected area as: [(UpLeftcorner),(DownRightCorner)]
    global img
    img = image.copy()
    print help_message
    update()
    cv2.namedWindow('SELECT AN AREA', cv2.WINDOW_AUTOSIZE )# cv2.WINDOW_AUTOSIZE
    cv2.setMouseCallback('SELECT AN AREA', onmouse)
    
    while True:
    
        ch = cv2.waitKey()
#        print ch
        if ch == 27:
            break
        if ch == 13:#enter key   
            cv2.destroyWindow('SELECT AN AREA')    
            return setCorners()
            break

#--------------------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------Display Functions

def show_1_image_OpenCV(image):
    ## Showing the image using OpenCV
    image=array(image)
    cv2.namedWindow('By OpenCV', cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV'Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
    cv2.imshow('By OpenCV', image) ## show the image in 'By OpenCV' window
    cv2.waitKey(0) ## the window will be closed with a (any)key press
def show_2_images_OpenCV(*image):
    ## Showing the image using OpenCV
    im=[]
    for i in image:
        im.append(array(i))
    
    cv2.namedWindow('1', cv2.WINDOW_AUTOSIZE)## create a window called 'By OpenCV' Using the \nw{cv2.WINDOW_AUTOSIZE} parameter when defining a window display the image with its actual size in the window.
    cv2.imshow('1', im[0]) ## show the image in 'By OpenCV' window
    cv2.namedWindow('2', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('2', im[1]) 
    cv2.waitKey(0) ## the window will be closed with a (any)key press

  
        
def show_1_image_pylab(image):
    ## Showing the image using pylab
    figure("By pylab")## Create a figure
    gray()
    title("1"); imshow(image)
    show()

def show_2_images_pylab(*images):
    ## Showing the image using pylab
    figure("By pylab")## Create a figure
    gray()
    subplot(1,2,1);title("1"); imshow(images[0])## more about 'subplot()' : <http://www.scipy.org/Cookbook/Matplotlib/Multiple_Subplots_with_One_Axis_Label>
    subplot(1,2,2);title("2"); imshow(images[1])
    show()  

def show_Multiple_images_OpenCV(**image):
    ## Showing the image using OpenCV
    for (k,v) in image.items():
        cv2.namedWindow(str(k), cv2.WINDOW_AUTOSIZE) 
        cv2.imshow(str(k) , array(v))
    cv2.waitKey(0) ## the window will be closed with a (any)key press
    
def show_Multiple_images_pylab(**imgs):
    ##When a double-starred parameter is declared such as $**imgs$, 
    ##then all the keyword arguments from that point till the end are collected as a dictionary called $'imgs'$.
    
    
    figure()
    gray()
    
    for (counter, (k,v)) in enumerate(imgs.items()): ##See how to use the $enumerate()$ function for creating a counter in a loop.
       
        subplot(1,len(imgs)  ,counter);imshow(v);title(k)

    show()  
    
#--------------------------------------------------------------------------------------------------------------------------------------------------------------------    
def to_uint8( data ) :
    # maximum pixel
    latch = np.zeros_like( data )
    latch[:] = 255
    # minimum pixel
    zeros = np.zeros_like( data )
    
    # unrolled to illustrate steps
    d = np.maximum( zeros, data )
    d = np.minimum( latch, d )
    
    # cast to uint8
    return np.asarray( d, dtype="uint8" )



def rotateImage(image, angle):
    image_center = tuple(np.array(image.shape)/2)
    rot_mat = cv2.getRotationMatrix2D(image_center[0:2],angle,1)
    size = image.shape
    result = cv2.warpAffine(image, rot_mat,dsize=size[0:2],flags=cv2.INTER_LINEAR)
    return result