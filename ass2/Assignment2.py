#from scipy import ndimage
import cv2
import numpy as np
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *
import math
import SIGBTools

H = np.array([[ -2.15055949e-01,  -4.16963375e+00,   4.70172592e+02],
       [  5.10377629e-01,  -1.36756222e+00,   9.75731969e+00],
       [ -3.78641113e-04,  -1.06728204e-02,   1.00000000e+00]])
calibration = [[(34.983870967741922, 206.98387096774195), (87.5, 199.88709677419354), (250.72580645161293, 164.40322580645164), (293.30645161290323, 229.69354838709677)], [(327.56451612903243, 208.98387096774206), (327.56451612903243, 187.69354838709694), (316.91935483870975, 102.53225806451621), (352.4032258064517, 98.983870967742064)]]
#found with createHomography()
#H = np.array([[ -2.37653399e-01,  -1.67132178e+00,   2.58125139e+02],
#       [ -1.56942938e-02,  -9.04541378e-01,   1.28741519e+02],
#       [ -9.20855403e-04,  -6.07667285e-03,   1.00000000e+00]])

#H = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

def frameTrackingData2BoxData(data):
    #Convert a row of points into tuple of points for each rectangle
    pts= [ (int(data[i]),int(data[i+1])) for i in range(0,11,2) ]
    boxes = [];
    for i in range(0,7,2):
        box = tuple(pts[i:i+2])
        boxes.append(box)   
    return boxes


def simpleTextureMap():

    I1 = cv2.imread('Images/ITULogo.jpg')
    I2 = cv2.imread('Images/ITUMap.bmp')

    #Print Help
    H,Points  = SIGBTools.getHomographyFromMouse(I1,I2,4)
    h, w,d = I2.shape
    overlay = cv2.warpPerspective(I1, H,(w, h))
    M = cv2.addWeighted(I2, 0.5, overlay, 0.5,0)

    cv2.imshow("Overlayed Image",M)
    cv2.waitKey(0)

def showImageandPlot(N):
    #A simple attenmpt to get mouse inputs and display images using matplotlib
    I = cv2.imread('groundfloor.bmp')
    drawI = I.copy()
    #make figure and two subplots
    fig = figure(1) 
    ax1  = subplot(1,2,1) 
    ax2  = subplot(1,2,2) 
    ax1.imshow(I) 
    ax2.imshow(drawI)
    ax1.axis('image') 
    ax1.axis('off') 
    points = fig.ginput(5) 
    fig.hold('on')
    
    for p in points:
        #Draw on figure
        subplot(1,2,1)
        plot(p[0],p[1],'rx')
        #Draw in image
        cv2.circle(drawI,(int(p[0]),int(p[1])),2,(0,255,0),10)
    ax2.cla
    ax2.imshow(drawI)
    draw() #update display: updates are usually defered 
    show()
    savefig('somefig.jpg')
    cv2.imwrite("drawImage.jpg", drawI)

blue = 255, 0, 0
green = 0, 255, 0

def displayTrace(squareInVideo):
    ituMap = cv2.imread('Images/ITUMap.bmp')
    p1, p2 = squareInVideo
    p1_map = multiplyPointByHomography(p1, H)
    p2_map = multiplyPointByHomography(p2, H)
    #cv2.rectangle(ituMap, p1_map, p2_map, green)
    cv2.circle(ituMap, p1_map, 1, blue, 5)
    cv2.imshow("Tracking", ituMap)


def multiplyPointByHomography(point, homography):
    #homography = np.linalg.inv(homography)
    p = np.ones(3)
    p[0] = point[0]
    p[1] = point[1]

    p_prime = np.dot(homography, p)
    #print p_prime
    p_prime = p_prime * 1 / p_prime[2]
    #print p,p_prime
    return (int(p_prime[0]), int(p_prime[1]))

def texturemapGridSequence():
    """ Skeleton for texturemapping on a video sequence"""
    fn = 'GridVideos/grid1.mp4'
    cap = cv2.VideoCapture(fn)
    drawContours = True;

    texture = cv2.imread('Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)


    mTex,nTex,t = texture.shape

    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape

    cv2.imshow("win2",imgOrig)

    pattern_size = (9, 6)

    idx = [0,8,45,53]
    while(running):
    #load Tracking data
        running, imgOrig = cap.read()
        if(running):
            imgOrig = cv2.pyrDown(imgOrig)
            gray = cv2.cvtColor(imgOrig,cv2.COLOR_BGR2GRAY)
            found, corners = cv2.findChessboardCorners(gray, pattern_size)
            if found:
                term = ( cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 0.1 )
                cv2.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
                cv2.drawChessboardCorners(imgOrig, pattern_size, corners, found)

                for t in idx:
                    cv2.circle(imgOrig,(int(corners[t,0,0]),int(corners[t,0,1])),10,(255,t,t))
            cv2.imshow("win2",imgOrig)
            cv2.waitKey(1)



def realisticTexturemap(scale,point,map):
    #H = np.load('H_G_M')
    print "Not implemented yet\n"*30

    #open picture and first frame of video
    #call getHomographyFromMouse
    #print resulting 3x3 matrix
def createHomography():
    img1 = cv2.imread('Images/ITUMap.bmp')

    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
    
    #load Tracking data
    _, img2 = cap.read()

    print SIGBTools.getHomographyFromMouse(img2, img1)



def showFloorTrackingData():
    #Load videodata
    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
    
    #load Tracking data
    running, imgOrig = cap.read()
    dataFile = np.loadtxt('GroundFloorData/trackingdata.dat')
    m,n = dataFile.shape
    
    fig = figure()
    for k in range(m):
        running, imgOrig = cap.read() 
        if(running):
            boxes= frameTrackingData2BoxData(dataFile[k,:])
            boxColors = [(255,0,0),(0,255,0),(0,0,255)]
            for k in range(0,3):
                aBox = boxes[k]
                cv2.rectangle(imgOrig, aBox[0], aBox[1], boxColors[k])
            cv2.imshow("boxes",imgOrig);
            displayTrace(boxes[1])
            cv2.waitKey(1)

def angle_cos(p0, p1, p2):
    d1, d2 = p0-p1, p2-p1
    return abs( np.dot(d1, d2) / np.sqrt( np.dot(d1, d1)*np.dot(d2, d2) ) )

def findSquares(img,minSize = 2000,maxAngle = 1):
    """ findSquares intend to locate rectangle in the image of minimum area, minSize, and maximum angle, maxAngle, between 
    sides"""
    squares = []
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
         cnt_len = cv2.arcLength(cnt, True)
         cnt = cv2.approxPolyDP(cnt, 0.08*cnt_len, True)
         if len(cnt) == 4 and cv2.contourArea(cnt) > minSize and cv2.isContourConvex(cnt):
             cnt = cnt.reshape(-1, 2)
             max_cos = np.max([angle_cos( cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
             if max_cos < maxAngle:
                 squares.append(cnt)
    return squares

def DetectPlaneObject(I,minSize=1000):
      """ A simple attempt to detect rectangular 
      color regions in the image"""
      HSV = cv2.cvtColor(I, cv2.COLOR_BGR2HSV)
      h = HSV[:,:,0].astype('uint8')
      s = HSV[:,:,1].astype('uint8')
      v = HSV[:,:,2].astype('uint8')
      
      b = I[:,:,0].astype('uint8')
      g = I[:,:,1].astype('uint8')
      r = I[:,:,2].astype('uint8')
     
      # use red channel for detection.
      s = (255*(r>230)).astype('uint8')
      iShow = cv2.cvtColor(s, cv2.COLOR_GRAY2BGR)
      cv2.imshow('ColorDetection',iShow)
      squares = findSquares(s,minSize)
      return squares
  
def texturemapObjectSequence():
    """ Poor implementation of simple texturemap """
    fn = 'BookVideos/Seq3_scene.mp4'
    cap = cv2.VideoCapture(fn) 
    drawContours = True;
    
    texture = cv2.imread('images/ITULogo.jpg')
    #texture = cv2.transpose(texture)
    mTex,nTex,t = texture.shape
    
    #load Tracking data
    running, imgOrig = cap.read()
    mI,nI,t = imgOrig.shape
    
    print running 
    while(running):
        for t in range(20):
            running, imgOrig = cap.read() 
        
        if(running):
            squares = DetectPlaneObject(imgOrig)
            
            for sqr in squares:
                 #Do texturemap here!!!!
                 #TODO
                 
                 if(drawContours):                
                     for p in sqr:
                         cv2.circle(imgOrig,(int(p[0]),int(p[1])),3,(255,0,0)) 
                 
            
            if(drawContours and len(squares)>0):    
                cv2.drawContours( imgOrig, squares, -1, (0, 255, 0), 3 )

            cv2.circle(imgOrig,(100,100),10,(255,0,0))
            cv2.imshow("Detection",imgOrig)
            cv2.waitKey(1)
#createHomography()
showFloorTrackingData()
#simpleTextureMap()
#realisticTexturemap(0,0,0)
#texturemapGridSequence()