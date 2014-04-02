#from scipy import ndimage
import cv2
import numpy as np
from pylab import *
from matplotlib import *
from matplotlib.pyplot import *
from scipy import *
import math
import SIGBTools

#found by createHomography with n=6
H = np.array([[  1.47453831e-01,   9.78826731e-01,   2.04270412e+02],
       [ -5.10193725e-01,   4.10523270e-01,   1.95623806e+02],
       [  1.70178472e-04,   1.12945293e-03,   1.00000000e+00]])
calibration = [[(34.983870967741922, 208.40322580645164), (84.661290322580641, 198.4677419354839), (249.30645161290323, 162.98387096774195), (296.14516129032256, 229.69354838709677), (202.4677419354839, 253.82258064516128), (172.66129032258067, 180.01612903225805)], [(331.11290322580658, 212.53225806451621), (331.11290322580658, 187.69354838709694), (324.01612903225828, 109.62903225806463), (359.50000000000023, 106.08064516129048), (366.59677419354853, 148.66129032258073), (334.66129032258073, 148.66129032258073)]]

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

def textureMapGroundFloor():
    #create H_T_G from first frame of sequence
    texture = cv2.imread('Images/ITULogo.jpg')

    fn = "GroundFloorData/sunclipds.avi"
    sequence = cv2.VideoCapture(fn)
    running, frame = sequence.read()

    h_t_g, calibration_points = SIGBTools.getHomographyFromMouse(texture, frame, -4)

    #fig = figure()
    while running:
        running, frame = sequence.read()

        if not running:
            return

        #texture map
        h,w,d = frame.shape
        warped_texture = cv2.warpPerspective(texture, h_t_g,(w, h))
        result = cv2.addWeighted(frame, .8, warped_texture, .2, 50)

        #display
        cv2.imshow("Texture Mapping", result)
        cv2.waitKey(1)

    #run sequence and map texture onto it according to H_T_G

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
    cv2.rectangle(ituMap, p1_map, p2_map, green)
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

def calibrateSharpening():
    frame = cv2.imread("failed_frame_224.png")
    new_frame = sharpen(frame)
    found, _ = cv2.findChessboardCorners(new_frame, (9,6))
    print found
    cv2.imshow("sharpened", new_frame)
    cv2.waitKey(0)

def sharpen(gray):
    #sharpening idea 1: use the laplacian to sharpen up the image
    #sharpen_mask = copy(gray)
    #cv2.cv.Laplace(cv2.cv.fromarray(gray), cv2.cv.fromarray(sharpen_mask), 3)
    #return sharpen_mask + gray

    #sharpening idea 2: subtract a blurred version from the original
    blur = cv2.GaussianBlur(gray, (0,0), 10)
    return cv2.addWeighted(gray, 1.5, blur, -.5, 0)

def textureOnGrid():
    texture = cv2.imread('Images/ITULogo.jpg')
    texture = cv2.pyrDown(texture)
    m,n,d = texture.shape

    fn = "GridVideos/grid1.mp4"
    sequence = cv2.VideoCapture(fn)
    running, frame = sequence.read()    
    pattern_size = (9, 6)
    idx = [0,8,53,45]
    frame_no = 1
    failed = 0
    while running:
        running, frame = sequence.read()
        frame_no += 1
        if not running:            
            print "FAILED: ", failed

        frame = cv2.pyrDown(frame)
        gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        gray = sharpen(gray)
        found, corners = cv2.findChessboardCorners(gray, pattern_size)

        #Define corner points of texture
        ip1 = np.array([[0.,0.],[float(n),0.],[float(n),float(m)],[0.,float(m)]])

        if not found:
            failed += 1
            print "GEMMER LIGE"
            cv2.imwrite("failed_frame_{}.png".format(frame_no), frame)
            continue

        r = []

        c = [(0,0,0),(75,75,75),(125,125,125),(255,255,255)]
        for i,t in enumerate(idx):
            r.append([float(corners[t,0,0]),float(corners[t,0,1])])
            cv2.circle(frame,(int(corners[t,0,0]),int(corners[t,0,1])),10,c[i])

        ip2 = np.array(r)

        h_t_s,mask = cv2.findHomography(ip1, ip2)

        #texture map
        h,w,d = frame.shape
        warped_texture = cv2.warpPerspective(texture, h_t_s,(w, h))
        result = cv2.addWeighted(frame, .6, warped_texture, .4, 50)

        #display
        cv2.imshow("Texture Mapping", result)
        cv2.waitKey(1)

    #run sequence and map texture onto it according to H_T_G



def realisticTexturemap(H_G_M, scale):
    map_img = cv2.imread('Images/ITUMap.bmp')
    point = getMousePointsForImageWithParameter(map_img, 1)[0]

    H_T_M = np.zeros(9).reshape(3,3)
    H_T_M[0][0] = scale
    H_T_M[1][1] = scale

    H_T_M[2][0] = point[0]
    H_T_M[2][0] = point[1]

    H_T_M[2][2] = 1

    #For sjov
    H_T_M = np.identity(3)


    H_M_G = np.linalg.inv(H_G_M)

    H_T_G = np.dot(H_T_M, H_M_G)

    H_T_G = H_T_G / H_T_G[2][2]



    texture = cv2.imread('Images/ITULogo.jpg')

    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)
    #load Tracking data
    running, frame = cap.read()

    h,w,d = frame.shape

    print H_T_G

    warped_texture = cv2.warpPerspective(texture, H_T_G,(w, h))
    result = cv2.addWeighted(frame, .6, warped_texture, .4, 50)

    cv2.imshow("Result", warped_texture)
    cv2.waitKey(0)


def createHomography():
    img1 = cv2.imread('Images/ITUMap.bmp')

    fn = "GroundFloorData/sunclipds.avi"
    cap = cv2.VideoCapture(fn)

    #load Tracking data
    _, img2 = cap.read()

    print SIGBTools.getHomographyFromMouse(img2, img1, 6)



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

def getMousePointsForImageWithParameter(image, points=1):
    '''GUI for inputting 4 points within an images width and height

       image is the image to input points within
    '''
    #Copy image
    drawImage1 = copy(image)

    #Make figure
    fig = figure("Point selection")
    title("Click 4 places on a plane in the image")

    #Show image and request input
    imshow(drawImage1)
    clickPoints = fig.ginput(points, -1)

    #Return points
    return clickPoints

#createHomography()
#showFloorTrackingData()
#simpleTextureMap()
#textureMapGroundFloor()
realisticTexturemap(H, 0.5)
#texturemapGridSequence()
#textureOnGrid()