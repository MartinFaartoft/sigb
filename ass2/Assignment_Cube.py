'''
Created on March 20, 2014

@author: Diako Mardanbegi (dima@itu.dk)
'''
from numpy import *
import numpy as np
from pylab import *
from scipy import linalg
import cv2
import cv2.cv as cv
from SIGBTools import *

def DrawLines(img, points):
    for i in range(1, 17):
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),5)
    return img



def update(img):
    image=copy(img)
    firstViewCorners = False
    if not firstViewCorners:
        first_view = cv2.imread("01.png")
        first_corners = findChessboardOuterCorners(first_view)
        firstViewCorners = True

    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistort the image'''
        image = cv2.undistort(image, cameraMatrix, distortionCoefficient)

    if (ProcessFrame):

        ''' <005> Here Find the Chess pattern in the current frame'''

        this_frame_corners = findChessboardOuterCorners(image)

        patternFound = len(this_frame_corners) != 0


        if patternFound ==True:
            for t in this_frame_corners:
                cv2.circle(image, (int(t[0]), int(t[1])), 10, (255,0,0))
            # Find the homography from first_view to this frame
            h, _ = cv2.findHomography(first_corners, this_frame_corners)

            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''
                #H_cs_1

            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''

                cv2.putText(image,str("frame:" + str(frameNumber)), (20,10),cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255))#Draw the text

            ''' <008> Here Draw the world coordinate system in the image'''

            if TextureMap:

                ''' <010> Here Do he texture mapping and draw the texture on the faces of the cube'''

                ''' <012>  calculate the normal vectors of the cube faces and draw these normal vectors on the center of each face'''

                ''' <013> Here Remove the hidden faces'''


            if ProjectPattern:
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points'''



            if WireFrame:
                ''' <009> Here Project the box into the current camera image and draw the box edges'''

    cv2.namedWindow('Web cam')
    cv2.imshow('Web cam', image)
    global result
    result=copy(image)

def findChessboardOuterCorners(img):
    idx = [0,8,45,53]
    pattern_size = (9, 6)

    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(gray, pattern_size)

    points = []

    if not found:
        return np.array([])

    for index in idx:
        points.append([float(corners[index, 0, 0]),float(corners[index, 0, 1])])

    return np.array(points)





def getImageSequence(capture, fastForward):
    '''Load the video sequence (fileName) and proceeds, fastForward number of frames.'''
    global frameNumber

    for t in range(fastForward):
        isSequenceOK, originalImage = capture.read()  # Get the first frames
        frameNumber = frameNumber+1
    return originalImage, isSequenceOK


def printUsage():
    print "Q or ESC: Stop"
    print "SPACE: Pause"
    print "p: turning the processing on/off "
    print 'u: undistorting the image'
    print 'g: project the pattern using the camera matrix (test)'
    print 'x: your key!'

    print 'the following keys will be used in the next assignment'
    print 'i: show info'
    print 't: texture map'
    print 's: save frame'



def run(speed,video):

    '''MAIN Method to load the image sequence and handle user inputs'''

    #--------------------------------video
    capture = cv2.VideoCapture(video)


    image, isSequenceOK = getImageSequence(capture,speed)

    if(isSequenceOK):
        update(image)
        printUsage()

    while(isSequenceOK):
        OriginalImage=copy(image)


        inputKey = cv2.waitKey(1)

        if inputKey == 32:#  stop by SPACE key
            update(OriginalImage)
            if speed==0:
                speed = tempSpeed;
            else:
                tempSpeed=speed
                speed = 0;

        if (inputKey == 27) or (inputKey == ord('q')):#  break by ECS key
            break

        if inputKey == ord('p') or inputKey == ord('P'):
            global ProcessFrame
            if ProcessFrame:
                ProcessFrame = False;

            else:
                ProcessFrame = True;
            update(OriginalImage)

        if inputKey == ord('u') or inputKey == ord('U'):
            global Undistorting
            if Undistorting:
                Undistorting = False;
            else:
                Undistorting = True;
            update(OriginalImage)
        if inputKey == ord('w') or inputKey == ord('W'):
            global WireFrame
            if WireFrame:
                WireFrame = False;

            else:
                WireFrame = True;
            update(OriginalImage)

        if inputKey == ord('i') or inputKey == ord('I'):
            global ShowText
            if ShowText:
                ShowText = False;

            else:
                ShowText = True;
            update(OriginalImage)

        if inputKey == ord('t') or inputKey == ord('T'):
            global TextureMap
            if TextureMap:
                TextureMap = False;

            else:
                TextureMap = True;
            update(OriginalImage)

        if inputKey == ord('g') or inputKey == ord('G'):
            global ProjectPattern
            if ProjectPattern:
                ProjectPattern = False;

            else:
                ProjectPattern = True;
            update(OriginalImage)

        if inputKey == ord('x') or inputKey == ord('X'):
            global debug
            if debug:
                debug = False;
            else:
                debug = True;
            update(OriginalImage)


        if inputKey == ord('s') or inputKey == ord('S'):
            name='Saved Images/Frame_' + str(frameNumber)+'.png'
            cv2.imwrite(name,result)

        if (speed>0):
            update(image)
            image, isSequenceOK = getImageSequence(capture,speed)

def loadCalibrationData():
    global translationVectors
    translationVectors = np.load('numpyData/translationVectors.npy')
    global cameraMatrix
    cameraMatrix = np.load('numpyData/camera_matrix.npy')
    global rotatioVectors
    rotatioVectors = np.load('numpyData/rotatioVectors.npy')
    global distortionCoefficient
    distortionCoefficient = np.load('numpyData/distortionCoefficient.npy')
    return cameraMatrix,rotatioVectors[0],translationVectors[0]

def calculateP(K,r,t):
    R,_ = cv2.Rodrigues(r)
    Rt = np.hstack((R,t))
    P = np.dot(K,Rt)
    return P

def displayNumpyPoints(C):
    points = np.load('numpyData/obj_points.npy')
    img = cv2.imread('01.png')

    X = points[0]
    ones = np.ones((X.shape[0],1))
    X =np.column_stack((X,ones)).T

    x = C.project(X)
    x = x.T

    for p in x:
        C = int(p[0]),int(p[1])
        cv2.circle(img,C, 2,(255,0,255),4)

    cv2.imshow('result',img)
    cv2.waitKey(0)

def projectChessBoardPoints(C, points):

    X = points[0]
    ones = np.ones((X.shape[0],1))
    X =np.column_stack((X,ones)).T

    x = C.project(X)
    x = x.T

    return x



'''-------------------MAIN BODY--------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------------------------------'''




'''-------variables------'''
global cameraMatrix
cameraMatrix = None
global distortionCoefficient
global homographyPoints
global calibrationPoints
global calibrationCamera
global chessSquare_size

ProcessFrame=False
Undistorting=False
WireFrame=False
ShowText=True
TextureMap=True
ProjectPattern=False
debug=True

tempSpeed=1
frameNumber=0
chessSquare_size=2



'''-------defining the cube------'''

box = getCubePoints([4, 2.5, 0], 1,chessSquare_size)


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim
j = array([ [0,3,2,1],[0,3,2,1] ,[0,3,2,1]  ])  # indices for the second dim
TopFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim
j = array([ [3,8,7,2],[3,8,7,2] ,[3,8,7,2]  ])  # indices for the second dim
RightFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim
j = array([ [5,0,1,6],[5,0,1,6] ,[5,0,1,6]  ])  # indices for the second dim
LeftFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim
j = array([ [5,8,3,0], [5,8,3,0] , [5,8,3,0] ])  # indices for the second dim
UpFace = box[i,j]


i = array([ [0,0,0,0],[1,1,1,1] ,[2,2,2,2]  ])  # indices for the first dim
j = array([ [1,2,7,6], [1,2,7,6], [1,2,7,6] ])  # indices for the second dim
DownFace = box[i,j]



'''----------------------------------------'''
'''----------------------------------------'''



''' <000> Here Call the calibrateCamera from the SIGBTools to calibrate the camera and saving the data'''
#calibrateCamera(5, (9,6), 2.0, 0)
''' <001> Here Load the numpy data files saved by the cameraCalibrate2'''
K,r,t = loadCalibrationData()
print "LOADED"
print cameraMatrix
''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2'''

P = calculateP(K,r,t)
C = Camera(P)


#displayNumpyPoints(C)

''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation'''



# Load 01.png
# Find four points in 01.png
# Find corresponding points in world coordinate chessboard
# FindHomography on these H_cs^1

''' <003a> Find homography H_cs^1 '''
idx = [0,8,45,53]
points_from_chess_board_plane = np.load('numpyData/obj_points.npy') 
points_from_first_view_plane = projectChessBoardPoints(C,points_from_chess_board_plane)

p1 = []
p2 = []
for i in idx:
    x11 = points_from_chess_board_plane[0][i][0]
    y12 = points_from_chess_board_plane[0][i][1]
    p1.append([float(x11),float(y12)])
    x21 = points_from_first_view_plane[i][0]
    y22 = points_from_first_view_plane[i][1]
    p2.append([float(x21),float(y22)])
p1 = np.array(p1)
p2 = np.array(p2)
H_cs_1,_ = cv2.findHomography(p1,p2)


# TODO find out what this means. Part in assignment Augmentation. 1.






run(1,0) #run(1,"Pattern.avi")
