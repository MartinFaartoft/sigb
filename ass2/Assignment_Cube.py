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
import math

def DrawLines(img, points):
    for i in range(1, len(points[0])):
         x1 = points[0, i - 1]
         y1 = points[1, i - 1]
         x2 = points[0, i]
         y2 = points[1, i]
         cv2.line(img, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0),5)
    return img

def findChessBoardCorners(image):
    pattern_size = (9, 6)
    flag = cv2.CALIB_CB_FAST_CHECK + cv2.cv.CV_CALIB_CB_NORMALIZE_IMAGE
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    return cv2.findChessboardCorners(gray, pattern_size, flags=flag)

def update(img):
    image=copy(img)

    if Undistorting:  #Use previous stored camera matrix and distortion coefficient to undistort the image
        ''' <004> Here Undistort the image'''
        image = cv2.undistort(image, cameraMatrix, distortionCoefficient)

    if (ProcessFrame):
        ''' <005> Here Find the Chess pattern in the current frame'''
        patternFound, corners = findChessBoardCorners(image)

        if patternFound == True:
            ''' <006> Here Define the cameraMatrix P=K[R|t] of the current frame'''
            if debug:
                P = findPFromHomography(corners)
            else:
                P = createPCurrentFromObjectPose(corners)

            if ShowText:
                ''' <011> Here show the distance between the camera origin and the world origin in the image'''

                cv2.putText(image,str("frame:" + str(frameNumber)), (20,10),cv2.FONT_HERSHEY_PLAIN,1, (255, 255, 255))#Draw the text

            ''' <008> Here Draw the world coordinate system in the image'''
            cam2 = Camera(P)
            coordinate_system = getCoordinateSystemChessPlane()
            transformed_coordinate_system = projectChessBoardPoints(cam2,coordinate_system)
            drawCoordinateSystem(image,transformed_coordinate_system)

            if TextureMap:

                ''' <010> Here Do he texture mapping and draw the texture on the faces of the cube'''

                ''' <012>  calculate the normal vectors of the cube faces and draw these normal vectors on the center of each face'''

                ''' <013> Here Remove the hidden faces'''


            if ProjectPattern:
                ''' <007> Here Test the camera matrix of the current view by projecting the pattern points'''
                cam2 = Camera(P)
                X = projectChessBoardPoints(cam2, points_from_chess_board_plane)

                for p in X:
                    C = int(p[0]),int(p[1])
                    cv2.circle(image,C, 2,(255,0,255),4)


            if WireFrame:
                ''' <009> Here Project the box into the current camera image and draw the box edges'''
                cam2 = Camera(P)
                if (Teapot):
                    teapot = parse_teapot()

                    #rotated_box = rotateFigure(box, 0, 0, angle, scale, scale, scale)
                    drawObjectScatter(cam2, image, teapot)
                ## stop
                else:
                    #figure = box if frameNumber % 100 < 50 else pyramid

                    angle = frameNumber * (math.pi / 50.0)
                    scale = 2 + math.sin(angle)
                    box1 = rotateFigure(box, 0, 0, -angle, 1, 1, 1)
                    box2 = getPyramidPoints([0, 0, -1], 1,chessSquare_size)
                    box2 = rotateFigure(box2, 0, 0, angle, 1, 1, 1)
                    #rotated_box = rotateFigure(figure, 0, 0, angle, scale, scale, scale)
                    drawFigure(image, cam2, box1)
                    drawFigure(image, cam2, box2)

    cv2.namedWindow('Web cam')
    cv2.imshow('Web cam', image)
    videoWriter.write(image)
    global result
    result=copy(image)


def drawFigure(image, camera, figure):
    X = figure.T
    ones = np.ones((X.shape[0],1))
    X = np.column_stack((X,ones)).T

    projected_figure = camera.project(X)
    DrawLines(image,projected_figure)

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

    resultFile = "recording.avi"



    image, isSequenceOK = getImageSequence(capture,speed)

    imSize = np.shape(image)
    global videoWriter
    videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 30.0,(imSize[1],imSize[0]),True) #Make a video writer

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

        if inputKey == ord('l') or inputKey == ord('L'):
            global Teapot
            Teapot = not Teapot
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
    global points_from_chess_board_plane
    points_from_chess_board_plane = np.load('numpyData/obj_points.npy')[0]
    return cameraMatrix,rotatioVectors[0],translationVectors[0]

def calculateP(K,r,t):
    R,_ = cv2.Rodrigues(r)
    Rt = np.hstack((R,t))
    P = np.dot(K,Rt)
    return P

def displayNumpyPoints(C):
    points = np.load('numpyData/obj_points.npy')

    img = cv2.imread('01.png')

    x = projectChessBoardPoints(C,points[0])

    for p in x:
        C = int(p[0]),int(p[1])
        cv2.circle(img,C, 2,(255,0,255),4)

    cv2.imshow('result',img)
    cv2.waitKey(0)

def projectChessBoardPoints(C, X):
    ones = np.ones((X.shape[0],1))
    X = np.column_stack((X,ones)).T
    x = C.project(X)
    x = x.T
    return x

def getCoordinateSystemChessPlane(axis_length = 2.0):
    o = [0., 0., 0.]
    x = [axis_length, 0., 0.]
    y = [0., axis_length, 0.]
    z = [0., 0., -axis_length] #positive z is away from camera, by default
    return np.array([o,x,y,z])

def drawObjectScatter(C,img, points):
    points = points.T
    ones = np.ones((points.shape[0],1))
    points = np.column_stack((points,ones)).T
    points = C.project(points)
    points = points.T

    for point in points:
        cv2.circle(img, (int(point[0]),int(point[1])), 3, (0, 255, 0), -1)


def drawCoordinateSystem(img, coordinate_system):
    o = coordinate_system[0]
    x = coordinate_system[1]
    y = coordinate_system[2]
    z = coordinate_system[3]

    cv2.line(img, (int(o[0]),int(o[1])), (int(x[0]),int(x[1])),(255, 0, 0),3)
    cv2.line(img, (int(o[0]),int(o[1])), (int(y[0]),int(y[1])), (255, 0, 0),3)
    cv2.line(img, (int(o[0]),int(o[1])), (int(z[0]),int(z[1])), (255, 0, 0),3)

    cv2.circle(img, (int(x[0]),int(x[1])), 3, (0, 255, 0), -1)
    cv2.circle(img, (int(y[0]),int(y[1])), 3, (0, 255, 0), -1)
    cv2.circle(img, (int(z[0]),int(z[1])), 3, (0, 255, 0), -1)
    cv2.circle(img, (int(o[0]),int(o[1])), 3, (0, 0, 255), -1)

def createPCurrentFromObjectPose(corners):
    found, r_vec, t_vec = cv2.solvePnP(points_from_chess_board_plane, corners, cameraMatrix, distortionCoefficient)
    return calculateP(cameraMatrix, r_vec, t_vec)

def findPFromHomography(corners):
    first_view = cv2.imread("01.png")
    _, first_all_corners = findChessBoardCorners(first_view)
    first_corners = findChessboardOuterCorners(first_all_corners)

    this_frame_outer_corners = findChessboardOuterCorners(corners)
    # for t in this_frame_corners:
    #     cv2.circle(image, (int(t[0]), int(t[1])), 10, (255,0,0))

    # Find the homography from first_view to this frame
    H_1_2, _ = cv2.findHomography(first_corners, this_frame_outer_corners)

    H_cs_2 = np.dot(H_1_2, H_cs_1)

    A = np.dot(np.linalg.inv(K), H_cs_2) # A = K^-1 * H(cs_2)

    R = np.array([A[:,0],A[:,1],np.cross(A[:,0],A[:,1])]).T # R = [r_1, r_1, r_1 x r_2]^T

    t = np.array([A[:,2]]).T

    Rt = np.hstack((R, t)) # Rt = [R|t]

    P2_Method1 = np.dot(K,Rt)

    return P2_Method1

def findChessboardOuterCorners(corners): #TODO remove? since it's kinda dumb to filter out useful points before calculating H
    idx = [0,8,45,53]

    points = []

    for index in idx:
        points.append([float(corners[index, 0, 0]),float(corners[index, 0, 1])])

    return np.array(points)

def findHomographyFromCSto1():
    # Load 01.png
    # Find four points in 01.png
    # Find corresponding points in world coordinate chessboard
    # FindHomography on these H_cs^1
    idx = [0,8,45,53]
    points_from_first_view_plane = projectChessBoardPoints(C,points_from_chess_board_plane)

    p1 = []
    p2 = []
    for i in idx:
        x11 = points_from_chess_board_plane[i][0]
        y12 = points_from_chess_board_plane[i][1]

        p1.append([float(x11),float(y12)])
        x21 = points_from_first_view_plane[i][0]
        y22 = points_from_first_view_plane[i][1]
        p2.append([float(x21),float(y22)])

    p1 = np.array(p1)
    p2 = np.array(p2)
    H_cs_1,_ = cv2.findHomography(p1,p2)

    return H_cs_1

def homographyTest(img, H):
    # Load first view
    if img == None:
        img = cv2.imread("01.png")

    # Load chess world points
    # Project chess world points according to H
    points_from_chess_board_plane = points_from_chess_board_plane
    points_from_chess_board_plane[:,2] = 1.0

    for point in points_from_chess_board_plane:
        point = np.dot(H, point)
        point = point / point[2]
        cv2.circle(img, (int(point[0]), int(point[1])), 10, (255,0,0))

    cv2.imshow("LOL", img)
    cv2.waitKey(0)

    # Draw

    # Profit

def parse_teapot():
    points = []
    with open("teapot.data", "r") as infile:
        lines = infile.read().splitlines()
        for line in lines:
            line = line.split(",")

            x = float(line[0]) + 5
            y = float(line[1]) + 5
            z = (float(line[2]) * -1) - 5
            points.append([x, y, z])

    result =  np.array(points).T
    return result * 2

def rotateFigure(figure, theta_x, theta_y, theta_z, scale_x, scale_y, scale_z):
    translate_to = [8, 6, -1]


    rotation_matrix_x = np.array([ [1, 0, 0], [0, cos(theta_x), -sin(theta_x)], [0, sin(theta_x), cos(theta_x)] ])
    rotation_matrix_y = np.array([ [cos(theta_y), 0, sin(theta_y)], [0, 1, 0], [-sin(theta_y), 0, cos(theta_y)] ])
    rotation_matrix_z = np.array([ [cos(theta_z), -sin(theta_z), 0], [sin(theta_z), cos(theta_z), 0], [0, 0, 1] ])

    rotated_x = []
    rotated_y = []
    rotated_z = []
    rotation = np.dot(rotation_matrix_y, np.dot(rotation_matrix_y, rotation_matrix_z))
    for i in range(len(figure[0])):
        p = np.array([figure[0][i], figure[1][i], figure[2][i]])
        p_rot = np.dot(rotation, p)
        rotated_x.append(scale_x * p_rot[0] + translate_to[0])
        rotated_y.append(scale_y * p_rot[1] + translate_to[1])
        rotated_z.append(scale_z * p_rot[2] + translate_to[2])


    result = np.array([rotated_x, rotated_y, rotated_z])
    return result


def getPyramidPoints(center, size,chessSquare_size):
    points = []

    tl = [center[0]-size, center[1]-size, center[2]]
    bl = [center[0]-size, center[1]+size, center[2]]
    br = [center[0]+size, center[1]+size, center[2]]
    tr = [center[0]+size, center[1]-size, center[2]]
    top = [center[0], center[1], center[2] - size * 2]

    #bottom
    points.append(tl)
    points.append(bl)
    points.append(br)
    points.append(tr)
    points.append(tl)

    #top
    points.append(top)

    #diagonals
    points.append(bl)
    points.append(br)
    points.append(top)
    points.append(tr)
    points=dot(points,chessSquare_size)
    return array(points).T



'''-------------------MAIN BODY--------------------------------------------------------------------'''
'''--------------------------------------------------------------------------------------------------------------'''



'''-------variables------'''
global cameraMatrix
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
Teapot = True

tempSpeed=1
frameNumber=0
chessSquare_size=2



'''-------defining the figures------'''

box = getCubePoints([0, 0, 1], 1,chessSquare_size)
pyramid = getPyramidPoints([0, 0, 1], 1,chessSquare_size)


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
''' <002> Here Define the camera matrix of the first view image (01.png) recorded by the cameraCalibrate2'''

P = calculateP(K,r,t)
C = Camera(P)

''' <003> Here Load the first view image (01.png) and find the chess pattern and store the 4 corners of the pattern needed for homography estimation'''
#displayNumpyPoints(C)


''' <003a> Find homography H_cs^1 '''
H_cs_1 = findHomographyFromCSto1()

#homographyTest(H_cs_1)
run(1,"sequence.mov")
