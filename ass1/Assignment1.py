import cv2
import cv
import pylab
import math
from SIGBTools import RegionProps
from SIGBTools import getLineCoordinates
from SIGBTools import ROISelector
from SIGBTools import getImageSequence
from SIGBTools import getCircleSamples
import SIGBTools
import numpy as np
import sys
from scipy.cluster.vq import *
from scipy.misc import *
from matplotlib.pyplot import *



inputFile = "Sequences/eye1.avi"
outputFile = "eyeTrackerResult.mp4"

#seems to work okay for eye1.avi
default_pupil_threshold = 25 #93

#--------------------------
#         Global variable
#--------------------------
global imgOrig,leftTemplate,rightTemplate,frameNr
imgOrig = [];
#These are used for template matching
leftTemplate = []
rightTemplate = []
frameNr =0;


def GetPupil(gray,thr, min_val, max_val):
	'''Given a gray level image, gray and threshold value return a list of pupil locations'''
	#tempResultImg = cv2.cvtColor(gray,cv2.COLOR_GRAY2BGR) #used to draw temporary results

	#Threshold image to get a binary image
	gray = cv2.equalizeHist(gray)
	cv2.imshow("TempResults", gray)
	val,binI =cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
	#print val
	#Morphology (close image to remove small 'holes' inside the pupil area)
	st = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
	#binI = cv2.morphologyEx(binI, cv2.MORPH_OPEN, st, iterations=10)
	#binI = cv2.morphologyEx(binI, cv2.MORPH_CLOSE, st, iterations=5)
	cv2.imshow("Threshold",binI)
	#cv2.moveWindow("Threshold", 800, 0)
	#Calculate blobs, and do edge detection on entire image (modifies binI)
	contours, hierarchy = cv2.findContours(binI, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)


	pupils = [];
	prop_calc = RegionProps()
	centroids = []
	for contour in contours:
		#calculate centroid, area and 'extend' (compactness of contour)
		props = prop_calc.CalcContourProperties(contour, ["centroid", "area", "extend"])
		x, y = props["Centroid"]
		area = props["Area"]
		extend = props["Extend"]
		#filter contours, so that their area lies between min_val and max_val, and then extend lies between 0.4 and 1.0
		if (area > min_val and area < max_val and extend > 0.5 and extend < 1.0):
			pupilEllipse = cv2.fitEllipse(contour)
			# center, radii, angle = pupilEllipse
			# max_radius = max(radii)
			# c_x = int(center[0])
			# c_y = int(center[1])
			# cv2.circle(tempResultImg,(c_x,c_y), int(max_radius), (0,0,255),4) #draw a circle
			#cv2.ellipse(tempResultImg, pupilEllipse,(0,255,0),1)
			pupils.append(pupilEllipse)

	#cv2.imshow("TempResults",tempResultImg)

	return pupils

def GetGlints(gray,thr):
	min_area = 10
	max_area = 150
	''' Given a gray level image, gray and threshold
	value return a list of glint locations'''
	#print thr
	val, binary_image = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY)


	st = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, st, iterations=8)
	binary_image = cv2.morphologyEx(binary_image, cv2.MORPH_OPEN, st, iterations=2)


	#cv2.imshow("Threshold", binary_image)
	#Calculate blobs, and do edge detection on entire image (modifies binI)
	contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	glints = [];
	prop_calc = RegionProps()
	centroids = []
	for contour in contours:
		#calculate centroid, area and 'extend' (compactness of contour)
		props = prop_calc.CalcContourProperties(contour, ["centroid", "area", "extend"])
		x, y = props["Centroid"]
		area = props["Area"]
		extend = props["Extend"]

		#filter contours, so that their area lies between min_val and max_val, and then extend lies between 0.4 and 1.0
		if area > min_area and area < max_area: #and extend > 0.4 and extend < 1.0):
			glints.append((x,y))
			#print x, y, area, extend
			#cv2.circle(tempResultImg,(int(x),int(y)), 2, (0,0,255),4) #draw a circle
	#cv2.imshow("TempResults",tempResultImg)

	return glints

def GetIrisUsingThreshold(gray, thr, min_val, max_val):
	''' Given a gray level image, gray and threshold
	value return a list of iris locations'''
	val,binary_image = cv2.threshold(gray, thr, 255, cv2.THRESH_BINARY_INV)
	cv2.imshow("Threshold", binary_image)

	contours, hierarchy = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

	irises = [];
	prop_calc = RegionProps()
	centroids = []
	for contour in contours:
		#calculate centroid, area and 'extend' (compactness of contour)
		props = prop_calc.CalcContourProperties(contour, ["centroid", "area", "extend"])
		x, y = props["Centroid"]
		area = props["Area"]
		extend = props["Extend"]
		#filter contours, so that their area lies between min_val and max_val, and then extend lies between 0.4 and 1.0
		if area > min_val and area < max_val and extend > 0.5 and extend < 1.0:
			irisEllipse = cv2.fitEllipse(contour)
			# center, radii, angle = pupilEllipse
			# max_radius = max(radii)
			# c_x = int(center[0])
			# c_y = int(center[1])
			# cv2.circle(tempResultImg,(c_x,c_y), int(max_radius), (0,0,255),4) #draw a circle
			#cv2.ellipse(tempResultImg, pupilEllipse,(0,255,0),1)
			irises.append(irisEllipse)
	return irises

def circularHough(gray):
	''' Performs a circular hough transform of the image, gray and shows the  detected circles
	The circe with most votes is shown in red and the rest in green colors '''
 #See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCircles
	blur = cv2.GaussianBlur(gray, (31,31), 11)

	dp = 6; minDist = 30
	highThr = 20 #High threshold for canny
	accThr = 850; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	maxRadius = 50;
	minRadius = 155;
	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,maxRadius, minRadius)

	#Make a color image from gray for display purposes
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	if (circles !=None):
	 #print circles
	 all_circles = circles[0]
	 M,N = all_circles.shape
	 k=1
	 for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			K=k+1
	 c=all_circles[0,:]
	 cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255),5)
	 cv2.imshow("hough",gColor)

def GetIrisUsingNormals(gray,pupil,normalLength):
	''' Given a gray level image, gray and the length of the normals, normalLength
	 return a list of iris locations'''
	# YOUR IMPLEMENTATION HERE !!!!
	pass

def GetIrisUsingSimplifyedHough(gray,pupil):
	''' Given a gray level image, gray
	return a list of iris locations using a simplified Hough transformation'''
	# YOUR IMPLEMENTATION HERE !!!!
	pass

def plotVectorField(I):
	g_x = cv2.Sobel(I, cv.CV_64F, 1,0, ksize=3)
	g_y = cv2.Sobel(I, cv.CV_64F, 0,1, ksize=3) # ksize=3 som **kwargs

	x_orig_dim, y_orig_dim = I.shape
	x_mesh_dim, y_mesh_dim = (3, 3)

	sample_g_x = g_x[0:x_orig_dim:x_mesh_dim,0:y_orig_dim:x_mesh_dim]
	sample_g_y = g_y[0:x_orig_dim:x_mesh_dim,0:y_orig_dim:x_mesh_dim]

	quiver(sample_g_x, sample_g_y)
	show()


def getGradientImageInfo(I):
	g_x = cv2.Sobel(I, cv.CV_64F, 1,0)
	g_y = cv2.Sobel(I, cv.CV_64F, 0,1) # ksize=3 som **kwargs

	X,Y = I.shape
	orientation = np.zeros(I.shape)
	magnitude = np.zeros(I.shape)
	sq_g_x = cv2.pow(g_x, 2)
	sq_g_y = cv2.pow(g_y, 2)
	fast_magnitude = cv2.pow(sq_g_x + sq_g_y, .5)

	# for x in range(X):
	# 	for y in range(Y):
	# 		#orientation[x][y] = np.arctan2(g_y[x][y], g_x[x][y]) * (180 / math.pi)
	# 		magnitude[x][y] = math.sqrt(g_y[x][y] ** 2 + g_x[x][y] ** 2)


	#print fast_magnitude[0]
	#print magnitude[0]

	return fast_magnitude,orientation

def GetEyeCorners(orig_img, leftTemplate, rightTemplate,pupilPosition=None):
	if leftTemplate != [] and rightTemplate != []:
		ccnorm_left = cv2.matchTemplate(orig_img, leftTemplate, cv2.TM_CCOEFF_NORMED)
		ccnorm_right = cv2.matchTemplate(orig_img, rightTemplate, cv2.TM_CCOEFF_NORMED)

		minVal, maxVal, minLoc, maxloc_left_from = cv2.minMaxLoc(ccnorm_left)
		minVal, maxVal, minLoc, maxloc_right_from, = cv2.minMaxLoc(ccnorm_right)

		l_x,l_y = leftTemplate.shape
		max_loc_left_from_x = maxloc_left_from[0]
		max_loc_left_from_y = maxloc_left_from[1]

		max_loc_left_to_x = max_loc_left_from_x + l_x
		max_loc_left_to_y = max_loc_left_from_y + l_y

		maxloc_left_to = (max_loc_left_to_x, max_loc_left_to_y)

		r_x,r_y = leftTemplate.shape
		max_loc_right_from_x = maxloc_right_from[0]
		max_loc_right_from_y = maxloc_right_from[1]

		max_loc_right_to_x = max_loc_right_from_x + r_x
		max_loc_right_to_y = max_loc_right_from_y + r_y
		maxloc_right_to = (max_loc_right_to_x, max_loc_right_to_y)

		return (maxloc_left_from, maxloc_left_to, maxloc_right_from, maxloc_right_to)

bgr_yellow = (0,255,255)
bgr_blue = 255, 0, 0
def circleTest(img, center_point):
	nPts = 20
	circleRadius = 100
	P = getCircleSamples(center=center_point, radius=circleRadius, nPoints=nPts)
	for (x,y,dx,dy) in P:
		point_coords = (int(x),int(y))
		cv2.circle(img, point_coords, 2, rgb_yellow, 2)
		cv2.line(img, point_coords, center_point, rgb_yellow)

def findEllipseContour(img, gradient_magnitude, estimatedCenter, estimatedRadius, nPts=30):
	P = getCircleSamples(center = estimatedCenter, radius = estimatedRadius, nPoints=nPts)
	newPupil = np.zeros((nPts,1,2)).astype(np.float32)
	t = 0
	for (x,y,dx,dy) in P:
		#< define normalLength as some maximum distance away from initial circle >
		#< get the endpoints of the normal -> p1,p2>
		point_coords = (int(x),int(y))
		#cv2.circle(img, point_coords, 2, bgr_blue, 2)
		center_point_coords = (int(estimatedCenter[0]), int(estimatedCenter[1]))
		max_point = findMaxGradientValueOnNormal(gradient_magnitude, point_coords, center_point_coords)
		cv2.circle(img, tuple(max_point), 2, bgr_yellow, 1)
		#< store maxPoint in newPupil>
		newPupil[t] = max_point
		t += 1
	#<fitPoints to model using least squares- cv2.fitellipse(newPupil)>
	return cv2.fitEllipse(newPupil)

def findMaxGradientValueOnNormal(gradient_magnitude, p1, p2):
    #Get integer coordinates on the straight line between p1 and p2
	pts = SIGBTools.getLineCoordinates(p1, p2)
	values = gradient_magnitude[pts[:,1],pts[:,0]]
	max_index = np.argmax(values)
	#Find index of max value in normalVals
	return pts[max_index]
	#return coordinate of max value in image coordinates

def FilterPupilGlint(pupils,glints):
	''' Given a list of pupil candidates and glint candidates returns a list of pupil and glints'''
	filtered_glints = []
	filtered_pupils = pupils
	for glint in glints:
		for pupil in pupils:
			if (is_glint_close_to_pupil(glint, pupil)):
				filtered_glints.append(glint)

	return filtered_pupils, filtered_glints

def is_glint_close_to_pupil(glint, pupil):
	center, radii, angle = pupil
	max_radius = max(radii)
	distance = euclidianDistance(center, glint)
	return (distance < max_radius)

def filterGlintsIris(glints, irises):
	new_glints = []
	if glints and irises:
		for glint in glints:
			for iris in irises:
				iris_x, irix_y, iris_radius = iris
				print glint
				iris_vector = np.array([iris_x, irix_y])
				distance = np.linalg.norm(glint - iris_vector)
				if distance < iris_radius:
					new_glints.append(glint)
				#print iris
	return new_glints



def update(I):
	'''Calculate the image features and display the result based on the slider values'''
	#global drawImg
	global frameNr,drawImg, gray
	img = I.copy()
	sliderVals = getSliderVals()
	gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)

	# Do the magic
	pupils = GetPupil(gray,sliderVals['pupilThr'], sliderVals['minSize'], sliderVals['maxSize'])
	glints = GetGlints(gray,sliderVals['glintThr'])
	#pupils, glints = FilterPupilGlint(pupils,glints)
	#irises = GetIrisUsingThreshold(gray, sliderVals['pupilThr'], sliderVals['minSize'], sliderVals['maxSize'])

	magnitude, orientation = getGradientImageInfo(gray)

	#plotVectorField(gray)
	#Do template matching
	global leftTemplate
	global rightTemplate

	#corners = GetEyeCorners(gray, leftTemplate, rightTemplate)

	#detectPupilHough(gray, 100)
	#irises = detectIrisHough(gray, 400)

	#glints = filterGlintsIris(glints,irises)

	#Display results
	global frameNr,drawImg
	x,y = 10,10
	#setText(img,(x,y),"Frame:%d" %frameNr)
	sliderVals = getSliderVals()

	# for non-windows machines we print the values of the threshold in the original image
	if sys.platform != 'win32':
		step=18
		cv2.putText(img, "pupilThr :"+str(sliderVals['pupilThr']), (x, y+step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
		cv2.putText(img, "glintThr :"+str(sliderVals['glintThr']), (x, y+2*step), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)
	cv2.imshow('Result',img)

	#Uncomment these lines as your methods start to work to display the result in the
	#original image

	for pupil in pupils:
		cv2.ellipse(img,pupil,(0,255,0),1)
		C = int(pupil[0][0]),int(pupil[0][1])
		cv2.circle(img,C, 2, (0,0,255),1)
		#def findEllipseContour(img, gradient_magnitude, estimatedCenter, estimatedRadius, nPts=30):
		#contour = findEllipseContour(img, magnitude, C, 80)
		#cv2.ellipse(img, contour, bgr_yellow, 1)
		#circleTest(img, C)
	for glint in glints:
	    C = int(glint[0]),int(glint[1])
	    #cv2.circle(img,C, 2,(255,0,255),5)



	# if corners:
	# 	left_from, left_to, right_from, right_to = corners
	# 	cv2.rectangle(img, left_from , left_to, 255)
	# 	cv2.rectangle(img, right_from , right_to, 255)

	# for iris in irises:
	# 	cv2.ellipse(img,iris,(0,255,0),1)
	# 	C = int(iris[0][0]),int(iris[0][1])
	# 	cv2.circle(img,C, 2, (0,0,255),4)

	cv2.imshow("Result", img)

		#For Iris detection - Week 2
		#circularHough(gray)

	#copy the image so that the result image (img) can be saved in the movie
	drawImg = img.copy()


def printUsage():
	print "Q or ESC: Stop"
	print "SPACE: Pause"
	print "r: reload video"
	print 'm: Mark region when the video has paused'
	print 's: toggle video  writing'
	print 'c: close video sequence'

def run(fileName,resultFile='eyeTrackingResults.avi'):

	''' MAIN Method to load the image sequence and handle user inputs'''
	global imgOrig, frameNr,drawImg, leftTemplate, rightTemplate, gray
	setupWindowSliders()
	props = RegionProps();
	cap,imgOrig,sequenceOK = getImageSequence(fileName)
	videoWriter = 0

	frameNr =0
	if(sequenceOK):
		update(imgOrig)
	printUsage()
	frameNr=0;
	saveFrames = False

	while(sequenceOK):
		sliderVals = getSliderVals();
		frameNr=frameNr+1
		ch = cv2.waitKey(1)
		#Select regions
		if(ch==ord('m')):
			if(not sliderVals['Running']):
				roiSelect=ROISelector(imgOrig)
				pts,regionSelected= roiSelect.SelectArea('Select eye corner',(400,200))
				if(regionSelected):
					if leftTemplate == []:
						leftTemplate = gray[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]
					else:
						rightTemplate = gray[pts[0][1]:pts[1][1],pts[0][0]:pts[1][0]]

		if ch == 27:
			break
		if (ch==ord('s')):
			if((saveFrames)):
				videoWriter.release()
				saveFrames=False
				print "End recording"
			else:
				imSize = np.shape(imgOrig)
				videoWriter = cv2.VideoWriter(resultFile, cv.CV_FOURCC('D','I','V','3'), 15.0,(imSize[1],imSize[0]),True) #Make a video writer
				saveFrames = True
				print "Recording..."



		if(ch==ord('q')):
			break
		if(ch==32): #Spacebar
			sliderVals = getSliderVals()
			cv2.setTrackbarPos('Stop/Start','Controls',not sliderVals['Running'])
		if(ch==ord('r')):
			frameNr =0
			sequenceOK=False
			cap,imgOrig,sequenceOK = getImageSequence(fileName)
			update(imgOrig)
			sequenceOK=True

		sliderVals=getSliderVals()
		if(sliderVals['Running']):
			sequenceOK, imgOrig = cap.read()
			if(sequenceOK): #if there is an image
				update(imgOrig)
			if(saveFrames):
				videoWriter.write(drawImg)
	if(videoWriter!=0):
		videoWriter.release()
        print "Closing videofile..."
#------------------------

def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
	''' Detects the pupil in the image, gray, using k-means
		gray              : grays scale image
		K                 : Number of clusters
		distanceWeight    : Defines the weight of the position parameters
		reSize            : the size of the image to do k-means on
	'''
	#Resize for faster performance
	smallI = cv2.resize(gray, reSize)
	M,N = smallI.shape
	#Generate coordinates in a matrix
	X,Y = np.meshgrid(range(M),range(N))
	#Make coordinates and intensity into one vectors
	z = smallI.flatten()
	x = X.flatten()
	y = Y.flatten()
	O = len(x)
	#make a feature vectors containing (x,y,intensity)
	features = np.zeros((O,3))
	features[:,0] = z;
	features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
	features[:,2] = x/distanceWeight;
	features = np.array(features,'f')
	# cluster data
	centroids,variance = kmeans(features,K)
	#use the found clusters to map
	label,distance = vq(features,centroids)
	# re-create image from
	labelIm = np.array(np.reshape(label,(M,N)))
	f = figure(1)
	imshow(labelIm)
	f.canvas.draw()
	f.show()

#------------------------------------------------
#   Methods for segmentation
#------------------------------------------------
def detectPupilKMeans(gray,K=2,distanceWeight=2,reSize=(40,40)):
	''' Detects the pupil in the image, gray, using k-means
			gray              : grays scale image
			K                 : Number of clusters
			distanceWeight    : Defines the weight of the position parameters
			reSize            : the size of the image to do k-means on
		'''
	#Resize for faster performance
	smallI = cv2.resize(gray, reSize)
	M,N = smallI.shape
	#Generate coordinates in a matrix
	X,Y = np.meshgrid(range(M),range(N))
	#Make coordinates and intensity into one vectors
	z = smallI.flatten()
	x = X.flatten()
	y = Y.flatten()
	O = len(x)
	#make a feature vectors containing (x,y,intensity)
	features = np.zeros((O,3))
	features[:,0] = z;
	features[:,1] = y/distanceWeight; #Divide so that the distance of position weighs less than intensity
	features[:,2] = x/distanceWeight;
	features = np.array(features,'f')
	# cluster data
	centroids,variance = kmeans(features,K)
	#use the found clusters to map
	label,distance = vq(features,centroids)
	# re-create image from
	labelIm = np.array(np.reshape(label,(M,N)))
	f = figure(1)
	imshow(labelIm)
	f.canvas.draw()
	f.show()

def detectPupilHough(gray, accThr=600):
	#Using the Hough transform to detect ellipses
	blur = cv2.GaussianBlur(gray, (9,9),9)
	##Pupil parameters
	dp = 6; minDist = 10
	highThr = 30 #High threshold for canny
	#accThr = 600; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	maxRadius = 50;
	minRadius = 30;
	#See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCirclesIn thus
	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
	#Print the circles
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	pupils = list(circles)
	if (circles !=None):
		#print circles
		all_circles = circles[0]
		M,N = all_circles.shape
		k=1
		for c in all_circles:
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			K=k+1
			#Circle with max votes
		c=all_circles[0,:]
		cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255))
	cv2.imshow("hough",gColor)
	return pupils

def detectIrisHough(gray, accThr=600):
	#Using the Hough transform to detect ellipses
	blur = cv2.GaussianBlur(gray, (11,11),9)
	##Pupil parameters
	dp = 6; minDist = 10
	highThr = 30 #High threshold for canny
	#accThr = 600; #accumulator threshold for the circle centers at the detection stage. The smaller it is, the more false circles may be detected
	maxRadius = 150;
	minRadius = 100;
	#See help for http://opencv.itseez.com/modules/imgproc/doc/feature_detection.html?highlight=houghcircle#cv2.HoughCirclesIn thus
	circles = cv2.HoughCircles(blur,cv2.cv.CV_HOUGH_GRADIENT, dp,minDist, None, highThr,accThr,minRadius, maxRadius)
	#Print the circles
	gColor = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
	irises = []
	if (circles !=None):
		#print circles
		all_circles = circles[0]
		M,N = all_circles.shape
		k=1
		for c in all_circles:
			irises.append(c)
			cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (int(k*255/M),k*128,0))
			K=k+1
			#Circle with max votes
		c=all_circles[0,:]
		cv2.circle(gColor, (int(c[0]),int(c[1])),c[2], (0,0,255))
	cv2.imshow("hough",gColor)
	return irises
#--------------------------
#         UI related
#--------------------------

def setText(dst, (x, y), s):
	cv2.putText(dst, s, (x+1, y+1), cv2.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 0), thickness = 2, lineType=cv2.CV_AA)
	cv2.putText(dst, s, (x, y), cv2.FONT_HERSHEY_PLAIN, 1.0, (255, 255, 255), lineType=cv2.CV_AA)

vertical_window_size = 523
horizontal_window_size = 640

def setupWindowSliders():
	''' Define windows for displaying the results and create trackbars'''
	cv2.namedWindow("Result")
	cv2.moveWindow("Result", 0, 0)
	cv2.namedWindow('Threshold')
	cv2.moveWindow("Threshold", 0, vertical_window_size)
	cv2.namedWindow('Controls')
	cv2.moveWindow("Controls", horizontal_window_size, 0)
	cv2.resizeWindow('Controls', horizontal_window_size, 0)
	cv2.namedWindow("TempResults")
	cv2.moveWindow("TempResults", horizontal_window_size, vertical_window_size)
	#Threshold value for the pupil intensity
	cv2.createTrackbar('pupilThr','Controls', default_pupil_threshold, 255, onSlidersChange)
	#Threshold value for the glint intensities
	cv2.createTrackbar('glintThr','Controls', 240, 255,onSlidersChange)
	#define the minimum and maximum areas of the pupil
	cv2.createTrackbar('minSize','Controls', 20, 2000, onSlidersChange)
	cv2.createTrackbar('maxSize','Controls', 2000,2000, onSlidersChange)
	#Value to indicate whether to run or pause the video
	cv2.createTrackbar('Stop/Start','Controls', 0,1, onSlidersChange)

def getSliderVals():
	'''Extract the values of the sliders and return these in a dictionary'''
	sliderVals={}
	sliderVals['pupilThr'] = cv2.getTrackbarPos('pupilThr', 'Controls')
	sliderVals['glintThr'] = cv2.getTrackbarPos('glintThr', 'Controls')
	sliderVals['minSize'] = 50*cv2.getTrackbarPos('minSize', 'Controls')
	sliderVals['maxSize'] = 50*cv2.getTrackbarPos('maxSize', 'Controls')
	sliderVals['Running'] = 1==cv2.getTrackbarPos('Stop/Start', 'Controls')
	return sliderVals

def onSlidersChange(dummy=None):
	''' Handle updates when slides have changed.
	 This  function only updates the display when the video is put on pause'''
	global imgOrig;
	sv=getSliderVals()
	if(not sv['Running']): # if pause
		update(imgOrig)

def euclidianDistance(a,b):
	a_x, a_y = a
	b_x, b_y = b
	return math.sqrt((a_x - b_x) ** 2 + (a_y - b_y) **2)

#--------------------------
#         main
#--------------------------
run(inputFile)
