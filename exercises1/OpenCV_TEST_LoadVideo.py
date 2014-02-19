import cv2
cap = cv2.VideoCapture("test-03.avi")
cv2.namedWindow("input")
f=True
while(f):
    f, img = cap.read()
    if f==True:
        cv2.imshow("input", img)
        ch = cv2.waitKey(1)
        if ch == 32:#  Break by SPACE key
            break