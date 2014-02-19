import cv2
cv2.namedWindow("camera",1)
capture = cv2.VideoCapture(0)
ret=True
while ret:
    imgs = []
    ret, img =capture.read()
    imgs.append(img)
    cv2.imshow("camera", img)
    ch = cv2.waitKey(1)
    if ch == 32:# Break by SPACE key
        break