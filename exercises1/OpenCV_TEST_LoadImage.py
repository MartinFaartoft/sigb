import cv2
img = cv2.imread("Nathan2.jpg")
cv2.namedWindow("Picture")
cv2.imshow("Picture", img)
while True:
    ch = cv2.waitKey(1)
    if ch == 32:# Break by SPACE key
        break