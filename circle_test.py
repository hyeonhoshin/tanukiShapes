import cv2
import numpy as np
import tanuki
import imutils
import matplotlib.pyplot as plt

img = cv2.imread('shapes/circle/3.png')
img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgOri = img
img = cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV)[1]

plt.imshow(img, cmap='gray')
plt.show()

cnts = cv2.findContours(img.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)

sd = tanuki.ShapeDetector()
img = sd.destrained(imgOri,cnts[0])

plt.imshow(img, cmap='gray')
plt.show()

img = cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV)[1]
img = cv2.medianBlur(img, 5)

circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 10,param1=100,param2=5000,minRadius=1, maxRadius=140)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)

cv2.imshow('img', cimg)
cv2.waitKey(0)
cv2.destroyAllWindows()