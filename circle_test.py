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
img = cv2.threshold(img,90,255,cv2.THRESH_BINARY_INV)[1]

area = np.sum(img==255)
r = 150
# Area check
perfect_c_area = (r**2)*np.pi
ratio = area/perfect_c_area

if 0.95 <= ratio and ratio <= 1/0.95:
    print("This is a circle")
else:
    print("This is not a circle")