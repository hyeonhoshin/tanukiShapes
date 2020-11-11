# import the necessary packages
import cv2
import imutils
import matplotlib.pyplot as plt

class ShapeDetector:
	def __init__(self):
		pass
	def destrained(self, img, cnt):
		# img : Grayscaled img, white backgrounded is no matter
		# Output : Straining imgs are processed
		x, y, w, h = cv2.boundingRect(cnt)
		patch = img[y-2:y+h+2,x-2:x+w+2]
		# plt.imshow(patch, cmap='gray')
		# plt.show()
		return cv2.resize(patch,dsize=(300,300))
		
	def detect_img(self, img):
		# img : Grayscaled img
		thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)[1] # 255, 255, 255 ... -> 0, 0, 0, 255... 
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		img = self.destrained(img,cnts[0])
		label = self.detect(cnts[0]) # only one shape

		return label

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = 4
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.01 * peri, True)

        # if the shape is a triangle, it will have 3 vertices
		if len(approx) == 3:
			shape = 1
		# if the shape has 4 vertices, it is either a square or
		# a rectangle
		elif len(approx) == 4:
			shape = 2
		# if the shape is a pentagon, it will have 5 vertices
		elif len(approx) == 5:
			shape = 3
		# otherwise, we assume the shape is a circle
		elif len(approx) > 20:
			shape = 4
		else:
			shape = 0
		# return the name of the shape
		return shape