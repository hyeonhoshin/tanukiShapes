# import the necessary packages
import cv2
import imutils
import matplotlib.pyplot as plt
import numpy as np

class ShapeDetector:
	def __init__(self):
		pass
	def destrained(self, img, cnt):
		# img : Grayscaled img, white backgrounded is no matter
		# Output : Straining imgs are processed
		x, y, w, h = cv2.boundingRect(cnt)
		patch = img[y:y+h,x:x+w]
		# plt.imshow(patch, cmap='gray')
		# plt.show()
		out = cv2.resize(patch,dsize=(300,300),interpolation=cv2.INTER_CUBIC)
		out = np.array(out)
		canny = cv2.Canny(out, 100, 255)
		out = (255-out)/255.0
		self.patch = out
		self.area = np.sum(self.patch)
		self.canny = canny/255.0
		self.peri = np.sum(self.canny)

		
	def detect_img(self, img):
		# img : Grayscaled img
		thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)[1] # 255, 255, 255 ... -> 0, 0, 0, 255... 
		self.area = np.sum(thresh==255)
		cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
			cv2.CHAIN_APPROX_SIMPLE)
		cnts = imutils.grab_contours(cnts)
		self.destrained(img,cnts[0])
		label = self.detect(cnts[0]) # only one shape

		return label

	def detect(self, c):
		# initialize the shape name and approximate the contour
		shape = 4
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.005 * peri, True)

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
		elif len(approx) <= 7:
			shape = 4
		else:


			area = np.sum(self.patch)
			r = 150
			# Area check
			perfect_c_area = (r**2)*np.pi
			ratio = area/perfect_c_area

			pred_peri = self.peri
			perfect_peri = 2*np.pi*150

			ratio_peri = pred_peri/perfect_peri

			if (0.95 <= ratio and ratio <= 1/0.95) and ratio_peri<1.3:
				shape = 0
			else:
				shape = 4
		# return the name of the shape
		return shape