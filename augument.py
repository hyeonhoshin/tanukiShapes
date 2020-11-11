import cv2
import numpy as np
import imutils
import matplotlib.pyplot as plt
import random
from numpy import all, any

def Augument(img):
    ROI = find_ROI(img)

    # plt.imshow(ROI,cmap='gray')
    # plt.show()
    
    ## Random rotate
    ang = np.random.randint(0,360)
    print("Rotation(Deg) :",ang)
    ROI_inv = cv2.threshold(ROI, 60, 255, cv2.THRESH_BINARY_INV)[1]
    ROI_rot = imutils.rotate_bound(ROI_inv,ang)
    ROI = cv2.threshold(ROI_rot, 60, 255, cv2.THRESH_BINARY_INV)[1]

    # plt.imshow(ROI,cmap='gray')
    # plt.show()

    ## Random scaling
    flag = True
    scale_max = 2
    while flag == True or any(np.array(ROI.shape) >= 300):
        fx = random.uniform(0.5,scale_max)
        fy = random.uniform(0.5,scale_max)
        print("Scaling in x :",fx)
        print("Scaling in y :",fy)

        ROI = cv2.resize(ROI,dsize=(0,0), fx=fx,fy=fy)
        flag = False

        scale_max *= 0.8

    # plt.imshow(ROI,cmap="gray")
    # plt.show()

    ## Random offset

    patch_width = ROI.shape[0]
    patch_height = ROI.shape[1]

    left_upper_x = np.random.randint(0,300-patch_width)
    left_upper_y = np.random.randint(0,300-patch_height)

    whites = np.tile(255,(300,300))
    whites[left_upper_x:left_upper_x+patch_width,
                        left_upper_y:left_upper_y+patch_height] = ROI
    img_out = whites

    # plt.imshow(img_out,cmap="gray")
    # plt.show()

    return img_out

def find_ROI(img):
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    thresh = cv2.threshold(blurred, 60, 255, cv2.THRESH_BINARY_INV)[1]

    ## Find ROI
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnt = cnts[0]

    x,y,w,h = cv2.boundingRect(cnt)

    ROI = img[x-1:x+w+1,y-1:y+h+1]
    return ROI

