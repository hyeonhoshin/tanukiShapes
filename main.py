import cv2
import numpy as np
import os
import imutils

# create lists to save the labels (the name of the shape)
train_labels, train_images = [], []
train_dir = './shapes'
shape_list = ['circle', 'triangle', 'tetragon', 'pentagon', 'other']

def destrained(img, cnt):
    # img : Grayscaled img, white backgrounded is no matter
    # Output : Straining imgs are processed
    x, y, w, h = cv2.boundingRect(cnt)
    patch = img[y:y+h, x:x+w]
    # plt.imshow(patch, cmap='gray')
    # plt.show()
    out = cv2.resize(patch, dsize=(300, 300), interpolation=cv2.INTER_CUBIC)
    out = np.array(out)
    canny = cv2.Canny(out, 100, 255)/255.0
    out = (255-out)/255.0

    return np.sum(out), np.sum(canny)

# function to preprocess data
def preprocess(images, labels):
    areas = []
    peris = []
    vertices = []
    for i, img in enumerate(images):
        thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnt = cnts[0]

        arc = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.005 * arc, True)

        area, peri = destrained(img, cnt)

        areas.append(area)
        peris.append(peri)
        vertices.append(len(approx))
        
    return (areas, peris, vertices), labels

# function to make classifier
def classify(features):
    areas = features[0]
    peris = features[1]
    vertices = features[2]

    preds = []
    for i, num_vertices in enumerate(vertices):
    # if the shape is a triangle, it will have 3 vertices
        if num_vertices == 3:
            shape = 1
        # if the shape has 4 vertices, it is either a square or
        # a rectangle
        elif num_vertices == 4:
            shape = 2
        # if the shape is a pentagon, it will have 5 vertices
        elif num_vertices == 5:
            shape = 3
        elif num_vertices <= 7:
            shape = 4
        else:

            area = areas[i]
            r = 150
            # Area check
            perfect_c_area = (r**2)*np.pi
            ratio = area/perfect_c_area

            pred_peri = peris[i]
            perfect_peri = 2*np.pi*150

            ratio_peri = pred_peri/perfect_peri

            if (0.95 <= ratio and ratio <= 1/0.95) and ratio_peri<1.3:
                shape = 0
            else:
                shape = 4
        
        # return the name of the shape
        preds.append(shape)
    return np.array(preds)

if __name__ == '__main__':
    # iterate through each shape
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(train_dir,shape)):
            train_images.append(cv2.imread(os.path.join(train_dir,shape,file_name), 0))
            # add an integer to the labels list
            train_labels.append(shape_list.index(shape))

    print('Number of training images: ', len(train_images))

    features, train_labels = preprocess(train_images, train_labels)

    pred_labels = classify(features)

    # Calculate accuracy (Do not erase or modify here)
    pred_acc = np.sum(pred_labels==train_labels)/len(train_labels)*100
    print("Accuracy = {}".format(pred_acc))


    """forTA (Do not erase here)
    test_dir = '../ForTA'
    test_labels, test_images = [], []
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(test_dir,shape)):
            test_images.append(cv2.imread(os.path.join(test_dir,shape,file_name), 0))
            # add an integer to the labels list
            test_labels.append(shape_list.index(shape))

    print('Number of test images: ', len(test_images))

    test_images, test_labels = preprocess(test_images, test_labels)
    pred_labels = model.predict(test_images)
    pred_acc = np.sum(pred_labels==test_labels)/len(test_labels)*100
    print("Test Accuracy = {}".format(pred_acc))
    """

