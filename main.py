import cv2
import numpy as np
import os

# create lists to save the labels (the name of the shape)
train_labels, train_images = [], []
train_dir = './shapes'
shape_list = ['circle', 'triangle', 'tetragon', 'pentagon', 'other']

patch_size = 32

def translate(image, x, y):
    # define the translation matrix and perform the translation
    M = np.float32([[1, 0, x], [0, 1, y]])
    shifted = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

    # return the translated image
    return shifted
def grab_contours(cnts):
    # if the length the contours tuple returned by cv2.findContours
    # is '2' then we are using either OpenCV v2.4, v4-beta, or
    # v4-official
    if len(cnts) == 2:
        cnts = cnts[0]

    # if the length of the contours tuple is '3' then we are using
    # either OpenCV v3, v4-pre, or v4-alpha
    elif len(cnts) == 3:
        cnts = cnts[1]

    # otherwise OpenCV has changed their cv2.findContours return
    # signature yet again and I have no idea WTH is going on
    else:
        raise Exception(("Contours tuple must have length 2 or 3, "
            "otherwise OpenCV changed their cv2.findContours return "
            "signature yet again. Refer to OpenCV's documentation "
            "in that case"))

    # return the actual contours array
    return cnts

def de_stretch(img, cnt):
    '''
    Input : Grayscaled, white background img, and contours
    Output : Aligned (Unroated) and De-streteched image
    '''

    img = 255-img #Inverse image

    # Align img
    ## Extract roated angle and roate
    center, size, angle = cv2.minAreaRect(cnt)
    img = translate(img,150-center[0],150-center[1])

    R = cv2.getRotationMatrix2D((150, 150), angle, scale=1)
    img_rotated = cv2.warpAffine(img, R, (300, 300))

    cnts = cv2.findContours(img_rotated.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = grab_contours(cnts)

    # De-stretched img
    ## Find Bounding box exactly fitted
    x,y,w,h = cv2.boundingRect(cnts[0])
    img_pad = np.pad(img_rotated,2,constant_values=0)
    img = img_pad[y:y+h+4, x:x+w+4]

    img = 255-img #Inverse image

    out = cv2.resize(img, dsize=(patch_size, patch_size), interpolation=cv2.INTER_LINEAR)
    out = np.array(out)
    
    # Filtering for denoising (edges have noises)
    out = cv2.bilateralFilter(out,9,25,25)

    # Feature extration
    canny = cv2.Canny(out, 100, 255)/255.0  # Calculate boundary using Canny detecter.
    out = (255-out)/255.0                   # Normalize img to 0 ~ 1

    return np.sum(out), np.sum(canny)

# function to preprocess data
def preprocess(images, labels):
    """
    Extract Scale, Rotation invariant features.
    Number of vertices, area of patches, perimeter of patches

    Input : images(N x width x height), labels(0~4 tuples)
    Output : features(3 x N), labels(0~4 tuples)
    """

    areas = []
    peris = []
    vertices = []
    for i, img in enumerate(images):

        # Find Contours
        thresh = cv2.threshold(img, 60, 255, cv2.THRESH_BINARY_INV)[1]
        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = grab_contours(cnts)
        cnt = cnts[0]

        # Approximate it.
        arc = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, 0.01 * arc, True)

        # Extract features and make them scale, rotation invariant.
        area, peri = de_stretch(img, cnt)

        # Save it tuples
        areas.append(area)
        peris.append(peri)
        vertices.append(len(approx))
        
    return (areas, peris, vertices), labels

# function to make classifier
def classify(features):
    '''
    Rule-based classifier.
    It classifies by the number of vertices.

    Input : features (areas, peris, vertices) from preprocess stage.
    Output : Predicted labels.
    '''
    # Unizip features
    areas = features[0]
    peris = features[1]
    vertices = features[2]

    preds = []
    for i, num_vertices in enumerate(vertices):
        # print("Classfies in...",i)
        
        if num_vertices == 3: # Triangle
            shape = 1
        
        elif num_vertices == 4: # Rectangle
            shape = 2
        elif num_vertices == 5: # Pentagon
            shape = 3
        elif num_vertices <= 7: # 6 or 7 vertices are different shape.
            shape = 4
        else:                   # Many vertices (includes circles and others)

            # Check area is same with pi*r^2
            area = areas[i]
            r = patch_size//2
            perfect_c_area = (r**2)*np.pi
            ratio_area = area/perfect_c_area

            # Check area is same with 2*pi*r
            pred_peri = peris[i]
            perfect_peri = 2*np.pi*r
            ratio_peri = pred_peri/perfect_peri

            print("{}th perimeter = {}, area = {}".format(i,ratio_peri,ratio_area))

            if (0.75 <= ratio_area and ratio_area <= 0.95) and (0.95<ratio_peri and ratio_peri<1.09):   # Experience-base determined thresholds.
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
    pred_labels = classify(features)                                    # Rule-based method do not need training step.

    print("Predicted lables :",pred_labels)

    # Calculate accuracy (Do not erase or modify here)
    pred_acc = np.sum(pred_labels==train_labels)/len(train_labels)*100
    print("Accuracy = {}".format(pred_acc))

    # Pre-made TA code which I made for convenience.

    # forTA (Do not erase here)
    test_dir = '../ForTA'
    test_labels, test_images = [], []
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(test_dir,shape)):
            test_images.append(cv2.imread(os.path.join(test_dir,shape,file_name), 0))
            # add an integer to the labels list
            test_labels.append(shape_list.index(shape))

    print('Number of test images: ', len(test_images))

    features, test_labels = preprocess(test_images, test_labels)
    pred_labels = classify(features)
    print(pred_labels)
    pred_acc = np.sum(pred_labels==test_labels)/len(test_labels)*100
    print("Test Accuracy = {}".format(pred_acc))


    # Original TA code.
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

