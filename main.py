import cv2
import numpy as np
import os
import tanuki

#create lists to save the labels (the name of the shape)
train_labels, train_images = [],[]
train_dir = './shapes'
shape_list = ['circle', 'triangle', 'tetragon', 'pentagon', 'other']

# function to make classifier
def classify(images):
    sd = tanuki.ShapeDetector()
    preds = []
    for img in images:
        pred = sd.detect_img(img)
        preds.append(pred)
    return np.array(preds)



if __name__ == '__main__':
    #iterate through each shape
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(train_dir,shape)):
            train_images.append(cv2.imread(os.path.join(train_dir,shape,file_name), 0))
            #add an integer to the labels list
            train_labels.append(shape_list.index(shape))

    print('Number of training images: ', len(train_images))

    pred_labels = classify(train_images)

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
            #add an integer to the labels list
            test_labels.append(shape_list.index(shape))

    print('Number of test images: ', len(test_images))

    test_images, test_labels = preprocess(test_images, test_labels)
    pred_labels = model.predict(test_images)
    pred_acc = np.sum(pred_labels==test_labels)/len(test_labels)*100
    print("Test Accuracy = {}".format(pred_acc))
    """
