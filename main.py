import cv2
import numpy as np
import os
from skimage.measure import regionprops
from skimage.transform import *
from sklearn.model_selection import train_test_split
from augument import *

#create lists to save the labels (the name of the shape)
train_labels, train_images = [],[]
train_dir = './shapes'
shape_list = ['circle', 'triangle', 'tetragon', 'pentagon', 'other']

aug_num = 10 # x100 dataset

#function to preprocess data
def preprocess(images, labels, augment=False):
    """You can make your preprocessing code in this function.
    Here, we just flatten the images, for example.
    In addition, you can split this data into the training set and validation set for robustness to the test(unseen) data.

    :params list images: (Number of images x row x column)
    :params list labels: (Number of images, 1)
    :rtype: array
    :return: preprocessed images and labels
    """

    # Dimension reduction
    #imgs = []
    #for img in images:
    #    imgs.append(cv2.resize(img,dsize=(128,128)))
    #images = np.array(imgs)

    dataDim = np.prod(images[0].shape)
    images = np.array(images)

    ## Data Augumentation
    if augment == True:
    
        out_imgs = []
        out_labels = []
        
        for i in range(images.shape[0]):
            img = images[i]
            for j in range(aug_num):
                out_imgs.append(Augument(img))
                out_labels.append(labels[i])

        images = np.array(out_imgs)
        labels = np.array(out_labels)
    else:
        out_imgs = []
        out_labels = []
        
        for i in range(images.shape[0]):
            img = images[i]
            roi = find_ROI(img)
            roi = cv2.threshold(roi, 60, 255, cv2.THRESH_BINARY_INV)[1]
            img = cv2.resize(roi,dsize=(300,300))
            # plt.imshow(img)
            # plt.show()
            out_imgs.append(img)
            out_labels.append(labels[i])

        images = np.array(out_imgs)
        labels = np.array(out_labels)

    images = images.reshape(len(images), dataDim)
    images = images.astype('float32')
    #images /=255

    return images, labels


# function to make classifier
def classify(images, labels):
    """You can make your classifier code in this function.
    Here, we use KNN classifier, for example.

    :params array images: (Number of images x row x column)
    :params array labels: (Number of images)
    :return: classifier model
    """
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=10)
    neigh.fit(images, labels)
    return neigh



if __name__ == '__main__':
    #iterate through each shape
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(train_dir,shape)):
            train_images.append(cv2.imread(os.path.join(train_dir,shape,file_name), 0))
            #add an integer to the labels list
            train_labels.append(shape_list.index(shape))

    print('Number of training images: ', len(train_images))
    
    # Separation validation set
    train_images, validation_images, train_labels, validation_labels = train_test_split(train_images,train_labels,test_size=0.1, random_state=999)

    # Preprocess (your own function)
    train_images, train_labels = preprocess(train_images, train_labels, augment=True)
    validation_images, validation_labels = preprocess(validation_images, validation_labels, augment=False)

    # Make a classifier (your own function)
    model = classify(train_images, train_labels)

    # Predict the labels from the model (your own code depending the output of the train function)
    pred_labels = model.predict(validation_images)

    # Calculate accuracy (Do not erase or modify here)
    pred_acc = np.sum(pred_labels==validation_labels)/len(validation_labels)*100
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
