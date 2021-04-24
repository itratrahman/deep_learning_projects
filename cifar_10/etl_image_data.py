import os
import time
import random
import math
import json
import numpy as np
from sklearn.utils import shuffle
from auto_augmentation import CIFAR10Policy
from PIL import Image
import cv2

BASE_DIR = os.path.abspath(os.path.dirname("__file__"))
INPUT_DIR = os.path.join(BASE_DIR, "cifar10")
SAVE_DIRECTORY = os.path.join(BASE_DIR, "cifar10","numpy_data")
IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS, N_CLASSES = 32,32,3,10

def generate_batch(indices, ids, labels,
                   IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                   N_CLASSES, one_hot_encode = False, augment=False, policy = None):
    """
    A function for generating batch of images and labels
    """
    # extract the id based of train/test set
    batch_ids = ids[indices]
    # initialize numpy arrays to hold images and masks
    X = np.zeros((len(batch_ids), IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS), dtype=np.float32)

    # iterate through each image
    for n, id_ in enumerate(batch_ids):
        img = cv2.imread(id_)
        # resize image
        if img.shape[0] != 32 or img.shape[1] != 32:
            img = cv2.resize(image, (IMG_HEIGHT, IMG_WIDTH))
        if augment:
            img = np.array(policy(Image.fromarray(img.astype(np.uint8))))
        # store the image in the designated numpy array
        X[n] = img.astype(np.float32)
        # del image to save runtime memory
        del(img)
        Y = np.array(labels[indices])
        # if onehot encoding is wanted
        if one_hot_encode:
            Y = np.eye(N_CLASSES)[Y]
        else:
            Y = Y[:,np.newaxis]

    return X, Y

if __name__ == "__main__":

    # set random seeds
    radom_seed = 0
    random.seed(radom_seed)
    numpy_seed = 0
    np.random.seed(numpy_seed)

    # create mapping dictionary of classes to labels and labels to classes
    classes_to_labels = dict(zip(list(os.walk(INPUT_DIR+"/train/"))[0][1], [i for i in range(10)]))
    labels_to_classes = dict(zip([i for i in range(10)], list(os.walk(INPUT_DIR+"/train/"))[0][1]))

    # create train labels
    train_ids = []
    train_labels = []
    for item in list(os.walk(INPUT_DIR+"/train/"))[0][1]:
        for image in list(os.walk(INPUT_DIR+"/train/"+str(item)+"/"))[0][2]:
            train_ids.append(INPUT_DIR+"/train/"+str(item)+"/"+image)
            train_labels.append(item)
    train_ids = np.array(train_ids)
    train_labels = [classes_to_labels[train_label] for train_label in train_labels]
    train_labels = np.array(train_labels)

    # create cifar10 auto augmentation policy
    policy = CIFAR10Policy()

    # randomly generate train and validation indices
    train_indices = np.random.choice(len(train_ids), round(len(train_ids)*0.96), replace=False)
    validation_indices = np.array(list(set(range(len(train_ids))) - set(train_indices)))
    print("Size of train set:", len(train_indices))
    print("Size of validation set:", len(validation_indices))
    print()

    # generate train data
    X_train_1, Y_train_1 = generate_batch(train_indices, train_ids, train_labels,
                                          IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                                          N_CLASSES, augment = False)
    X_train_2, Y_train_2 = generate_batch(train_indices, train_ids, train_labels,
                                            IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                                            N_CLASSES, augment = True, policy=policy)
    X_train = np.append(X_train_1, X_train_2, axis = 0)
    Y_train = np.append(Y_train_1, Y_train_2, axis = 0)
    # X_train_1, X_train_2, Y_train_1, Y_train_2 = None, None, None, None
    X_train_1 = X_train_1/255.0
    X_train = X_train/255.0
    Y_train = np.squeeze(Y_train)
    X_train, Y_train = shuffle(X_train, Y_train)
    print("Finished creating train image data.\n")

    # generate valid data
    X_valid, Y_valid = generate_batch(validation_indices, train_ids, train_labels,
                                    IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                                    N_CLASSES)
    X_valid = X_valid/255.0
    Y_valid = np.squeeze(Y_valid)
    print("Finished creating validation image data.\n")

    # create train labels
    test_ids = []
    test_labels = []
    for item in list(os.walk(INPUT_DIR+"/test/"))[0][1]:
        for image in list(os.walk(INPUT_DIR+"/test/"+str(item)+"/"))[0][2]:
            test_ids.append(INPUT_DIR+"/test/"+str(item)+"/"+image)
            test_labels.append(item)
    test_ids = np.array(test_ids)
    test_labels = [classes_to_labels[test_label] for test_label in test_labels]
    test_labels = np.array(test_labels)
    # create test data
    X_test, Y_test = generate_batch(list(range(len(test_ids))), test_ids, test_labels,
                                IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                                N_CLASSES, one_hot_encode = False)
    X_test = X_test/255.0
    Y_test = np.squeeze(Y_test)
    print("Finished creating test image data.\n")

    # make directory to store numpy arrays if a directory does not exists
    if "numpy_data" not in os.listdir(INPUT_DIR):
        os.makedirs(SAVE_DIRECTORY)
    # if save directory does not exit then create a directory
    if not os.path.exists(SAVE_DIRECTORY):
        os.mkdir(SAVE_DIRECTORY)
    # save numpy array
    np.save(os.path.join(SAVE_DIRECTORY, "X_train.npy"), X_train)
    np.save(os.path.join(SAVE_DIRECTORY, "X_train_pure.npy"), X_train_1)
    np.save(os.path.join(SAVE_DIRECTORY, "X_valid.npy"), X_valid)
    np.save(os.path.join(SAVE_DIRECTORY, "X_test.npy"), X_test)
    np.save(os.path.join(SAVE_DIRECTORY, "Y_train.npy"), Y_train)
    np.save(os.path.join(SAVE_DIRECTORY, "Y_train_pure.npy"), Y_train_pure)
    np.save(os.path.join(SAVE_DIRECTORY, "Y_valid.npy"), Y_valid)
    np.save(os.path.join(SAVE_DIRECTORY, "Y_test.npy"), Y_test)
    print("Saved data in numpy arrays.\n")
