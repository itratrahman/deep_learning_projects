import os
import time
import argparse
import random
import math
import json
import numpy as np
from sklearn.utils import shuffle
from auto_augmentation import CIFAR10Policy
from PIL import Image
import cv2

parser = argparse.ArgumentParser()
parser.add_argument("--img_size", "-size",
                    type=int, help="size of the image")

BASE_DIR = os.path.abspath(os.path.dirname("__file__"))
INPUT_DIR = os.path.join(BASE_DIR, "data", "NA_Fish_Dataset")
SAVE_DIRECTORY = os.path.join(BASE_DIR, "data", "numpy_data")

args = parser.parse_args()
IMG_HEIGHT = IMG_WIDTH = \
args.img_size if (args.img_size is not None) else 256

IMG_CHANNELS, N_CLASSES = 3,9

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
        if img.shape[0] != IMG_HEIGHT or img.shape[1] != IMG_WIDTH:
            img = cv2.resize(img, (IMG_HEIGHT, IMG_WIDTH))
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

    print("Image dimension:{}x{}x{}\n".format(IMG_HEIGHT,IMG_WIDTH,IMG_CHANNELS))

    print("Folders:",os.listdir(INPUT_DIR))

    classes_to_labels = dict(zip(list(os.walk(INPUT_DIR))[0][1], [i for i in range(10)]))
    labels_to_classes = dict(zip([i for i in range(10)], list(os.walk(INPUT_DIR))[0][1]))

    train_ids = []
    train_labels = []
    for item in list(os.walk(INPUT_DIR))[0][1]:
        for image in list(os.walk(os.path.join(INPUT_DIR, str(item))))[0][2]:
            train_ids.append(os.path.join(INPUT_DIR, str(item), image))
            train_labels.append(item)
    train_ids = np.array(train_ids)
    train_labels = [classes_to_labels[train_label] for train_label in train_labels]
    train_labels = np.array(train_labels)

    # randomly generate train and validation indices
    train_indices = np.random.choice(len(train_ids), round(len(train_ids)*0.85), replace=False)
    validation_indices = np.array(list(set(range(len(train_ids))) - set(train_indices)))
    print("Size of train set:", len(train_indices))
    print("Size of validation set:", len(validation_indices))
    print()

    # generate train data
    policy = CIFAR10Policy()
    X_train_1, Y_train_1 = generate_batch(train_indices, train_ids, train_labels,
                                          IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                                          N_CLASSES, augment = False)
    X_train_2, Y_train_2 = generate_batch(train_indices, train_ids, train_labels,
                                            IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS,
                                            N_CLASSES, augment = True, policy=policy)
    X_train = np.append(X_train_1, X_train_2, axis = 0)
    Y_train = np.append(Y_train_1, Y_train_2, axis = 0)
    X_train_1, X_train_2, Y_train_1, Y_train_2 = None, None, None, None
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
    X_valid, Y_valid = shuffle(X_valid, Y_valid)
    print("Finished creating test image data.\n")

    # if save directory does not exit then create a directory
    if not os.path.exists(SAVE_DIRECTORY):
        os.mkdir(SAVE_DIRECTORY)
    # save numpy array
    np.save(os.path.join(SAVE_DIRECTORY, "X_train_{}.npy".format(IMG_HEIGHT)), X_train)
    np.save(os.path.join(SAVE_DIRECTORY, "X_valid_{}.npy".format(IMG_HEIGHT)), X_valid)
    np.save(os.path.join(SAVE_DIRECTORY, "Y_train_{}.npy".format(IMG_HEIGHT)), Y_train)
    np.save(os.path.join(SAVE_DIRECTORY, "Y_valid_{}.npy".format(IMG_HEIGHT)), Y_valid)
    print("Saved data in numpy arrays.\n")
