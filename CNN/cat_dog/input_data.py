# Copyright 2017 kairos03. All Right Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# ===============================================================
import zipfile
import random
import numpy as np
import cv2


def next_batch(batch_size, one_hot=True):
    """get next train batch data

    return random image with label
    no normalization
    resizing to 200x200

    Args:
        batch_size: size of next batch
        one_hot: label encoding with one hot (is not working now)

    Returns:
        batch_x: features of next batch
        batch_y: label of next batch
    """
    data_path = "./data/train.zip"
    label_class = ['cat', 'dog']

    batch_x = []
    batch_y = []

    # open data zipfile
    with zipfile.ZipFile(data_path) as myzip:
        # get namelist of zipfile
        namelist = myzip.namelist()
        # shuffle namelist
        random.shuffle(namelist)

        # pick next batch data
        for i in range(batch_size):

            # label
            # cat: 0, dog: 1
            label = namelist[i][6:9]
            for n in range(len(label_class)):
                if label == label_class[n]:
                    label = n
                    break
            batch_y.append(label)

            # image
            img = myzip.read(namelist[i])
            img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)

            # resizing
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)
            img = img.astype('float32')
            batch_x.append(img)

    # one_hot encoding
    # if one_hot:

    return batch_x, batch_y


def test_data(one_hot=True):
    """get test data

        return test image with label
        no normalization
        resizing to 200x200

        Args:
            one_hot: label encoding with one hot (is not working now)

        Returns:
            test_x: features of test set
            test_y: label of test set
        """
    data_path = "./data/test.zip"
    label_class = ['cat', 'dog']

    test_x = []
    test_y = []

    # open data zipfile
    with zipfile.ZipFile(data_path) as myzip:
        # get namelist of zipfile
        namelist = myzip.namelist()

        # pick next batch data
        for i in range(len(namelist)):

            # label
            # cat: 0, dog: 1
            label = namelist[i][6:9]
            for n in range(len(label_class)):
                if label == label_class[n]:
                    label = n
                    break
            test_y.append(label)

            # image
            img = myzip.read(namelist[i])
            img = cv2.imdecode(np.frombuffer(img, np.uint8), 1)

            # resizing
            img = cv2.resize(img, (200, 200), interpolation=cv2.INTER_AREA)

            test_x.append(img)

    # one_hot encoding
    # if one_hot:

    return test_x, test_y