import numpy as np

from mnist import MNIST

from classification_base import MNIST_PATH

def prep_binary_classes(y, digit=2):
    return np.array([int(x==digit) for x in y])

def mnist_training():
    mndata = MNIST(MNIST_PATH)
    train_ims, train_labels = mndata.load_training()
    train_X = np.array(train_ims)
    train_y = np.array(train_labels)
    return train_X, train_y

def mnist_testing():
    mndata = MNIST(MNIST_PATH)
    test_ims, test_labels = mndata.load_testing()
    test_X = np.array(test_ims)
    test_y = np.array(test_labels)
    return test_X, test_y

def mnist_training_binary(num):
    X, y = mnist_training()
    return X, np.array([int(yi==num) for yi in y])

def mnist_testing_binary(num):
    X, y = mnist_testing()
    return X, np.array([int(yi==num) for yi in y])

