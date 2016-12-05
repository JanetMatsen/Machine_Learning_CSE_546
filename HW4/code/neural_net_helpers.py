import numpy as np

from mnist import MNIST

MNIST_PATH = '../../data/python-mnist/data'

def mnist_training():
    """
    Returns MNIST training data with N columns and d rows
    """
    mndata = MNIST(MNIST_PATH)
    train_ims, train_labels = mndata.load_training()
    train_X = np.array(train_ims).T
    train_y = np.array(train_labels).T
    return train_X, train_y

def mnist_testing(shuffled = True):
    """
    Returns MNIST test data with N columns and d rows
    """
    mndata = MNIST(MNIST_PATH)
    test_ims, test_labels = mndata.load_testing()
    test_X = np.array(test_ims).T
    test_y = np.array(test_labels).T
    return test_X, test_y

def shuffle(X, y):
    shuffler = np.arange(len(y))
    np.random.shuffle(shuffler)
    X = X[:, shuffler]
    y = y[shuffler]
    return X, y
