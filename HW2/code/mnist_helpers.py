import numpy as np

from mnist import MNIST

from classification_base import MNIST_PATH

def prep_binary_classes(y, digit=2):
    return np.array([int(x==digit) for x in y])

def mnist_training(shuffled=True):
    mndata = MNIST(MNIST_PATH)
    train_ims, train_labels = mndata.load_training()
    train_X = np.array(train_ims)
    train_y = np.array(train_labels)
    if shuffled:
        return shuffle(train_X, train_y)
    else:
        return train_X, train_y

def mnist_testing(shuffled = True):
    mndata = MNIST(MNIST_PATH)
    test_ims, test_labels = mndata.load_testing()
    test_X = np.array(test_ims)
    test_y = np.array(test_labels)
    if shuffled:
        return shuffle(test_X, test_y)
    else:
        return test_X, test_y

def mnist_training_binary(num, shuffled=True):
    X, y = mnist_training()
    if shuffled:
        X, y = shuffle(X, y)
    return X, np.array([int(yi==num) for yi in y])

def mnist_testing_binary(num):
    X, y = mnist_testing()
    return X, np.array([int(yi==num) for yi in y])

def shuffle(X, y):
    shuffler = np.arange(len(y))
    print(shuffler)
    np.random.shuffle(shuffler)
    X = X[shuffler, :]
    y = y[shuffler]
    return X, y

def shuffle_train_and_test(X1, y1, X2, y2):
    print('Warning: shuffling train and test should not be used for work '
          'that is to be submitted.')
    np.random.seed(12345)
    all_X = np.concatenate((X1, X2))
    all_y = np.concatenate((y1, y2))
    X, y = shuffle(all_X, all_y)

    X1_shuffled = X[0:X1.shape[0], :]
    y1_shuffled = y[0:X1.shape[0]]
    X2_shuffled = X[X1.shape[0]:, :]
    y2_shuffled = y[X1.shape[0]:]
    return X1_shuffled, y1_shuffled, X2_shuffled, y2_shuffled

def binary_shuffled(num):
    X1, y1 = mnist_training_binary(num)
    X2, y2 = mnist_testing_binary(num)
    return shuffle_train_and_test(X1, y1, X2, y2)
