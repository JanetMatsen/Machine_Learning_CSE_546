import numpy as np

def prep_binary_classes(y, digit=2):
    return np.array([int(x==digit) for x in y])
