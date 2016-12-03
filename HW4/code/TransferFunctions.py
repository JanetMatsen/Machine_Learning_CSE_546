import numpy as np

class LinearTF:
    def __init__(self, n_nodes):
        self.name = 'linear'
        self.n = n_nodes
        pass

    @staticmethod
    def f(z):
        self.z = z
        return z

    @staticmethod
    def grad(z):
        return z

    def initialize_weights(self):
        pass


class TanhTF:
    def __init__(self):
        self.name = 'tanh'
        pass

    @staticmethod
    def f(z):
        """
        element-wise application of tanh
        """
        return np.tanh(z)

    @staticmethod
    def grad(z):
        """
        f′(z) = 1−(f(z))^2
        :param x: z
        :return: derivative of transfer function at z
        """
        # TODO: vectorize
        d, n = z.shape
        return np.ones(shape=(d,n)) - np.square(self.f(z))


class ReLuTF:
    def __init__(self):
        self.name = 'ReLu'
        pass

    @staticmethod
    def f(z):
        """
        element-wise application of ReLu
        """
        return z.clip(min=0)

    @staticmethod
    def grad(z):
        if z < 0:
            return 0
        else:
            return 1
