import numpy as np

class LinearTF:
    def __init__(self, n_in, n_nodes):
        self.name = 'linear'
        self.n_in = n_in
        self.n_nodes = n_nodes
        pass

    @staticmethod
    def f(z):
        return z

    @staticmethod
    def grad(z):
        return 1

    def initialize_weights(self):
        return np.random.normal(0, 1, size=(self.n_nodes, self.n_in))


class TanhTF:
    def __init__(self, n_in, n_nodes):
        self.name = 'tanh'
        self.n_in = n_in
        self.n_nodes = n_nodes
        pass

    @staticmethod
    def f(z):
        """
        element-wise application of tanh
        """
        return np.tanh(z)

    def grad(self, z):
        """
        f′(z) = 1−(f(z))^2
        :param x: z
        :return: derivative of transfer function at z
        """
        # TODO: vectorize
        d, n = z.shape
        return np.ones(shape=(d,n)) - np.square(self.f(z))


class ReLuTF:
    def __init__(self, n_in, n_nodes):
        self.name = 'ReLu'
        self.n_in = n_in
        self.n_nodes = n_nodes
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
