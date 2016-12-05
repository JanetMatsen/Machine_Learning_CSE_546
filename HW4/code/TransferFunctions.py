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

    def initialize_W1(self, neural_net):
        # note: X is not used
        return np.random.normal(0, 1, size=(self.n_nodes, self.n_in))

    def initialize_W2(self, neural_net):
        return self.initialize_W1(neural_net)


class TanhTF:
    def __init__(self, n_in, n_nodes):
        self.name = 'tanh'
        self.n_in = n_in
        self.n_nodes = n_nodes
        self.W_shape = (self.n_nodes, self.n_in)
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
        d, n = z.shape
        return np.ones(shape=(d,n)) - np.square(self.f(z))

    def initialize_W1(self, neural_net):
        X = neural_net.X
        norm = np.linalg.norm(X)
        norm_squared =  np.multiply(norm, norm)
        return np.random.normal(0, 1/norm_squared, size = self.W_shape)

    def initialize_W2(self, neural_net):
        return np.random.normal(0, 1/self.n_nodes**2, size=self.W_shape)


class ReLuTF:
    def __init__(self, n_in, n_nodes):
        self.name = 'ReLu'
        self.n_in = n_in
        self.n_nodes = n_nodes
        self.W_shape = (self.n_nodes, self.n_in)
        pass

    @staticmethod
    def f(z):
        """
        element-wise application of ReLu
        """
        return z.clip(min=0)

    @staticmethod
    def grad(z):
        return np.maximum(z, 0, z)

    def initialize_W1(self, neural_net):
        X = neural_net.X
        norm = np.linalg.norm(X)
        norm_squared =  np.multiply(norm, norm)
        return np.random.normal(0, 1/norm_squared, size = self.W_shape)

    def initialize_W2(self, neural_net):
        # Want E[Y] <= 0.1 E[Y]
        # What is E[Y]?
        E = np.mean(neural_net.Y, axis=1)
        print("E[Y]:\n{}".format(E))
        return E/neural_net.C


