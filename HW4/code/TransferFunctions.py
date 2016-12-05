import numpy as np


class TF(object):
    def __init__(self, n_in, n_nodes, scale_W1=1, scale_W2=1):
        self.n_in = n_in
        self.n_nodes = n_nodes
        self.W_shape = (self.n_nodes, self.n_in)
        self.scale_W1 = scale_W1
        self.scale_W2 = scale_W2

    def initialize_weights_X_norm_squared(self, neural_net):
        X = neural_net.X
        d = np.linalg.norm(X) # ||X||^2
        weights = np.random.normal(0, 1/d, size = self.W_shape)

        print('normalize W1 by {}'.format(self.scale_W1))
        return weights/self.scale_W1


class LinearTF(TF):
    def __init__(self, n_in, n_nodes, scale_W1=None, scale_W2=None):
        super(LinearTF, self).__init__(n_in=n_in,
                                       n_nodes=n_nodes,
                                       scale_W1=scale_W1,
                                       scale_W2=scale_W2)
        self.name = 'linear'

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


class TanhTF(TF):
    def __init__(self, n_in, n_nodes, scale_W1=1, scale_W2=1):
        super(TanhTF, self).__init__(n_in=n_in,
                                     n_nodes=n_nodes,
                                     scale_W1=scale_W1,
                                     scale_W2=scale_W2)
        self.name = 'tanh'

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
        return super(TanhTF, self).initialize_weights_X_norm_squared(neural_net)

    def initialize_W2(self, neural_net):
        weights = np.random.normal(0, 1/self.n_nodes**2, size=self.W_shape)
        return weights/self.scale_W2


class ReLuTF(TF):
    def __init__(self, n_in, n_nodes, scale_W1=1, scale_W2=1):
        super(ReLuTF, self).__init__(n_in=n_in, n_nodes=n_nodes,
                                     scale_W1=scale_W1, scale_W2=scale_W2)
        self.name = 'ReLu'

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
        return super(ReLuTF, self).initialize_weights_X_norm_squared(neural_net)

    def initialize_W2(self, neural_net):
        # Want E[Y] <= 0.1 E[Y]
        # What is E[Y]?
        E = np.mean(neural_net.Y, axis=1)
        E = E/self.scale_W2
        print("E[Y]:\n{}".format(E))
        return E/neural_net.C

