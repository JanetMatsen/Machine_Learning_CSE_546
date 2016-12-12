import numpy as np


class TF(object):
    def __init__(self, n_in, n_nodes, scale_W1=1, scale_W2=1,
                 W1_init_strategy=None, W2_init_strategy=None):
        self.n_in = n_in
        self.n_nodes = n_nodes
        self.W_shape = (self.n_nodes, self.n_in)
        self.scale_W1 = scale_W1
        self.scale_W2 = scale_W2
        self.W1_init_strategy = W1_init_strategy
        self.W2_init_strategy = W2_init_strategy

    def initialize_weights_X_norm_squared(self, neural_net):
        X = neural_net.X
        d = np.linalg.norm(X) # ||X||^2
        weights = np.random.normal(0, 1/d, size = self.W_shape)

        print('normalize W1 by {}'.format(self.scale_W1))
        return weights/self.scale_W1

    def initialize_Xavier(self, neural_net, hidden=True):
        # Var(W_i) = 1/n_in
        # np.random.normal(0, 1/n_in**0.5, size = self.W_shape)
        # http://andyljones.tumblr.com/post/110998971763/an-explanation-of-xavier-initialization
        print('initializing with Xavier')
        if hidden:
            denom = neural_net.d**0.5
        else:  # TODO
            print("Xavier not set up for output")
            raise Exception
        return np.random.normal(0, 1/denom, size = self.W_shape)


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
        W = np.random.normal(0, 1, size=(self.n_nodes, self.n_in))
        print('scale W1 by {}'.format(self.scale_W1))
        return W/self.scale_W1

    def initialize_W2(self, neural_net):
        W = np.random.normal(0, 1, size=(self.n_nodes, self.n_in))
        print('scale W2 by {}'.format(self.scale_W2))
        return W*1./self.scale_W2


class TanhTF(TF):
    def __init__(self, n_in, n_nodes, scale_W1=1, scale_W2=1,
                 W1_init_strategy=None, W2_init_strategy=None):
        super(TanhTF, self).__init__(n_in=n_in,
                                     n_nodes=n_nodes,
                                     scale_W1=scale_W1,
                                     scale_W2=scale_W2,
                                     W1_init_strategy=W1_init_strategy,
                                     W2_init_strategy=W2_init_strategy)
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
        if self.W1_init_strategy is None:
            W = super(TanhTF, self).initialize_weights_X_norm_squared(neural_net)
        elif self.W1_init_strategy == 'Xavier':
            W = super(TanhTF, self).initialize_Xavier(neural_net)
        return W/self.scale_W1

    def initialize_W2(self, neural_net):
        weights = np.random.normal(0, 1/self.n_nodes**2, size=self.W_shape)
        return weights/self.scale_W2


class ReLuTF(TF):
    def __init__(self, n_in, n_nodes, scale_W1=1, scale_W2=1,
                 W1_init_strategy=None, W2_init_strategy=None):
        super(ReLuTF, self).__init__(n_in=n_in, n_nodes=n_nodes,
                                     scale_W1=scale_W1, scale_W2=scale_W2,
                                     W1_init_strategy=W1_init_strategy,
                                     W2_init_strategy=W2_init_strategy)
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
        W = super(ReLuTF, self).initialize_weights_X_norm_squared(neural_net)
        # W is already normalized in parent method
        return W

    def initialize_W2(self, neural_net):
        # Want E[Y] <= 0.1 E[Y]
        # What is E[Y]?
        E = np.mean(neural_net.Y, axis=1)
        E = E/self.scale_W2
        print("E[Y]:\n{}".format(E))
        return E/neural_net.C

