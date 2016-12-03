import numpy as np

from TransferFunctions import LinearTF, TanhTF, ReLuTF

class NeuralNet:
    """
    Neural net classifier with one hidden layer
    """
    def __init__(self, X, y,
                 hidden_nodes,
                 hiddenTF,
                 outputTF,
                 minibatch_size):
        self.X = X  # columns are data points, rows are features
        self.y = y
        self.d, self.N = X.shape
        self.C = np.unique(y).shape[0]
        self.Y = self.y_to_Y()
        assert self.Y.shape == (self.C, self.N)


        self.hiddenTF = hiddenTF(n_in = self.d, n_nodes=hidden_nodes)
        self.hidden_n = hidden_nodes
        # Weights to multiply incoming x by; feeds the hidden layer
        self.W1 = self.hiddenTF.initialize_weights()  # Shape = (# hidden nodes, self.d)
        #self.hidden_z = None # W.dot(X)  (minibatch sized)
        self.hidden_a = None # transfer_fun(W.dot(X)) # used in feed forward, minibatch sized
        self.hidden_delta = None

        self.outputTF = outputTF(n_in = hidden_nodes, n_nodes = self.C)
        # Weights between the hidden layer and the output layer.
        # Shape = (self.d, n_above = self.C)
        self.W2 = self.outputTF.initialize_weights()
        self.output_z = None # W.dot(a_below) # used in back prop (minibatch sized)
        self.output_a = None # transfer_fun(W.dot(a_below)) # used in back prop, minibatch sized
        self.output_delta = None

        self.minibatch_size = minibatch_size
        self.step = 0

    def y_to_Y(self):
        '''
        One-hot encoding of Y.
        Sort the y values into columns.  1 if the Y was on for that column.
        E.g. y = [1, 1, 0] --> Y = [[0, 1], [0, 1], [1, 0]]
        '''
        Y = np.zeros(shape=(self.C, self.N)) # columns are data points
        #Y[np.squeeze(self.y), np.arange(len(self.y))] = 1
        Y[self.y, np.arange(len(self.y))] = 1
        return Y

    def loss_function(self, X=None, y=None):
        pass

    def gradients(self):
        pass

    def step(self):
        pass

    def output(self):
        pass

    def feed_forward(self, X):
        # make sure the dimensions are (2, n_cols)
        assert X.ndim == 2, 'X shape is {}; need (2, # of points), ' \
                            'e.g. X[:, 0:1]'.format(X.shape)
        _, n_points = X.shape

        # combo of X weights coming into the hidden layer:
        z = self.W1.dot(X)
        assert z.shape == (self.hidden_n, n_points)
        # activate z according to the hidden layer's activation function
        self.hidden_a = self.hiddenTF.f(z)
        assert self.hidden_a.shape == (self.hidden_n, n_points)

        # linear combo of hidden layer weights for output layer:
        self.output_z = self.W2.dot(self.hidden_a)
        assert self.output_z.shape == (self.C, n_points)
        # activate according to the output layer's transfer function
        self.output_a = self.outputTF.f(self.output_z)
        assert self.output_a.shape == (self.C, n_points)

        # predict using output's activated values
        #predictions = self.predict(self.hidden_a)

    def out_layer_grad(self):
        diff = Y - Y_hat  # [Y - hat{Y}]
        grad = self.outputTF.grad(self.output_z)  # f'(z^(n_l)
        # element-wise multiplication:
        return np.multiply(diff, grad)

    def backprop(self, Y):
        # check dim of Y matches stashed a, z dimensions.
        pass

    def run(self, steps):
        # should *not* be written assuming it will only be called once

        # TODO: shuffle X, Y
        for i in range(steps):
            X = None  # grab subset
            Y = None  # grab subset
            step += 1

            self.feed_forward(X)
            self.backprop(Y)

        pass

    def predict_y_from_Y(self):
        # for a one-hot-encoded Y, predict y
        pass

    def loss_01(self):
        # given y, and hat{y}, how many are wrong?
        pass

