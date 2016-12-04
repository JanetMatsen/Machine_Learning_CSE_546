import math
import matplotlib.pyplot as plt
import numpy as np

import pandas as pd

from TransferFunctions import LinearTF, TanhTF, ReLuTF

class NeuralNet:
    """
    Neural net classifier with one hidden layer
    """
    def __init__(self, X, y,
                 hidden_nodes,
                 hiddenTF,
                 outputTF,
                 minibatch_size,
                 eta0,
                 summarise_frequency = 1 # steps
                 ):
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
        self.eta0 = eta0
        self.eta = eta0
        self.steps = 0
        self.epochs = 0
        self.summarise_frequency = summarise_frequency

        self.results = pd.DataFrame()

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

    def feed_forward(self, X):
        # make sure the dimensions are (2, n_cols)
        assert X.ndim == 2, 'X shape is {}; need (2, # of points), ' \
                            'e.g. X[:, 0:1]'.format(X.shape)
        _, n_points = X.shape

        # combo of X weights coming into the hidden layer:
        z = self.W1.dot(X)
        self.hidden_z = z
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
        # A is the prediction for each class:
        # large and positive if yes, negative if no.

    def predict(self, X):
        self.feed_forward(X)
        return self.output_a  # also known as "predictions"

    def loss_function(self, X=None, y=None):
        pass

    def gradients(self):
        pass

    def step(self):

        pass

    def output(self):
        pass

    #def feed_forward(self, X, Y):
    #    Y_hat = self.predict(X)
    #    errors = Y - Y_hat
    #    print("Y: {}\n".format(Y))
    #    print("hat(Y): \n{}".format(Y_hat))
    #    self.errors = errors

    def backprop(self, X, Y):
        _, n_points = X.shape
        assert X.shape == Y.shape, "X and Y need to have same # of points"
        # check dim of Y matches stashed a, z dimensions.
        self.output_delta = -np.multiply(Y - self.output_a, # error at output
                                         self.outputTF.grad(self.output_z))
        assert self.output_delta.shape == (self.C, n_points)

        self.hidden_delta = np.multiply(self.W2.T.dot(self.output_delta),
                                        self.hiddenTF.grad(self.hidden_z))
        assert self.hidden_delta.shape == (self.hidden_n, n_points)

        #W2_grad = self.output_delta.dot(self.output_a.T)
        #W1_grad = self.hidden_delta.dot(self.hidden_a)
        W2_grad = self.output_delta.dot(self.hidden_a.T)
        W1_grad = self.hidden_delta.dot(X.T)

        return W2_grad, W1_grad

    def update_weights(self, W2_grad, W1_grad, n_pts):
        # TODO: some eta decay strategy.
        self.W2 += - self.eta*W2_grad
        self.W1 += - self.eta*W1_grad

    def run(self, epochs):
        print("loss before: \n")
        predictions = self.predict(self.X)
        print(self.square_loss(predictions, self.Y))

        # should *not* be written assuming it will only be called once
        # TODO: shuffle X, Y
        for epoch in range(epochs + 1):
            epoch_step = 0
            for step in range(self.N):
                print('step {} of epoch {}'.format(epoch_step, epoch))
                X = self.X[:, epoch_step-1:epoch_step]  # grab subset
                Y = self.Y[:, epoch_step-1:epoch_step]

                self.feed_forward(X)
                W2_grad, W1_grad = self.backprop(X, Y)
                self.update_weights(W2_grad, W1_grad, n_pts=X.shape[1])

                if self.steps%self.summarise_frequency == 0:
                    self.summarise()

                epoch_step += 1
                self.steps += 1
                self.epochs += 1

        print('Iterated {} epoch(s)'.format(epochs))

        print("loss after: \n")
        predictions = self.predict(self.X)
        print(self.square_loss(predictions, self.Y))

    def square_loss(self, Y_predicted, Y_truth):
        # given y, and hat{y}, how many are wrong?
        assert Y_predicted.shape == Y_truth.shape, 'shapes unequal'
        n_pts = Y_truth.shape[1]
        errors = Y_predicted - Y_truth
        errors_squared = np.multiply(errors, errors)
        squares_sum = errors_squared.sum()
        assert not math.isnan(squares_sum)
        return squares_sum

    def loss_01(self, y_predicted, y_truth):
        # given y, and hat{y}, how many are wrong?
        assert y_predicted.shape == y_truth.shape, 'shapes unequal'
        n_pts = y_truth.shape[1]
        return n_pts - np.equal(y_predicted, y_truth).sum()

    def predict_y_from_Y(self):
        # for a one-hot-encoded Y, predict y
        pass

    def summarise(self):
        results_row = {}

        results_row['epoch'] = self.epochs
        results_row['step'] = self.steps
        results_row['eta'] = self.eta
        results_row['eta0'] = self.eta0

        # TODO: will need to loop over the points to build up predictions
        square_loss = self.square_loss(self.Y, self.predict(self.Y))
        results_row['square loss'] = square_loss
        results_row['(square loss)/N'] = square_loss/self.N

        # TODO: will need to loop over the points to build up predictions
        loss_01 = self.loss_01(self.Y, self.predict(self.X))
        results_row['0/1 loss'] = loss_01
        results_row['(0/1 loss)/N'] = loss_01/self.N

        results_row = {k:[v] for k, v in results_row.items()}
        self.results = pd.concat([self.results, pd.DataFrame(results_row)])


    def plot_ys(self, x, y_value_list, ylabel=None, df=None,
                logx=True, logy=False, y0_line = False,
                colors=None, figsize=(4, 3), filepath=None):
        if df is None:
            assert self.results is not None
            df = self.results

        if logx and not logy:
            plot_fun = plt.semilogx
        elif logy and not logx:
            plot_fun = plt.semilogy
        elif logx and logy:
            plot_fun = plt.loglog
        else:
            plot_fun = plt.plot

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if colors is None:
            colors = ['#bcbddc', '#756bb1', '#74c476', '#006d2c']

        ys_plotted = 0
        for y in y_value_list:
            plot_fun(df[x], df[y], linestyle='--',
                         marker='o', color=colors[ys_plotted])
            ys_plotted += 1

        if y0_line:
            ax.axhline(y=0, color='k')

        plt.gca().set_ylim(bottom=0)

        plt.legend(loc = 'best')
        plt.xlabel(x)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            pass
        if filepath is not None:
            fig.savefig(filepath) # + '.pdf')
        return fig

    def plot_square_loss(self, x='epoch', y_values=['(square loss)/N'],
                         filepath=None):

        p = self.plot_ys(x=x, y_value_list=y_values, ylabel=y_values[0],
                         logx=False, logy=False, filepath=filepath)
        return p

    def plot_01_loss(self, x='epoch', y_values=['(0/1 loss)/N'],
                         filepath=None):

        p = self.plot_ys(x=x, y_value_list=y_values, ylabel=y_values[0],
                         logx=False, logy=False, filepath=filepath)
        return p



