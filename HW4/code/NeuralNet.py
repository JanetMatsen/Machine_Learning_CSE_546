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
                 summarise_frequency = 1, # steps
                 convergence_delta = 0.01
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
        self.Y_hat = None # transfer_fun(W.dot(a_below)) # used in back prop, minibatch sized
        self.output_delta = None

        self.minibatch_size = minibatch_size
        self.eta0 = eta0
        self.eta = eta0
        self.steps = 0
        self.points_stepped = 0
        self.epochs = 0
        self.convergence_delta = convergence_delta
        self.converged = False
        self.summarise_frequency = summarise_frequency

        self.results = pd.DataFrame()
        self.W1_tracking = pd.DataFrame()
        self.W2_tracking = pd.DataFrame()

    def y_to_Y(self):
        '''
        One-hot encoding of Y.
        Sort the y values into columns.  1 if the Y was on for that column.
        E.g. y = [1, 1, 0] --> Y = [[0, 1], [0, 1], [1, 0]]
        '''
        Y = np.zeros(shape=(self.C, self.N)) # columns are data points
        #Y[np.squeeze(self.y), np.arange(len(self.y))] = 1
        Y[self.y, np.arange(len(self.y))] = 1
        assert Y.shape == (self.C, self.N)
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
        self.Y_hat = self.outputTF.f(self.output_z)
        assert self.Y_hat.shape == (self.C, n_points)
        # A is the prediction for each class:
        # large and positive if yes, negative if no.

    def feed_forward_and_predict_Y(self, X):
        self.feed_forward(X)
        return self.Y_hat  # also known as "predictions"

    def backprop(self, X, Y, debug=False):
        _, n_points = X.shape  # for checking dimensionality
        assert X.shape[1] == Y.shape[1], "X and Y need to have same # of points"
        # check dim of Y matches stashed a, z dimensions.

        if debug:
            import pdb; pdb.set_trace()
        self.output_delta = -np.multiply(Y - self.Y_hat,  # error at output
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
        self.W2 += - (self.eta/n_pts)*W2_grad
        self.W1 += - (self.eta/n_pts)*W1_grad

    def step(self, X, Y):
        self.feed_forward(X)
        W2_grad, W1_grad = self.backprop(X, Y)
        self.update_weights(W2_grad, W1_grad, n_pts=X.shape[1])

    def run(self, epochs):
        # turn off convergence so it will run:
        if self.converged:
            print("setting self.converged to False for re-run")
            self.converged = False

        predictions = self.feed_forward_and_predict_Y(self.X)
        print("loss before:")
        print(self.square_loss(predictions, self.Y))

        # should *not* be written assuming it will only be called once
        # TODO: shuffle X, Y
        for epoch in range(epochs + 1):
            epoch_step = 1
            num_pts = 0
            while num_pts < self.N:
                index_start = num_pts
                index_stop = num_pts + self.minibatch_size
                X = self.X[:, index_start:index_stop]  # grab subset
                Y = self.Y[:, index_start:index_stop]
                assert X.shape[1] == Y.shape[1], 'size mismatch for X and Y'
                num_pts += X.shape[1]
                self.points_stepped += X.shape[1]
                print('step {} of epoch {}'.format(epoch_step, epoch))

                self.step(X, Y)

                # check status sometimes.
                if self.points_stepped%self.summarise_frequency == 0:
                    self.summarise()
                    self.track_weights()
                    # check if it's time to exit the loop
                    if self.results.shape[0] >= 2:
                        self.test_convergence()
                    if self.converged:
                        print('model converged for eta = {}'.format(self.eta))
                        return

                epoch_step += 1
                self.steps += 1
                self.epochs += 1

        print('Iterated {} epoch(s)'.format(epochs))

        print("loss after: \n")
        predictions = self.feed_forward_and_predict_Y(self.X)
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
        n_pts = len(y_truth)
        return n_pts - np.equal(y_predicted, y_truth).sum()

    def predict_y_from_Y(self, Y):
        # for a one-hot-encoded Y, predict y
        y = np.argmax(Y, axis=0)
        return y

    def predict_y(self, X):
        Y_hat = self.feed_forward_and_predict_Y(X)
        y = self.predict_y_from_Y(Y_hat)
        return y

    def summarise(self):
        results_row = {}

        results_row['epoch'] = self.epochs
        results_row['step'] = self.steps
        results_row['points stepped'] = self.points_stepped
        results_row['eta'] = self.eta
        results_row['eta0'] = self.eta0
        results_row['converged'] = self.converged

        # TODO: will need to loop over the points to build up predictions
        Y_hat = self.feed_forward_and_predict_Y(self.X)
        square_loss = self.square_loss(self.Y, Y_hat)

        results_row['square loss'] = square_loss
        results_row['(square loss)/N'] = square_loss/self.N

        # TODO: will need to loop over the points to build up predictions
        y = self.predict_y_from_Y(Y_hat)
        loss_01 = self.loss_01(self.y, y)
        results_row['0/1 loss'] = loss_01
        results_row['(0/1 loss)/N'] = loss_01/self.N

        results_row = {k:[v] for k, v in results_row.items()}
        self.results = pd.concat([self.results, pd.DataFrame(results_row)])

    def track_weights(self):

        def prep_colnames(W):
            ravel_indices = []
            (rownum, colnum) = W.shape
            for r in range(rownum):
                for c in range(colnum):
                    ravel_indices.append((r, c))
            return ravel_indices

        def prep_df(W):
            colnames = prep_colnames(W)
            W = np.ravel(W)
            row = {}
            for c, w in zip(colnames, W):
                row[c] = w
            #row['epochs'] = self.epochs
            row['steps'] = self.steps
            #row['points stepped'] = self.points_stepped
            row = {k:[v] for k, v in row.items()}
            return pd.DataFrame(row)

        W1_row = prep_df(self.W1)
        W2_row = prep_df(self.W2)
        self.W1_tracking = pd.concat([self.W1_tracking, W1_row])
        self.W2_tracking = pd.concat([self.W2_tracking, W2_row])


    def test_convergence(self):
        """
        Test convergence using the last values in the Pandas summary
        """
        if self.results.shape[0] < 2:
            pass
        last_losses = self.results.tail(2)['square loss'].reset_index(drop=True)
        old_square_loss = last_losses[0]
        square_loss = last_losses[1]
        improvement = old_square_loss - square_loss
        if improvement < 0:
            print("warning: square loss increased from {} to {}".format(
                old_square_loss, square_loss))
            raise NeuralNetException("square loss grew to {}".format(square_loss))
        if abs(improvement)/self.N > 1000:
            print("warning: large improvement")
        if abs(improvement) < self.convergence_delta:
            self.converged = True

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

    #def plot_2_subplots(self, x1, y1, x2, y2, df1=None, df2=None,
    #                    pandas=True, title=None):
    #    if df1 is None:
    #        df1 = self.results
    #    if df2 is None:
    #        df2 = self.results

    #    fig, axs = plt.subplots(2, 1, figsize=(5, 4))
    #    colors = ['c','b']
    #    if pandas:
    #        y1_range = (0, max(self.results[y1])*1.05)
    #        y2_range = (0, max(self.results[y2])*1.05)
    #        df1.plot(kind='scatter', ax=axs[0], x=x1, y=y1,
    #                          color=colors[0], logx=True, ylim=y1_range)
    #        df2.plot(kind='scatter', ax=axs[1], x=x2, y=y2,
    #                          color=colors[1], logx=True, ylim=y2_range)
    #    else: # use matplotlib
    #        x1 = df1[x1]
    #        y1 = df1[y1]
    #        x2 = df1[x2]
    #        y2 = df2[y2]
    #        axs[0].plot(x1, y1, linestyle='--', marker='o', color=colors[0])
    #        # doing loglog for eta.
    #        axs[1].plot(x2, y2, linestyle='--', marker='o', color=colors[1])

    #    axs[0].axhline(y=0, color='k')
    #    # fill 2nd plot
    #    axs[1].axhline(y=0, color='k')

    #    if title is not None:
    #        plt.title(title)
    #    plt.tight_layout()
    #    return fig

    def plot_weight_evolution(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        self.W1_tracking.set_index('steps').plot(ax=ax[0])#, figsize=(3,3))
        ax[0].set_title("W1")

        self.W2_tracking.set_index('steps').plot(ax=ax[1])#, figsize=(3,3))
        ax[1].set_title("W2")
        return fig


class NeuralNetException(Exception):
    def __init__(self, message):
        #self.message = message
        print(message)




