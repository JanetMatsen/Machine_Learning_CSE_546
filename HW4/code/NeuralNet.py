import copy
import math
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy as sp
import sys

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
                 summarise_frequency = None, # unit of steps
                 convergence_delta = 0.01,
                 verbose=False,
                 X_test = None,
                 y_test = None,
                 monitor_test_data = True,
                 PCA = None # load HW3 pickle for plotting
                 ):
        self.X = X  # columns are data points, rows are features
        self.y = y
        self.d, self.N = X.shape
        self.C = np.unique(y).shape[0]
        self.Y = self.y_to_Y()
        assert self.Y.shape == (self.C, self.N)

        if X_test is not None:
            self.X_test = X_test
            self.y_test = y_test
        self.monitor_test_data = monitor_test_data
        if X_test is None or y_test is None:
            self.monitor_test_data = False

        self.hiddenTF = hiddenTF(n_in = self.d, n_nodes=hidden_nodes)
        self.hidden_n = hidden_nodes
        # Weights to multiply incoming x by; feeds the hidden layer
        # TODO: randomly sample X once I am working with bigger data
        self.W1 = self.hiddenTF.initialize_W1(self)  # Shape = (# hidden nodes, self.d)
        #self.hidden_z = None # W.dot(X)  (minibatch sized)
        self.hidden_a = None # transfer_fun(W.dot(X)) # used in feed forward, minibatch sized
        self.hidden_delta = None

        self.outputTF = outputTF(n_in = hidden_nodes, n_nodes = self.C)
        # Weights between the hidden layer and the output layer.
        # Shape = (self.d, n_above = self.C)
        # TODO: randomly sample X once I am working with bigger data
        self.W2 = self.outputTF.initialize_W2(self)
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
        if summarise_frequency is None:
            if self.N < 100:
                self.summarise_frequency = 1 # check every step
            else:
                self.summarise_frequency = int(self.N/2)
        else:
            self.summarise_frequency = summarise_frequency

        assert self.summarise_frequency%minibatch_size == 0, \
            "won't monitor as much as you think"

        self.results = pd.DataFrame()
        self.W1_tracking = pd.DataFrame()
        self.W2_tracking = pd.DataFrame()
        self.verbose = verbose

        self.PCA = PCA # load HW3 pickle for plotting

    def copy(self):
        # TODO: not deep for some purposes.
        model = copy.deepcopy(self)
        # TODO: clear out DF?
        return model

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
        # Todo: break into chunks if N for this x is large (??)
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

    def backprop(self, X, Y):
        _, n_points = X.shape  # for checking dimensionality
        assert X.shape[1] == Y.shape[1], "X and Y need to have same # of points"
        # check dim of Y matches stashed a, z dimensions.

        self.output_delta = -np.multiply(Y - self.Y_hat,  # error at output
                                         self.outputTF.grad(self.output_z))
        assert self.output_delta.shape == (self.C, n_points)

        self.hidden_delta = np.multiply(self.W2.T.dot(self.output_delta),
                                        self.hiddenTF.grad(self.hidden_z))
        assert self.hidden_delta.shape == (self.hidden_n, n_points)

        W1_grad = self.hidden_delta.dot(X.T)
        W2_grad = self.output_delta.dot(self.hidden_a.T)

        return W1_grad, W2_grad

    def gradients(self, X, Y, rounded=False):
        self.feed_forward(X)
        W1_grad, W2_grad = self.backprop(X, Y)

        if rounded:
            return np.around(W1_grad,2), np.around(W2_grad, 2)
        else:
            return W1_grad, W2_grad

    def numerical_derivative_of_element(self, W_name, i,j):
        """
        Compute the numerical derivative for a single element of a single
        weight matrix.
        :param W_name: which weight matrix to alter: 'W1' or 'W2'
        :param i: the row of the element to get the derivative of
        :param j: the column to get the derivative of
        :return: the numerical derivative for one weight element.
        """
        # TODO: doesn't scale to larger data sets.  E.g. 60k data points.
        n = self.copy()

        W = getattr(n, W_name)
        x0 = W[i,j]

        def f(x):
            W[i,j] = x
            Y_hat = n.feed_forward_and_predict_Y(n.X)
            return n.square_loss(n.Y, Y_hat)

        return sp.misc.derivative(f, x0, 0.01)/2

    def numerical_derivatives(self, rounded=True):
        def derivative(W_name):
            W = getattr(self, W_name)
            d = np.zeros(shape = W.shape)
            (rownum, colnum) = W.shape
            for r in range(rownum):
                for c in range(colnum):
                    d[r, c] = \
                        self.numerical_derivative_of_element(W_name, r, c)
            return d

        W1_deriv = derivative('W1')
        W2_deriv = derivative('W2')
        if rounded:
            return np.around(W1_deriv, 2), np.around(W2_deriv,2)
        else:
            return W1_deriv, W2_deriv

    def update_weights(self, W1_grad, W2_grad, n_pts):
        # TODO: some eta decay strategy.
        assert n_pts > 0
        self.W1 += - (self.eta/n_pts)*W1_grad
        self.W2 += - (self.eta/n_pts)*W2_grad

    def step(self, X, Y):
        W1_grad, W2_grad = self.gradients(X, Y)
        n_pts = X.shape[1]
        self.update_weights(W1_grad, W2_grad, n_pts=n_pts)

    def run(self, epochs):
        # turn off convergence so it will run:
        if self.converged:
            print("setting self.converged to False for re-run")
            self.converged = False

        predictions = self.feed_forward_and_predict_Y(self.X)
        print("loss before:")
        print(self.square_loss(predictions, self.Y))

        # shuffle X, Y
        X_shuffled, Y_shuffled = self.shuffle(self.X.copy(), self.Y.copy())
        assert X_shuffled.shape == self.X.shape
        assert Y_shuffled.shape == self.Y.shape

        for epoch in range(epochs + 1):
            epoch_step = 1
            num_pts = 0
            while num_pts < self.N:
                index_start = num_pts
                index_stop = num_pts + self.minibatch_size
                X = X_shuffled[:, index_start:index_stop]  # grab subset
                Y = Y_shuffled[:, index_start:index_stop]
                assert X.shape[1] == Y.shape[1], 'size mismatch for X and Y'
                num_pts += X.shape[1]
                self.points_stepped += X.shape[1]
                if self.verbose:
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
                self.epochs = self.points_stepped/self.N

            sys.stdout.write(".")

        print('Iterated {} epoch(s)'.format(epochs))

        print("loss after: \n")
        predictions = self.feed_forward_and_predict_Y(self.X)
        print(self.square_loss(predictions, self.Y))

    @staticmethod
    def shuffle(X, Y):
        assert X.shape[1] == Y.shape[1]
        shuffler = np.arange(Y.shape[1])
        np.random.shuffle(shuffler)
        X = X.copy()[:, shuffler]
        Y = Y.copy()[:, shuffler]
        return X, Y

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

    def build_up_Y_hat(self, chunk_size = 1000):
        n = chunk_size
        n_done = 0

        while n_done < self.N:
            X_chunk = self.X[:, n_done:n_done + n]
            Y_hat_chunk = self.feed_forward_and_predict_Y(X_chunk)
            if n_done == 0:
                Y_hat = Y_hat_chunk
            else:
                Y_hat = np.hstack([Y_hat, Y_hat_chunk])
            n_done += X_chunk.shape[1]
        assert(Y_hat.shape[0] == self.C)
        return Y_hat

    def summary_row(self):
        results_row = {}

        results_row['epoch'] = self.epochs
        results_row['step'] = self.steps
        results_row['points stepped'] = self.points_stepped
        results_row['eta'] = self.eta
        results_row['eta0'] = self.eta0
        results_row['converged'] = self.converged

        # TODO: will need to loop over the points to build up predictions
        if self.N > 1000:
            # build up Y_hat in sets of n points
            Y_hat = self.build_up_Y_hat()

        else:
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
        return pd.DataFrame(results_row)

    def assess_test_data(self):
        test_model = self.copy()
        test_model.data_type = 'testing data loaded'
        test_model.d, test_model.N = self.X_test.shape
        test_model.X = test_model.X_test
        test_model.y = test_model.y_test
        test_model.Y = test_model.y_to_Y()
        return test_model.summary_row()

    def summarise(self):
        results_row = self.summary_row()
        colnames = [c + ', training' if 'loss' in c else c for c in
                    results_row.columns]
        results_row.columns = colnames

        if self.monitor_test_data:
            merge_col = 'step'
            test_results = self.assess_test_data()
            cols_to_keep = [c for c in test_results.columns if
                            ('loss' in c)]
            cols_to_keep.append(merge_col)
            test_results = test_results[cols_to_keep]
            colnames = test_results.columns.tolist()
            colnames = [c + ', testing' if 'loss' in c else c
                        for c in colnames]
            test_results.columns = colnames
            # rename rows with 'loss' in them to have ' testing' at the end
            results_row = pd.merge(results_row, test_results)

            # only keep some columns
        self.results = pd.concat([self.results, results_row])

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
        last_losses = \
            self.results.tail(2)['square loss, training'].reset_index(drop=True)
        old_square_loss = last_losses[0]
        square_loss = last_losses[1]
        improvement = old_square_loss - square_loss
        percent_improvement = (improvement)/old_square_loss*100

        if self.verbose:
            print('Loss improvement: {} - {} --> {}'.format(
                old_square_loss, square_loss, improvement))
        if improvement < 0:
            print("warning: square loss increased "
                  "{0:.2f}%;".format( -percent_improvement) +
                  " {} --> {}".format(old_square_loss, square_loss))
            if percent_improvement < -10:
                raise NeuralNetException("square loss grew to {0:.2f}"
                                         "".format(square_loss))

        if abs(improvement)/self.N > 1000:
            print("large improvement: {}%".format(percent_improvement))
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
            #colors = ['#bcbddc', '#756bb1', '#74c476', '#006d2c']
            colors = ['#939393', '#006d2c']

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

    def plot_square_loss(self, x='epoch', filepath=None, normalized=True):

        y_values = [c for c in self.results.columns if 'square loss' in c]
        if normalized:
            y_values = [c for c in y_values if "/N" in c]
            y_label = '(square loss)/N'
        else:
            y_values = [c for c in y_values if "/N" not in c]
            y_label = 'square loss)'

        p = self.plot_ys(x=x, y_value_list=y_values, ylabel=y_label,
                         logx=False, logy=False, filepath=filepath)
        return p

    def plot_01_loss(self, x='epoch', filepath=None, normalized=True):

        y_values = [c for c in self.results.columns if '0/1 loss' in c]

        if normalized:
            y_values = [c for c in y_values if "/N" in c]
            y_label = '(0/1 loss)/N'
        else:
            y_values = [c for c in y_values if "/N" not in c]
            y_label = '0/1 loss'

        p = self.plot_ys(x=x, y_value_list=y_values, ylabel=y_label,
                         logx=False, logy=False, filepath=filepath)
        return p

    def plot_weight_evolution(self):
        fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
        self.W1_tracking.set_index('steps').plot(ax=ax[0])#, figsize=(3,3))
        ax[0].set_title("W1 element weights")

        self.W2_tracking.set_index('steps').plot(ax=ax[1])#, figsize=(3,3))
        ax[1].set_title("W2 element weights")

        # remove the legend if > 10 points
        if self.W1_tracking.shape[1] > 10:
            ax[0].legend_.remove()
        if self.W2_tracking.shape[1] > 10:
            ax[1].legend_.remove()

        return fig

    def display_hidden_node_as_image(self, weights, filename=None):
        assert self.PCA is not None, "need PCA pickle loaded for use"
        assert weights.shape == (50,), "expected shape (50,); " \
                                       "got {}".format(weights.shape)
        print(weights)

        # Take it out of PCA space.
        image_vector = self.PCA.transform_number_up(weights, center=False)

        def make_image(data, path=None):
            plt.figure(figsize=(0.7,0.7))
            p=plt.imshow(data.reshape(28, 28), origin='upper', interpolation='none')
            p.set_cmap('gray_r')
            plt.axis('off')
            if path is not None:
                plt.savefig(path)
                plt.close()

        make_image(image_vector, filename)

    def visualize_10_W1_weights(self):
        random_indices = np.random.choice(range(self.W1.shape[1]),
                                          size=(10,), replace=False)
        # select 10 at random
        # save images for each
        # stitch them togethe with subprocess.check_call() 
        pass


def make_dir(path):
    if not os.path.exists(path):
        print('make dir')
        os.mkdir(path)
    else:
        print('path {} exists'.format(path))

class NeuralNetException(Exception):
    def __init__(self, message):
        #self.message = message
        print(message)




