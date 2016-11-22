import copy
import matplotlib.pyplot as plt
import numpy as np
import re
import scipy.sparse as sp

import pandas as pd

MNIST_PATH = '../../data/python-mnist/data'

class ClassificationBase:
    """
    Methods common to classification.
    """
    def __init__(self, X, y, W=None, scale_X=False,
                 sparse=False, binary=False):

        self.sparse = sparse
        self.binary = binary # should y be (N,) or (N,1)
        self.X = X
        self.N, self.d = self.X.shape
        self.y = y
        if self.y.shape == (self.N, ):
            self.y = np.reshape(self.y, newshape=(self.N, 1))
        # number of classes may be 2, or more than 2.
        self.C = np.unique(y).shape[0]

        # number of classes may be 2, or more than 2.
        self.C = np.unique(y).shape[0]

        # Sort the y values into columns.  1 if the Y was on for that column.
        # E.g. y = [1, 1, 0] --> Y = [[0, 1], [0, 1], [1, 0]]
        self.make_Y_from_y()

        if W is None:
            self.W = np.zeros(shape=(self.d, self.C))
        elif type(W) == np.ndarray:
            self.W = W
        else:
            assert False, "W is not None or a numpy array."
        assert self.W.shape == (self.d ,self.C), \
            "shape of W is {}".format(self.W.shape)

        # Filled in as the models are fit.
        self.results = pd.DataFrame()

    def make_Y_from_y(self):
        # Sort the y values into columns.  1 if the Y was on for that column.
        # E.g. y = [1, 1, 0] --> Y = [[0, 1], [0, 1], [1, 0]]
        if self.binary:
            return
        Y = np.zeros(shape=(self.N, self.C))
        Y[np.arange(len(self.y)), np.squeeze(self.y)] = 1
        if self.is_sparse():
            Y = sp.csc_matrix(Y)
        self.Y = Y
        assert self.Y.shape == (self.N, self.C)

    def copy(self, reset=True):
        model = copy.copy(self)
        if reset:
            model.reset_model()
        return model

    def reset_model(self):
        # TODO: move some of the least_squares_sgd stuff here
        raise NotImplementedError

    def get_weights(self):
        return self.W

    def predict(self, X=None, Y=None):
        raise NotImplementedError

    def loss_01(self):
        return self.pred_to_01_loss(self.predict())

    def pred_to_01_loss(self, class_calls):
        """
        + one point for every class that's incorrectly called.
        Works for both binary classifiers and multi-class.
        """
        y = self.y.copy()
        y = y.reshape(1, self.N)
        return self.N - np.equal(y, class_calls).sum()

    def num_nonzero_weights(self, z=0.001):
        nonzero_weights = np.absolute(self.get_weights()) > z
        return nonzero_weights.sum()

    def results_row(self):
        """
        Return interesting facts about the model.
        Used to return details about fit as the model fits.

        Note that the binary class below has it's own method for now
        because it's weights are w and these are W.  #todo: make W&w same.
        """
        loss_01 = self.loss_01()
        return {
            "weights": [self.get_weights().copy()],
            "training 0/1 loss": [loss_01],
            "training (0/1 loss)/N": [loss_01/self.N],
            "# nonzero weights": [self.num_nonzero_weights()]
        }

    def is_sparse(self):
        """
        Some models don't have a notion of sparsity.  This returns true only
        if the model does have a sparsity attribute, and that attribute
        is True.
        """
        if 'sparse' in self.__dict__.keys():
            if self.sparse:
                return True
        else:
            return False

    def replace_X_and_y(self, X, y):
        self.X = X
        self.N = X.shape[0] # num points may change.
        if self.is_sparse():
            self.X = sp.csc_matrix(X)
        if y.shape == (self.N, ) and not self.binary:
            self.y = np.reshape(y, newshape=(self.N, 1))
        else:
            self.y = y
        if not self.binary:
            self.make_Y_from_y()
            assert self.Y.shape[0] == y.shape[0]

    def replace_weights(self, new_weights):
        if "W" in self.__dict__.keys():
            if self.is_sparse():
                self.W = sp.csc_matrix(new_weights)
        elif "w" in self.__dict__.keys():
            self.w = new_weights

    def apply_model(self, X, y, data_name, use_W_bar=True):
        """
        Apply existing weights (for "base_model") to give predictions
        on different X data.
        """
        # need a new model to do this.  Don't reset steps, etc.
        new_model = self.copy(reset=False)
        new_model.replace_X_and_y(X, y)

        assert new_model.X.shape == X.shape
        assert new_model.N == X.shape[0]
        if new_model.binary:
            assert new_model.y.shape == (y.shape[0], )
        else:
            assert new_model.Y.shape[0] == y.shape[0]

        if use_W_bar and self.W_bar is not None:
            print("Using bar{W} to apply model to new data.")
            self.W = self.W_bar
        elif self.W_bar is None:
            print("Using W because bar{W} does not exist.")
        else:
            print("???")

        # not training the new model this time!
        results = new_model.results_row()
        # rename column names from "training" to data_name
        results = {re.sub("training", data_name, k): v
                   for k, v in results.items()}
        return results

    def plot_ys(self, x, y1, y2=None, ylabel=None,
                df=None, head_n=None, tail_n=None,
                logx=True, logy=False, y0_line = False,
                colors=None, figsize=(4, 3), filename=None):
        if df is None:
            assert self.results is not None
            df = self.results
        assert not (head_n is not None and tail_n is not None), \
            "Can't set the head and tail parameters for plotting."
        if head_n is not None:
            df = df.head(head_n)
        if tail_n is not None:
            df = df.head(tail_n)

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
            colors = ['c','b']

        plot_fun(df[x], df[y1], linestyle='--',
                     marker='o', color=colors[0])
        if y2:
            plot_fun(df[x], df[y2], linestyle='--',
                         marker='o', color=colors[1])

        if y0_line:
            ax.axhline(y=0, color='k')

        plt.gca().set_ylim(bottom=0)

        plt.legend(loc = 'best')
        plt.xlabel(x)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            pass
        if filename is not None:
            fig.savefig(filename + '.pdf')
        return fig

    def plot_01_loss(self, y="(0/1 loss)/N", ylabel="fractional 0/1 loss",
                     filename=None, logx=False, head_n=None, tail_n=None):
        fig = self.plot_ys(x='epoch', y1=y, ylabel=ylabel, logx=logx,
                           head_n=head_n, tail_n=tail_n)
        if filename:
            fig.savefig(filename + '.pdf')

    def plot_log_loss(self, y="-(log loss)", ylabel="negative(log loss)",
                      filename=None):
        # TODO: break into another class that's IterativeModel
        fig = self.plot_ys(x='epoch', y1=y, ylabel=ylabel)
        if filename:
            fig.savefig(filename + '.pdf')

    def plot_2_subplots(self, x, y1, y2, pandas=True, title=None):
        fig, axs = plt.subplots(2, 1, figsize=(5, 4))
        colors = ['c','b']
        if pandas:
            y1_range = (0, max(self.results[y1])*1.05)
            y2_range = (0, max(self.results[y2])*1.05)
            self.results.plot(kind='scatter', ax=axs[0], x=x, y=y1,
                              color=colors[0], logx=True, ylim=y1_range)
            self.results.plot(kind='scatter', ax=axs[1], x=x, y=y2,
                              color=colors[1], logx=True, ylim=y2_range)
        else: # use matplotlib
            x=self.results[x]
            y1=self.results[y1]
            y2=self.results[y2]
            axs[0].plot(x, y1, linestyle='--', marker='o', color=colors[0])
            # doing loglog for eta.
            axs[1].plot(x, y2, linestyle='--', marker='o', color=colors[1])

        axs[0].axhline(y=0, color='k')
        # fill 2nd plot
        axs[1].axhline(y=0, color='k')

        if title is not None:
            plt.title(title)
        plt.tight_layout()
        return fig

    def plot_loss_and_eta(self, pandas=True, logloss=False):
        if len([s for s in self.results.columns.tolist() if 'log loss' in s]) > 0:
            if logloss:
                loss = "-(log loss)/N, training"
            else:
                loss = "-(log loss), training"
        elif len([s for s in self.results.columns.tolist() if 'square loss' in s]) > 0:
            if logloss:
                loss = "(square loss)/N, training"
            else:
                loss = "(square loss), training"
        x = 'epoch'
        return self.plot_2_subplots(x=x, y1=loss, y2='eta', pandas=pandas)

    def plot_log_loss_normalized_and_eta(self, pandas=True):
        return self.plot_loss_and_eta(logloss=True, pandas=pandas)


    @staticmethod
    def shuffle(X, y):
        shuffler = np.arange(len(y))
        np.random.shuffle(shuffler)
        X = X.copy()[shuffler, :] # todo: not sure if .copy() is needed.
        y = y.copy()[shuffler]
        return X, y



class ModelFitException(Exception):
    def __init__(self, message):
        #self.message = message
        print(message)
