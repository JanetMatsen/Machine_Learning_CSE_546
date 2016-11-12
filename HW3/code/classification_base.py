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

    def copy(self):
        return copy.copy(self)

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
        self.X = X.copy()
        self.N = X.shape[0] # num points may change.
        if self.is_sparse():
            self.X = sp.csc_matrix(X)
        if y.shape == (self.N, ) and not self.binary:
            self.y = np.reshape(y, newshape=(self.N, 1))
        else:
            self.y = y.copy()
        if not self.binary:
            self.make_Y_from_y()
            assert self.Y.shape[0] == y.shape[0]

    def apply_model(self, X, y, data_name):
        """
        Apply existing weights (for "base_model") to give predictions
        on different X data.
        """
        # need a new model to do this.
        new_model = self.copy()
        new_model.replace_X_and_y(X, y)

        assert new_model.X.shape == X.shape
        assert new_model.N == X.shape[0]
        if new_model.binary:
            assert new_model.y.shape == (y.shape[0], )
        else:
            assert new_model.Y.shape[0] == y.shape[0]

        # not training the new model this time!
        results = new_model.results_row()
        # rename column names from "training" to data_name
        results = {re.sub("training", data_name, k): v
                   for k, v in results.items()}
        return results

    def plot_ys(self, x, y1, df=None, y2=None, ylabel=None, logx=True,
                colors=None, figsize=(4, 3)):
        if df is None:
            assert self.results is not None
            df = self.results

        fig, ax = plt.subplots(1, 1, figsize=figsize)
        if colors is None:
            colors = ['c','b']
        if logx:
            plt.semilogx(df[x], df[y1], linestyle='--',
                         marker='o', color=colors[0])
            if y2:
                plt.semilogx(df[x], df[y2], linestyle='--',
                             marker='o', color=colors[1])
        else:
            plt.plot(df[x], df[y1], linestyle='--',
                     marker='o', color=colors[0])
            if y2:
                plt.semilogx(df[x], df[y2], linestyle='--',
                             marker='o', color=colors[1])
        plt.legend(loc = 'best')
        plt.xlabel(x)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            pass
        return fig

    def plot_01_loss(self, y="(0/1 loss)/N", ylabel="fractional 0/1 loss",
                     filename=None):
        # TODO: break into another class that's IterativeModel
        fig = self.plot_ys(x='step', y1=y, ylabel=ylabel)
        if filename:
            fig.savefig(filename + '.pdf')

    def plot_log_loss(self, y="-(log loss)", ylabel="negative(log loss)",
                      filename=None):
        # TODO: break into another class that's IterativeModel
        fig = self.plot_ys(x='step', y1=y, ylabel=ylabel)
        if filename:
            fig.savefig(filename + '.pdf')

    def plot_2_subplots(self, x, y1, y2, pandas=True, title=None):
        fig, axs = plt.subplots(2, 1, figsize=(4, 3))
        colors = ['c','b']
        if pandas:
            self.results.plot(kind='scatter', ax=axs[0], x=x, y=y1,
                              color=colors[0], logx=True)
            self.results.plot(kind='scatter', ax=axs[1], x=x, y=y2,
                              color=colors[1], logx=True)
        else: # use matplotlib
            x=self.results[x]
            y1=self.results[y1]
            y2=self.results[y2]
            axs[0].semilogx(x, y1, linestyle='--', marker='o', color=colors[0])
            # doing loglog for eta.
            axs[1].loglog(x, y2, linestyle='--', marker='o', color=colors[1])

        axs[0].axhline(y=0, color='k')
        # fill 2nd plot
        axs[1].axhline(y=0, color='k')

        if title is not None:
            plt.title(title)
        plt.tight_layout()

    def plot_log_loss_and_eta(self, pandas=True):
            self.plot_2_subplots(x='step',
                                 y1='-(log loss), training',
                                 y2='eta',
                                 pandas=pandas)

    def plot_log_loss_normalized_and_eta(self, pandas=True):
            self.plot_2_subplots(x='step',
                                 y1='-(log loss)/N, training',
                                 y2='eta',
                                 pandas=pandas)

    @staticmethod
    def shuffle(X, y):
        shuffler = np.arange(len(y))
        np.random.shuffle(shuffler)
        X = X.copy()[shuffler, :] # todo: not sure if .copy() is needed.
        y = y.copy()[shuffler]
        return X, y



class ModelFitExcpetion(Exception):
    def __init__(self, message):
        print(message)
