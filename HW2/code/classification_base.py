from math import log
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ClassificationBase:
    """
    Methods common to classification.
    """
    def __init__(self, X, y, W=None):

        self.X = X #sp.csc_matrix(X)
        self.N, self.d = self.X.shape
        self.y = y
        if self.y.shape == (self.N, ):
            self.y = np.reshape(self.y, newshape=(self.N, 1))

        # number of classes may be 2, or more than 2.
        self.C = np.unique(y).shape[0]

        # Sort the y values into columns.  1 if the Y was on for that column.
        # E.g. y = [1, 1, 0] --> Y = [[0, 1], [0, 1], [1, 0]]
        Y = np.zeros(shape=(self.N, self.C))
        Y[np.arange(len(y)), y] = 1
        self.Y = Y
        assert self.y.shape == (self.N, 1)

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

    def pred_to_01_loss(self, class_calls):
        """
        + one point for every class that's correctly called.
        """
        # TODO: update for matrix form.
        y = self.y.copy()
        y = y.reshape(1, self.N)
        return self.N - np.equal(y, class_calls).sum()

    #def pred_to_normalized_01_loss(self, class_calls):
    #    return self.loss_01(class_calls)/self.N

    def plot_ys(self, x,y1, y2=None, ylabel=None):
        assert self.results is not None

        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        colors = ['c','b']
        plt.semilogx(self.results[x], self.results[y1],
                     linestyle='--', marker='o',
                     color=colors[1])
        if y2:
            plt.semilogx(self.results[x], self.results[y2],
                         linestyle='--', marker='o',
                         color=colors[3])
        plt.legend(loc = 'best')
        plt.xlabel(x)
        if ylabel:
            plt.ylabel(ylabel)
        else:
            pass
        return fig

    def plot_01_loss(self, ylabel = "fractional 0/1 loss", filename=None):
        fig = self.plot_ys(x='iteration', y1="(0/1 loss)/N")
        if filename:
            fig.savefig(filename + '.pdf')

    def plot_log_loss(self, ylabel = "negative(log loss)", filename=None):
        fig = self.plot_ys(x='iteration', y1="-(log loss)")
        if filename:
            fig.savefig(filename + '.pdf')

    def step(self):
        pass

    def run(self):
        pass


class ClassificationBaseBinary:
    """
    Methods common to classification.
    """
    def __init__(self, X, y, w=None, w0=0):

        self.X = X #sp.csc_matrix(X)
        self.N, self.d = self.X.shape
        self.y = y
        assert self.y.shape == (self.N, )

        # number of classes may be 2, or more than 2.
        self.C = np.unique(y).shape[0]

        if w is None:
            self.w = np.zeros(self.d)
        elif type(w) == np.ndarray:
            self.w = w
        else:
            assert False, "w is not None or a numpy array."
        assert self.w.shape == (self.d ,), \
            "shape of w is {}".format(self.w.shape)
        if w0 is None:
            #y_num = y.sum()
            #self.w0 = log(y_num/(self.N-y_num))
            self.w0 = 0
        else:
            self.w0 = w0

    def pred_to_01_loss(self, class_calls):
        """
        + one point for every class that's correctly called.
        """
        return self.N - np.equal(self.y, class_calls).sum()
