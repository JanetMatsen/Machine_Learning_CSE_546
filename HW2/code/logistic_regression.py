import numpy as np
import sys
import pandas as pd

import matplotlib.pyplot as plt

from classification_base import ClassificationBase


class LogisticRegression(ClassificationBase):
    """
    Train *one* model.
    """

    def __init__(self, X, y, n0, lam, w=None, w0=None,
                 max_iter=10**6, delta=10e-4):
        # call the base class's methods first
        super(LogisticRegression, self).__init__(X, y, w, w0)
        self.nu = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta = delta

    def optimiize_weights(self):
        pass

    def apply_weights(self):
        """
        calc w0 + Xw
        This quantity is labeled q in my planning.
        :return: vetor of weights applied to X.
        """
        w0_array = np.ones(self.N)*self.w0
        return w0_array + self.X.dot(self.w)

    def probability_array(self):
        """
        Calculate the array of probabilities.
        :return: An Nx1 array.
        """
        q = self.apply_weights()
        return np.exp(q)/(1 + np.exp(q))

    def predict(self, threshold=0.5):
        """
        Produce an array of class predictions
        """
        probabilities = self.probability_array()
        classes = np.zeros(self.N)
        classes[probabilities > threshold] = 1
        return classes

    def loss_01(self):
        return self.pred_to_01_loss(self.predict())

    def log_loss(self):
        probabilities = self.probability_array().copy()
        # need to flip the probabilities for p < 0.5 with this binary case.
        # 1 - old_val is same as oldval*-1 + 1.  Do in 2 steps:
        probabilities[np.equal(0, self.y)] *= -1
        probabilities[np.equal(0, self.y)] += 1
        # when multiclass: np.amax(probabilities, 1)
        return np.log(probabilities).sum()

    def step(self):
        """
        Update the weights and bias
        """
        P = self.probability_array()
        E = self.y - P  # prediction error (0 to 1)

        self.w0 += (self.nu/self.N**0.5)*E.sum()
        assert self.w0.shape == ()
        assert isinstance(self.w0, np.float64)

        self.w += (self.nu/self.N**0.5)*(-self.lam*self.w + self.X.T.dot(E))
        assert self.w.shape == (self.d ,), \
            "shape of w is {}".format(self.w.shape)

    def shrink_nu(self):
        self.nu = self.nu*0.99 # may want to scale w/ batch size.

    def run(self):
        results = pd.DataFrame()

        # Step until converged
        for s in range(1, self.max_iter+1):
            self.shrink_nu()
            old_loss_normalized = self.loss_01()/self.N
            old_w = self.w.copy()

            self.step()
            sys.stdout.write(".")

            new_loss = self.loss_01()
            new_loss_normalized = self.loss_01()/self.N
            one_val = pd.DataFrame({"iteration": [s],
                                    #"probability 1": [self.probability_array()],
                                    "0/1 loss": [new_loss],
                                    "(0/1 loss)/N": [new_loss_normalized],
                                    "-(log loss)": [-self.log_loss()],
                                    "log loss": [self.log_loss()]})
            results = pd.concat([results, one_val])

            assert not self.has_increased_significantly(
                old_loss_normalized, new_loss),\
                "Normalized loss: {} --> {}".format(old_loss_normalized, new_loss)
            if abs(old_w - self.w).max() < self.delta:
                break
        self.results = results

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)

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

        #, y2='-(log loss)"')
        #return fig



