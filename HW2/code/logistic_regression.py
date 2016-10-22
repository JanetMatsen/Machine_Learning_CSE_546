import numpy as np
import sys
import pandas as pd

import matplotlib.pyplot as plt

from classification_base import ClassificationBase
from classification_base import ClassificationBaseBinary


class LogisticRegression(ClassificationBase):
    """
    Train *one* model.
    """

    def __init__(self, X, y, n0, lam, W=None, W0=None,
                 max_iter=10**6, delta=1e-6):
        # call the base class's methods first
        super(LogisticRegression, self).__init__(X, y, W, W0)
        self.nu_init = n0
        self.nu = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta = delta

    def optimiize_weights(self):
        pass

    def apply_weights(self):
        """
        calc W0 + XW
        This quantity is labeled q in my planning.
        :return: vetor of Weights applied to X.
        """
        # Make a matrix version of W0 where the weights are repeated C times.
        W0_matrix = np.ones(shape=(self.N, self.C))*self.W0
        # All rows should be identical.
        assert (W0_matrix == W0_matrix [0]).all(), \
            "W0 matrix is supposed to have all rows identical"

        return W0_matrix + self.X.dot(self.W)

    def probability_array(self):
        """
        Calculate the array of probabilities.
        :return: An Nx1 array.
        """
        R = np.exp(self.apply_weights())
        assert R.shape == (self.N, self.C)
        # When you sum across a row, axis=1
        R_sum_ax1 = np.sum(R, axis=1)
        R_sum_ax1 = np.reshape(R_sum_ax1, newshape=(self.N, 1))
        assert R_sum_ax1.shape == (self.N, 1)

        # Divide each element in R by
        return R/R_sum_ax1

    def predict(self):
        """
        Produce an array of class predictions
        """
        probabilities = self.probability_array()
        # THIS ASSUMES the classifiers are in order: 0th column of the
        # probabilities corresponds to label = 0, ..., 9th col is for 9.
        classes = np.argmax(probabilities, axis=1)
        return classes

    def loss_01(self):
        return self.pred_to_01_loss(self.predict())

    def log_loss(self):
        probabilities = self.probability_array().copy()
        # get just the probability for the correct label.
        probabilities = np.multiply(self.Y, probabilities)
        # collapse it into an Nx1 array:
        probabilities = np.amax(probabilities, axis=1)
        #probabilities = np.reshape(probabilities, newshape=(self.N, 1))
        sum = np.log(probabilities).sum()
        #return np.log(probabilities).sum()
        if sum > 0:
            print("log loss > 1. {}".format(sum))
        return sum

    def step(self):
        """
        Update the weights and bias
        """
        P = self.probability_array()
        E = self.Y - P  # prediction error for each class (column). (0 to 1)

        T = np.reshape(E.sum(axis=0), newshape=(1, self.C))
        W0_update = self.nu*T
        W0_update = np.reshape(W0_update, newshape=(1, self.C))
        self.W0 += W0_update
        assert self.W0.shape == (1, self.C)

        self.W += self.nu*(-self.lam*self.W + self.X.T.dot(E))
        assert self.W.shape == (self.d ,self.C), \
            "shape of W is {}".format(self.W.shape)

    def shrink_nu(self, s):
        self.nu = self.nu_init/self.N/s**0.5

    def run(self):

        # Step until converged
        for s in range(1, self.max_iter+1):
            self.shrink_nu(s)
            old_log_loss_normalized = -self.log_loss()/self.N

            self.step()
            sys.stdout.write(".")

            new_loss = self.loss_01()
            new_log_loss_normalized = -self.log_loss()/self.N
            one_val = pd.DataFrame({
                "iteration": [s],
                "nu": [self.nu],
                #"probability 1": [self.probability_array()],
                "probability array":[self.probability_array()],
                "weights": [self.W],
                "bias": [self.W0],
                "0/1 loss": [new_loss],
                "(0/1 loss)/N": [new_loss/self.N],
                "-(log loss)": [-self.log_loss()],
                "log loss": [self.log_loss()]
                })
            self.results = pd.concat([self.results, one_val])

            assert not self.has_increased_significantly(
                old_log_loss_normalized, new_log_loss_normalized),\
                "Normalized loss: {} --> {}".format(
                    old_log_loss_normalized, new_log_loss_normalized)
            if abs(old_log_loss_normalized - new_log_loss_normalized) < \
                    self.delta:
                print("Loss optimized.  Old/N: {}, new/N:{}".format(
                    old_log_loss_normalized, new_log_loss_normalized))
                break

        self.results.reset_index(drop=True, inplace=True)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)



class LogisticRegressionBinary(ClassificationBaseBinary):
    """
    Train *one* model.
    """

    def __init__(self, X, y, n0, lam, w=None, w0=None,
                 max_iter=10**6, delta=1e-6):
        # call the base class's methods first
        super(LogisticRegressionBinary, self).__init__(X, y, w, w0)
        self.nu = n0
        self.nu_init = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta = delta
        self.results = pd.DataFrame()

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

        self.w0 += self.nu*E.sum()
        assert self.w0.shape == ()
        assert isinstance(self.w0, np.float64)

        self.w += self.nu*(-self.lam*self.w + self.X.T.dot(E))
        assert self.w.shape == (self.d ,), \
            "shape of w is {}".format(self.w.shape)

    def shrink_nu(self, s):
        self.nu = self.nu_init/self.N/s**0.5


    def run(self):
        results = pd.DataFrame()

        # Step until converged
        for s in range(1, self.max_iter+1):
            self.shrink_nu(s)
            old_log_loss_normalized = -self.log_loss()/self.N

            self.step()
            sys.stdout.write(".")

            new_loss_01 = self.loss_01()
            new_log_loss_normalized = -self.log_loss()/self.N
            one_val = pd.DataFrame({
                "iteration": [s],
                "nu": [self.nu],
                #"probability 1": [self.probability_array()],
                "probability array":[self.probability_array()],
                "weights": [self.w],
                "bias": [self.w0],
                "0/1 loss": [new_loss_01],
                "(0/1 loss)/N": [new_loss_01/self.N],
                "-(log loss)": [-self.log_loss()],
                "log loss": [self.log_loss()]
            })
            self.results = pd.concat([self.results, one_val])

            assert not self.has_increased_significantly(
                old_log_loss_normalized, new_log_loss_normalized),\
                "Normalized loss: {} --> {}".format(
                    old_log_loss_normalized, new_log_loss_normalized)
            if abs(old_log_loss_normalized - new_log_loss_normalized) < self.delta:
                print("Loss optimized.  Old/N: {}, new/N:{}".format(
                    old_log_loss_normalized, new_log_loss_normalized))
                break

        self.results.reset_index(drop=True, inplace=True)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)