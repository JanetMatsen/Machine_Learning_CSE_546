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

    def __init__(self, X, y, n0, lam, W=None, max_iter=10**6,
                 delta_percent=1e-3):
        '''
        No bias!
        '''
        # call the base class's methods first
        super(LogisticRegression, self).__init__(X, y, W)
        self.eta_init = n0
        self.eta = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta_percent = delta_percent

    def optimiize_weights(self):
        pass

    def apply_weights(self):
        """
        calc XW.  No bias.
        This quantity is labeled q in my planning.
        :return: vector of Weights applied to X.
        """
        return self.X.dot(self.W)

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

        return np.log(probabilities).sum()

    def step(self):
        """
        Update the weights and bias
        """
        P = self.probability_array()
        E = self.Y - P  # prediction error for each class (column). (0 to 1)

        self.W += self.eta*(-self.lam*self.W + self.X.T.dot(E))
        assert self.W.shape == (self.d ,self.C), \
            "shape of W is {}".format(self.W.shape)

    def shrink_eta(self, s):
        self.eta = self.eta_init/self.N/s**0.5

    def run(self):

        num_diverged_steps = 0

        # Step until converged
        for s in range(1, self.max_iter+1):
            self.shrink_eta(s)
            old_log_loss_normalized = -self.log_loss()/self.N

            self.step()
            sys.stdout.write(".")

            new_loss = self.loss_01()
            new_log_loss_normalized = -self.log_loss()/self.N
            one_val = pd.DataFrame({
                "iteration": [s],
                "eta": [self.eta],
                "probability array":[self.probability_array()],
                "weights": [self.W],
                "# nonzero weights": [self.num_nonzero_coefs()],
                "0/1 loss": [new_loss],
                "(0/1 loss)/N": [new_loss/self.N],
                "-(log loss)": [-self.log_loss()],
                "-(log loss)/N": [-self.log_loss()/self.N],
                "log loss": [self.log_loss()]
                })
            self.results = pd.concat([self.results, one_val])

            log_loss_percent_change = \
                (new_log_loss_normalized - old_log_loss_normalized)/\
                old_log_loss_normalized*100

            if log_loss_percent_change > 0:
                num_diverged_steps += 1
            else:
                num_diverged_steps = 0
            if num_diverged_steps == 10:
                assert False, "log loss grew 10 times in a row!"

            assert not self.has_increased_significantly(
                old_log_loss_normalized, new_log_loss_normalized),\
                "Normalized loss: {} --> {}".format(
                    old_log_loss_normalized, new_log_loss_normalized)
            if abs(log_loss_percent_change) < self.delta_percent:
                print("Loss optimized.  Old/N: {}, new/N:{}. Eta: {}".format(
                    old_log_loss_normalized, new_log_loss_normalized, self.eta))
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
                 max_iter=10**6, delta_percent=1e-3):
        # call the base class's methods first
        super(LogisticRegressionBinary, self).__init__(X, y, w, w0)
        self.eta = n0
        self.eta_init = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta_percent = delta_percent
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

        self.w0 += self.eta*E.sum()
        assert self.w0.shape == ()
        assert isinstance(self.w0, np.float64)

        self.w += self.eta*(-self.lam*self.w + self.X.T.dot(E))
        assert self.w.shape == (self.d ,), \
            "shape of w is {}".format(self.w.shape)

    def shrink_eta(self, s):
        self.eta = self.eta_init/self.N/s**0.5

    def run(self):
        results = pd.DataFrame()

        num_diverged_steps = 0

        # Step until converged
        for s in range(1, self.max_iter+1):
            self.shrink_eta(s)
            old_log_loss_normalized = -self.log_loss()/self.N

            self.step()
            sys.stdout.write(".")

            new_loss_01 = self.loss_01()
            new_log_loss_normalized = -self.log_loss()/self.N

            one_val = pd.DataFrame({
                "iteration": [s],
                "eta": [self.eta],
                #"probability 1": [self.probability_array()],
                "probability array":[self.probability_array()],
                "weights": [self.w],
                "# nonzero weights": [self.num_nonzero_coefs()],
                "bias": [self.w0],
                "0/1 loss": [new_loss_01],
                "(0/1 loss)/N": [new_loss_01/self.N],
                "-(log loss)": [-self.log_loss()],
                "-(log loss)/N": [-self.log_loss()/self.N],
                "log loss": [self.log_loss()]
            })
            self.results = pd.concat([self.results, one_val])

            log_loss_percent_change = \
                (new_log_loss_normalized - old_log_loss_normalized)/\
                old_log_loss_normalized*100

            if log_loss_percent_change > 0:
                num_diverged_steps += 1
            else:
                num_diverged_steps = 0
            if num_diverged_steps == 10:
                assert False, "log loss grew 10 times in a row!"

            if abs(log_loss_percent_change) < self.delta_percent:
                print("Loss optimized.  Old/N: {}, new/N:{}, eta: {}".format(
                    old_log_loss_normalized, new_log_loss_normalized, self.eta))
                break

        self.results.reset_index(drop=True, inplace=True)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)