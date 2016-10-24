import numpy as np
import sys
import pandas as pd

import matplotlib.pyplot as plt

from classification_base import ClassificationBase


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
        super(LogisticRegression, self).__init__(X=X, y=y, W=W)
        self.eta_init = n0
        self.eta = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta_percent = delta_percent

    def apply_weights(self):
        """
        calc XW.  No bias.
        This quantity is labeled q in my planning.
        :return: vector of Weights applied to X.
        """
        return self.X.dot(self.get_weights())

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
        # todo: generalize to minibatch.
        P = self.probability_array()
        E = self.Y - P  # prediction error for each class (column). (0 to 1)

        self.W += self.eta*(-self.lam*self.W + self.X.T.dot(E))
        assert self.W.shape == (self.d ,self.C), \
            "shape of W is {}".format(self.W.shape)

    def shrink_eta(self, s, s_exp=0.5):

        self.eta = self.eta_init/self.N/(s**s_exp)

    def results_row(self):
        """
        Return a dictionary that can be put into a Pandas DataFrame.
        """
        results_row = super(LogisticRegression, self).results_row()

        # append on logistic regression-specific results
        neg_log_loss = -self.log_loss()
        more_details = {
            "eta": [self.eta],  # learning rate
            "log loss": [self.log_loss()],
            "-(log loss)": [neg_log_loss],
            "-(log loss)/N": [neg_log_loss/self.N],
            }
        results_row.update(more_details)
        return results_row

    def run(self):

        num_diverged_steps = 0

        # Step until converged
        for s in range(1, self.max_iter+1):
            self.shrink_eta(s)
            old_neg_log_loss_norm = -self.log_loss()/self.N

            self.step()
            sys.stdout.write(".")

            results_row = self.results_row()
            new_neg_log_loss_norm = results_row['-(log loss)/N'][0]
            log_loss_percent_change = (new_neg_log_loss_norm - old_neg_log_loss_norm)/ \
                old_neg_log_loss_norm*100

            results_row['iteration'] = s
            results_row['log loss percent change'] = log_loss_percent_change
            one_val = pd.DataFrame(results_row)
            self.results = pd.concat([self.results, one_val])

            if log_loss_percent_change > 0:
                num_diverged_steps += 1
            else:
                num_diverged_steps = 0
            if num_diverged_steps == 10:
                assert False, "log loss grew 10 times in a row!"

            assert not self.has_increased_significantly(
                old_neg_log_loss_norm, new_neg_log_loss_norm),\
                "Normalized loss: {} --> {}".format(
                    old_neg_log_loss_norm, new_neg_log_loss_norm)
            if abs(log_loss_percent_change) < self.delta_percent:
                print("Loss optimized.  Old/N: {}, new/N:{}. Eta: {}".format(
                    old_neg_log_loss_norm, new_neg_log_loss_norm, self.eta))
                break

        self.results.reset_index(drop=True, inplace=True)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)



class LogisticRegressionBinary(LogisticRegression):
    """
    Train *one* model.
    """
    def __init__(self, X, y, n0, lam, w=None, w0=None,
                 max_iter=10**6, delta_percent=1e-3):
        # Stuff that would be in a base class:
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

        # call the base class's methods first
        self.eta = n0
        self.eta_init = n0
        self.lam = lam
        self.max_iter = max_iter
        self.delta_percent = delta_percent
        self.results = pd.DataFrame()

    def get_weights(self):
        """
        Overwrite parent's methods accessing of self.W
        """
        return self.w

    def apply_weights(self):
        """
        calc w0 + Xw
        This quantity is labeled q in my planning.
        :return: vetor of weights applied to X.
        """
        w0_array = np.ones(self.N)*self.w0
        return w0_array + self.X.dot(self.w)

    def pred_to_01_loss(self, class_calls):
        """
        Overwrite the parent class.
        Add pme point for every incorrect classification.
        """
        return self.N - np.equal(self.y, class_calls).sum()

    def probability_array(self):
        """
        Overwrite the parent class.
        Calculate the array of probabilities.
        :return: An Nx1 array.
        """
        q = self.apply_weights()
        return np.exp(q)/(1 + np.exp(q))

    def predict(self, threshold=0.5):
        """
        Overwrite the parent class.
        Produce an array of class predictions
        """
        probabilities = self.probability_array()
        classes = np.zeros(self.N)
        classes[probabilities > threshold] = 1
        return classes

    def log_loss(self):
        """
        Overwrite the parent class.
        """
        probabilities = self.probability_array().copy()
        # need to flip the probabilities for p < 0.5 with this binary case.
        # 1 - old_val is same as oldval*-1 + 1.  Do in 2 steps:
        probabilities[np.equal(0, self.y)] *= -1
        probabilities[np.equal(0, self.y)] += 1
        # when multiclass: np.amax(probabilities, 1)
        return np.log(probabilities).sum()

    def step(self):
        """
        Overwrite the parent class.
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
