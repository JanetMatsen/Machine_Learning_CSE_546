import numpy as np
import sys
import pandas as pd

import matplotlib.pyplot as plt

from classification_base import ClassificationBase
from classification_base import ModelFitExcpetion


class LogisticRegression(ClassificationBase):
    """
    Multi-class classifications, with stochastic gradient descent.
    No bias
    """
    def __init__(self, X, y, eta0, lam, W=None,
                 max_iter=10**6, # of times passing through N pts
                 batch_size = 100,
                 progress_monitoring_freq=15000,
                 delta_percent=1e-3, verbose=False,
                 test_X=None, test_y=None): #
        # call the base class's methods first
        super(LogisticRegression, self).__init__(X=X, y=y, W=W)
        self.eta0 = eta0
        self.eta = eta0
        self.lam = lam
        self.lam_norm = lam/(np.linalg.norm(X)/self.N) # np norm defaults to L2
        self.max_iter = max_iter
        self.delta_percent = delta_percent
        self.iteration = 0
        self.verbose = verbose
        self.test_X = test_X
        self.test_y = test_y
        self.batch_size = batch_size
        assert progress_monitoring_freq%batch_size == 0, \
            "need to monitor at frequencies that are multiples of the " \
            "mini-batch size."
        print("Remember not to check the log los too often.  Expensive!")
        self.progress_monitoring_freq = progress_monitoring_freq
        self.num_passes_through_N_pts = 0
        self.points_sampled = 0

    def apply_weights(self, X):
        """
        calc XW.  No bias.
        This quantity is labeled q in my planning.
        :return: vector of Weights applied to X.
        """
        return X.dot(self.get_weights())

    def probability_array(self, X):
        """
        Calculate the array of probabilities.
        :return: An Nx1 array.
        """
        n = X.shape[0]
        R = np.exp(self.apply_weights(X))
        if R.shape[0] != n:
            print("R has shape {}".format(R.shape))
        assert R.shape[0] == n, "X dot W has dim {}".format(R.shape)
        # When you sum across a row, axis=1
        R_sum_ax1 = np.sum(R, axis=1)
        R_sum_ax1 = np.reshape(R_sum_ax1, newshape=(n, 1))
        assert R_sum_ax1.shape == (n, 1)

        # Divide each element in R by
        return R/R_sum_ax1

    def predict(self):
        """
        Produce an array of class predictions
        """
        probabilities = self.probability_array(self.X)
        # THIS ASSUMES the classifiers are in order: 0th column of the
        # probabilities corresponds to label = 0, ..., 9th col is for 9.
        classes = np.argmax(probabilities, axis=1)
        return classes

    def log_loss(self, X, Y):
        probabilities = self.probability_array(X).copy()
        # get just the probability for the correct label.
        probabilities = np.multiply(Y, probabilities)
        # collapse it into an Nx1 array:
        probabilities = np.amax(probabilities, axis=1)

        return np.log(probabilities).sum()

    def grad_desc(self):
        # break data into chunks:
        if self.batch_size == self.N:
            self.step(self.X, self.Y)

    def step(self, X, Y):
        """
        Update the weights and bias
        """
        n, d = X.shape
        assert n == Y.shape[0]
        P = self.probability_array(X)
        E = Y - P  # prediction error for each class (column). (0 to 1)

        # TODO: scale eta by batch size.
        self.W += (self.eta/self.N)*(-self.lam_norm*self.W + X.T.dot(E)/n)
        assert self.W.shape == (self.d ,self.C), \
            "shape of W is {}".format(self.W.shape)
        self.iteration += 1

    def shrink_eta(self, s, s_exp=0.5):

        self.eta = self.eta0/(s**s_exp)

    def results_row(self):
        """
        Return a dictionary that can be put into a Pandas DataFrame.
        """
        results_row = super(LogisticRegression, self).results_row()

        # append on logistic regression-specific results
        neg_log_loss = -self.log_loss(self.X, self.Y)
        more_details = {
            "lambda":[self.lam],
            "lambda normalized":[self.lam_norm],
            "eta0":[self.eta0],
            "eta": [self.eta],  # learning rate
            "log loss": [self.log_loss(self.X, self.Y)],
            "-(log loss), training": [neg_log_loss],
            "-(log loss)/N, training": [neg_log_loss/self.N],
            "iteration": [self.iteration],
            "batch size": [self.batch_size],
            "# of passes through N pts": [self.num_passes_through_N_pts]
            }
        results_row.update(more_details)
        return results_row

    def record_status(self):
        results_row = self.results_row()
        results_row['minibatches tested'] = [self.num_passes_through_N_pts]
        return results_row

    def assess_model_on_test_data(self):
        test_results = pd.DataFrame(
            self.apply_model(X=self.test_X, y=self.test_y,
                             data_name = 'testing'))
        t_columns = [c for c in test_results.columns
                     if 'test' in c or 'lambda' == c]
        return pd.DataFrame(test_results[t_columns])

    def run(self):

        num_diverged_steps = 0
        fast_convergence_steps = 0

        # Step until converged
        for s in range(1, self.max_iter+1):
            if self.verbose:
                print('loop through all the data. {}th time'.format(s))
            # Shuffle each time we loop through the entire data set.
            X, Y = self.shuffle(self.X, self.Y)

            num_pts = 0  # initial # of points seen in this pass through N pts
            # record status of log_loss before loop.

            # Don't compute loss every time; expensive!
            # TODO: move this into the loop below and get log_loss from the
            # Pandas result so I don't compute it extra times.  (Expensive!)
            old_neg_log_loss_norm = -self.log_loss(self.X, self.Y)/self.N

            # loop over ~all of the data points in little batches.
            while num_pts < self.N:
                idx_start = 0
                idx_stop = self.batch_size
                # TODO: what happens if you split training data and you ask fo'
                # more data then there is?   
                X_sample = X[idx_start:idx_stop, ]
                Y_sample = Y[idx_start:idx_stop, ]
                self.step(X_sample, Y_sample)
                num_pts += self.batch_size
                self.points_sampled += self.batch_size

                # Take the pulse once and a while, but not too much.
                if self.points_sampled%self.progress_monitoring_freq == 0:
                    # TODO: move all log-loss checking down here. Break out
                    # another function like .assess_progress()?
                    training_results = self.record_status()
                    # print(self.log_loss(self.X, self.Y))
                    training_results = pd.DataFrame(training_results)
                     # also find the log loss & 0/1 loss using test data.
                    test_results = self.assess_model_on_test_data()
                    row_results = pd.merge(training_results, test_results)
                    self.results = pd.concat([self.results, row_results])

            s+=1
            self.num_passes_through_N_pts +=1
            sys.stdout.write(".") # one dot per pass through ~ N pts

            # print every 5th pass through all N-ish data points
            new_neg_log_loss_norm = self.results_row()['-(log loss)/N, training'][0]
            if self.verbose:
                if s%10%self.batch_size == 0: print(new_neg_log_loss_norm)

            # shrink eta if we aren't moving quickly towards the optimum.
            self.shrink_eta(s - fast_convergence_steps + 1)

            neg_log_loss_percent_change = \
                (new_neg_log_loss_norm - old_neg_log_loss_norm)/ \
                old_neg_log_loss_norm*100

            #results_row['log loss % change'] = neg_log_loss_percent_change

            if neg_log_loss_percent_change > 0: num_diverged_steps += 1
            elif neg_log_loss_percent_change < -2 and num_diverged_steps == 0:
                fast_convergence_steps += 1
            else:
                num_diverged_steps = 0
            if num_diverged_steps == 10:
                raise ModelFitExcpetion("log loss grew 10 times in a row!")

            assert not self.has_increased_significantly(
                old_neg_log_loss_norm, new_neg_log_loss_norm),\
                "Normalized loss: {} --> {}".format(
                    old_neg_log_loss_norm, new_neg_log_loss_norm)
            if abs(neg_log_loss_percent_change) < self.delta_percent:
                print("Loss optimized.  Old/N: {}, new/N:{}. Eta: {}".format(
                    old_neg_log_loss_norm, new_neg_log_loss_norm, self.eta))
                # TODO: sample status a final time?  Check if it was just sampled?
                break

            if s == self.max_iter:
                # TODO: sample status a final time?  Check if it was just sampled?
                print('max iterations ({}) reached.'.format(self.max_iter))

        print('final normalized training -(log loss): {}'.format(
            new_neg_log_loss_norm))
        self.results.reset_index(drop=True, inplace=True)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)

    def plot_test_and_train_log_loss_during_fitting(
            self, filename=None, colors=['#756bb1', '#2ca25f']):

        train_y = "-(log loss)/N, training"
        test_y = "-(log loss)/N, testing"

        fig = super(LogisticRegression, self).plot_ys(
            x='iteration', y1=train_y, y2=test_y, ylabel="normalized log loss",
            logx=False, colors=colors)
        if filename is not None:
            fig.savefig(filename + '.pdf')
        return fig

    def plot_test_and_train_01_loss_during_fitting(
            self, filename=None, colors=['#756bb1', '#2ca25f']):
        train_y = "training (0/1 loss)/N"
        test_y = "testing (0/1 loss)/N"

        fig = super(LogisticRegression, self).plot_ys(
            x='iteration', y1=train_y, y2=test_y, ylabel="normalized 0/1 loss",
            logx=False, colors=colors)
        if filename is not None:
            fig.savefig(filename + '.pdf')


