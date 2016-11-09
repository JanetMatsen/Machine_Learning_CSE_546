import numpy as np
import sys
import pandas as pd

import matplotlib.pyplot as plt

from classification_base import ClassificationBase
from classification_base import ModelFitExcpetion


class LeastSquaresSGD(ClassificationBase):
    """
    Multi-class classifications, with stochastic gradient descent.
    No bias
    """
    def __init__(self, X, y, eta0=None, W=None,
                 max_steps=10**6, # of times passing through N pts
                 batch_size = 100,
                 progress_monitoring_freq=15000,
                 delta_percent=1e-3, verbose=False,
                 test_X=None, test_y=None): #
        # call the base class's methods first
        super(LeastSquaresSGD, self).__init__(X=X, y=y, W=W)
        self.max_steps = max_steps
        self.delta_percent = delta_percent
        self.steps = 0
        self.verbose = verbose
        if test_X is None and test_y is None:
            print("No test data was provided.")
        self.test_X = test_X
        self.test_y = test_y
        self.batch_size = batch_size
        assert progress_monitoring_freq%batch_size == 0, \
            "need to monitor at frequencies that are multiples of the " \
            "mini-batch size."
        print("Remember not to check the log loss too often.  Expensive!")
        self.progress_monitoring_freq = progress_monitoring_freq
        self.num_passes_through_N_pts = 0
        self.points_sampled = 0
        self.converged = False # Set True if converges.
        if eta0 is None:
            self.eta0 = self.find_good_learning_rate()
        else:
            self.eta0 = eta0
        self.eta = self.eta0

    def find_good_learning_rate(self, starting_eta0=0.01):
        eta0 = starting_eta0
        change_factor = 10
        passed = True
        while passed is True:
            try:
                # increse eta0 until we see divergence
                eta0 = eta0*change_factor
                print('testing eta0 = {}'.format(eta0))
                # Test high learning rates until the model diverges.
                model = self.copy()
                # reset weights (can't assert!)
                model.W = np.zeros(model.W.shape)
                model.eta0 = eta0
                model.eta = eta0
                model.max_steps = 101 # make sure it fails pretty fast.
                model.run()
                if model.steps < model.max_steps and model.converged == False:
                    passed = False
                # If that passed without exception, passed = True
            except:
                passed = False
        assert eta0 != starting_eta0, "\n eta0 didn't change; start lower"
        print("Exploration for good eta0 started at {}; stopped passing when "
              "eta0  grew to {}".format(starting_eta0, eta0))
        # return an eta almost as high as the biggest one one that
        # didn't cause divergence
        # todo: he says dividing by 2 works.  I'm getting bouncy w/o.
        self.eta0 = eta0/4
        return eta0/4

    def apply_weights(self, X):
        """
        Calculate the prediction matrix: Y_hat = XW.  No bias.
        """
        return X.dot(self.get_weights())

    def derivative(self, X):
        """
        Derivative w.r.t. a subsample of training points, without the 2
        constant
        """

    def step(self, X, Y):
        """
        Update the weights and bias
        """
        n, d = X.shape  # n and d of the sub-sample of X
        assert n == Y.shape[0]
        # TODO: be positive I use W for all the points so far.
        gradient = -(1./n)*X.T.dot(Y - X.dot(self.W))
        assert gradient.shape == (self.d, self.C)

        # TODO: do I scale eta by N still?
        # TODO: subtract the gradient for grad descent (?)
        assert self.eta is not None
        self.W += -(self.eta/n)*gradient
        assert self.W.shape == (self.d ,self.C), \
            "shape of W is {}".format(self.W.shape)
        self.steps += 1

    def calc_Yhat(self, X, Y):
        """
        Produce an (NxC) array of classes predictions.
        """
        Yhat = X.dot(self.W)
        assert Yhat.shape == (X.shape[0], self.C)
        return Yhat

    def predict(self):
        """
        Predict for the entire X matrix.  We only calc 0/1 loss on whole set.
        :return:
        """
        Yhat = self.calc_Yhat(self.X, self.Y)
        classes = np.argmax(Yhat, axis=1)
        return classes

    def square_loss(self, X, Y):
        Yhat = self.calc_Yhat(X, Y)
        errors = Y - Yhat
        # element-wise squaring:
        errors_squared = np.multiply(errors, errors)
        return errors_squared.sum()

    def shrink_eta(self, s, s_exp=0.5):
        # TODO: think about shrinking eta with time. :%
        self.eta = self.eta0/(s**s_exp)

    def results_row(self):
        """
        Return a dictionary that can be put into a Pandas DataFrame.

        Expensive!  Computes stuff for the (Nxd) X matrix.
        """
        results_row = super(LeastSquaresSGD, self).results_row()

        # append on logistic regression-specific results
        square_loss = self.square_loss(self.X, self.Y)
        more_details = {
            "eta0":[self.eta0],
            "eta": [self.eta],  # learning rate
            "square loss": [self.square_loss(self.X, self.Y)],
            "(square loss), training": [square_loss],
            "(square loss)/N, training": [square_loss/self.N],
            "step": [self.steps],
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
        """
        Note: this has nothing to do with model fitting.
        It is only for reporting and gaining intuition.
        """
        test_results = pd.DataFrame(
            self.apply_model(X=self.test_X, y=self.test_y,
                             data_name = 'testing'))
        t_columns = [c for c in test_results.columns
                     if 'test' in c or 'step' == c]
        return pd.DataFrame(test_results[t_columns])

    def run(self):
        num_diverged_steps = 0
        fast_convergence_steps = 0

        # Step until converged
        for s in range(1, self.max_steps+1):
            if self.verbose:
                print('loop through all the data. {}th time'.format(s))
            # Shuffle each time we loop through the entire data set.
            X, Y = self.shuffle(self.X, self.Y)

            num_pts = 0  # initial # of points seen in this pass through N pts
            # record status of square_loss before loop.

            # Don't compute loss every time; expensive!
            # TODO: move this into the loop below and get square_loss from the
            # Pandas result so I don't compute it extra times.  (Expensive!)
            old_square_loss_norm = self.square_loss(self.X, self.Y)/self.N

            # loop over ~all of the data points in little batches.
            while num_pts < self.N:
                idx_start = 0
                idx_stop = self.batch_size
                # TODO: what happens if you split training data and you ask fo'
                # more data then there is?   
                X_sample = X[idx_start:idx_stop, ] # works even if you ask for too many rows.
                Y_sample = Y[idx_start:idx_stop, ]
                self.step(X_sample, Y_sample)
                num_pts += X.shape[0]
                self.points_sampled += num_pts

                # Take the pulse once and a while, but not too much.
                if self.points_sampled%self.progress_monitoring_freq == 0:
                    # TODO: move all log-loss checking down here. Break out
                    # another function like .assess_progress()?
                    row_results = pd.DataFrame(self.record_status())
                    # print(self.square_loss(self.X, self.Y))
                     # also find the square loss & 0/1 loss using test data.
                    if (self.test_X is not None) and (self.test_y is not None):
                        test_results = self.assess_model_on_test_data()
                        row_results = pd.merge(row_results, test_results)
                    self.results = pd.concat([self.results, row_results])

            s+=1
            self.num_passes_through_N_pts +=1
            sys.stdout.write(".") # one dot per pass through ~ N pts

            # print every 5th pass through all N-ish data points
            new_square_loss_norm = self.results_row()['(square loss)/N, training'][0]
            if self.verbose:
                if s%10%self.batch_size == 0: print(new_square_loss_norm)

            # shrink eta if we aren't moving quickly towards the optimum.
            self.shrink_eta(s - fast_convergence_steps + 1)

            square_loss_percent_change = \
                (new_square_loss_norm - old_square_loss_norm)/ \
                old_square_loss_norm*100

            #results_row['square loss % change'] = neg_log_loss_percent_change

            if square_loss_percent_change > 0: num_diverged_steps += 1
            elif square_loss_percent_change < -2 and num_diverged_steps == 0:
                fast_convergence_steps += 1
            else:
                num_diverged_steps = 0
            if num_diverged_steps == 10:
                print("\nWarning: model diverged 10 steps in a row.")
            elif num_diverged_steps == 100:
                raise ModelFitExcpetion("\nSquare loss grew 100 times in a row!")

            assert not self.has_increased_significantly(
                old_square_loss_norm, new_square_loss_norm),\
                "Normalized loss: {} --> {}".format(
                    old_square_loss_norm, new_square_loss_norm)
            if abs(square_loss_percent_change) < self.delta_percent:
                self.converged = True # flag that it converged.
                print("Loss optimized.  Old/N: {}, new/N:{}. Eta: {}".format(
                    old_square_loss_norm, new_square_loss_norm, self.eta))
                # TODO: sample status a final time?  Check if it was just sampled?
                break

            if s == self.max_steps:
                # TODO: sample status a final time?  Check if it was just sampled?
                print('\n!!! Max steps ({}) reached. !!!'.format(self.max_steps))

        print('final normalized training (square loss): {}'.format(
            new_square_loss_norm))
        self.results.reset_index(drop=True, inplace=True)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)

    def plot_01_loss(self, filename=None):
        super(LeastSquaresSGD, self).plot_01_loss(y="training (0/1 loss)/N",
                                                  filename=filename)

    def plot_square_loss(self, filename=None, last_steps=None):
        fig = self.plot_ys(x='step', y1="(square loss)/N, training",
                           ylabel="(square loss)/N")
        if filename:
            fig.savefig(filename + '.pdf')

    def plot_test_and_train_square_loss_during_fitting(
            self, filename=None, colors=['#756bb1', '#2ca25f']):

        train_y = "(square loss)/N, training"
        test_y = "(square loss)/N, testing"

        fig = super(LeastSquaresSGD, self).plot_ys(
            x='step', y1=train_y, y2=test_y,
            ylabel="normalized square loss",
            logx=False, colors=colors)
        if filename is not None:
            fig.savefig(filename + '.pdf')
        return fig

    def plot_test_and_train_01_loss_during_fitting(
            self, filename=None, colors=['#756bb1', '#2ca25f']):
        train_y = "training (0/1 loss)/N"
        test_y = "testing (0/1 loss)/N"

        fig = super(LeastSquaresSGD, self).plot_ys(
            x='step', y1=train_y, y2=test_y, ylabel="normalized 0/1 loss",
            logx=False, colors=colors)
        if filename is not None:
            fig.savefig(filename + '.pdf')


