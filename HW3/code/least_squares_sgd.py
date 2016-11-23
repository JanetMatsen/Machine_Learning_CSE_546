from functools import partial
import datetime
import math
import numpy as np
import sys
import pandas as pd
import re

import matplotlib.pyplot as plt

from classification_base import ClassificationBase
from classification_base import ModelFitException
from kernel import RBFKernel, Fourier


class LeastSquaresSGD(ClassificationBase):
    """
    Multi-class classifications, with stochastic gradient descent.
    No bias
    """
    def __init__(self, X, y, eta0=None, W=None,
                 kernel=Fourier,
                 kernel_kwargs=None,
                 eta0_search_start=10,  # gets normalized by N
                 eta0_max_pts=None,
                 max_epochs=50,  # of times passing through N pts
                 batch_size=10,
                 delta_percent=0.01, verbose=False,
                 check_W_bar_fit_during_fitting=False,
                 test_X=None, test_y=None,
                 assess_test_data_during_fitting=False):

        # check data
        assert X.shape[0] == y.shape[0]
        # Sometimes we check the loss of test_X and test_y during fitting
        if test_X is None and test_y is None:
            print("No test data was provided.")
        self.test_X = test_X  # ok if None
        self.test_y = test_y  # ok if None
        self.assess_test_data_during_fitting = assess_test_data_during_fitting
        if assess_test_data_during_fitting:
            print('Only assess test data during final model fitting, not '
                  'during hyperparameter exploration.  (Slows fitting down!)')
            assert test_X is not None and test_y is not None, \
                "Specify `assess_test_data_during_fitting=False` if not supplying test data"
            assert test_X.shape[0] == test_y.shape[0]

        # call the base class's methods first
        super(LeastSquaresSGD, self).__init__(X=X, y=y, W=W)

        # set up the kernel
        if kernel_kwargs is not None:
            self.kernel = kernel(X, **kernel_kwargs)
        else:
            self.kernel = kernel(X)
        # write over base class's W
        self.W = np.zeros(shape=(self.kernel.d, self.C))

        # set up attributes used for fitting
        self.epochs = 0
        self.max_epochs = max_epochs
        self.delta_percent = delta_percent
        self.steps = 0
        self.fast_steps = 0
        self.verbose = verbose

        self.batch_size = batch_size
        self.points_sampled = 0
        self.converged = False # Set True if converges.

        # keep track of last n sets of weights to compute \bar(w)
        self.W_sums_for_epoch = self.W.copy() # Will average all the weight vectors for the epoch.
        self.W_vectors_in_sum = 1 # reset to 0 at the beginning of each epoch
        # Will be average weight vector for all steps in the epoch.
        self.W_bar = None # W_sums_for_epoch/W_vectors_in_sum
        if check_W_bar_fit_during_fitting:
            print("Checking bar{W} as we go.  Adds expense!")
        self.check_W_bar_fit_during_fitting = check_W_bar_fit_during_fitting

        self.eta0_search_start = eta0_search_start
        if eta0 is None:
            self.eta0_search_calls = 0
            if eta0_max_pts is None:
                eta0_max_pts=3000
            self.find_good_learning_rate(eta0_max_pts)
        else:
            self.eta0 = eta0
        self.eta = self.eta0
        # \bar{Y} is expensive to calc, so share it across functions
        self.Yhat = None

    def copy(self, reset=True):
        model = super(LeastSquaresSGD, self).copy(reset=reset)
        return model

    def zero_weights(self):
        self.W = np.zeros(shape=(self.kernel.d, self.C))

    def reset_model(self):
        """
        Reset everything *except* the weights, and model-fitting to date
        """
        self.epochs = 1
        self.points_sampled = 0
        self.converged = False
        self.diverged = False
        self.W_bar_variance_df = pd.DataFrame()
        self.steps = 0
        self.points = 0
        # for eta0 search resets, we don't start with an eta0.
        if 'eta0' in self.__dict__.keys():
            self.eta0_search_start = self.eta0
            self.eta = self.eta0
        self.eta0_search_calls = 0
        self.zero_weights()
        self.results = None

    def find_good_learning_rate(self,max_pts=5000):
        """
        Follow Sham's advice of cranking up learning rate until the model
        diverges, then cutting it back down 50%.

        The final learning rate that is found via Sham's advice is dependent
        on how much you crank up the learning rate each time, and
        how you define divergence.

        My tool defines divergence by having a string of sequential
        diverging update steps.
        """
        max_divergence_streak_length=2
        max_expochs=3

        # First scale the eta0 value by the number of points in the training set.
        eta0 = self.eta0_search_start/self.N
        print("eta0 search begins with eta0 = {}/{} = {}"
              "".format(self.eta0_search_start, self.N, eta0))
        starting_eta0 = eta0
        change_factor = 5
        eta0 = eta0/change_factor  # so we don't skip first value

        num_pts = min(max_pts, self.N)
        print("Determining eta0 using {} points".format(num_pts))
        random_indices = np.random.choice(self.N, num_pts, replace=False)
        X = self.X[random_indices]
        y = self.y[random_indices]
        model = self.copy()
        model.assess_test_data_during_fitting = False
        model.check_W_bar_fit_during_fitting = False
        model.replace_X_and_y(X, y)

        # passed will become False once the learning rate is cranked up
        # enough to cause a model fit exception.
        passed = True
        rates_tried = 0
        max_rates = 30
        while passed is True and self.eta0_search_calls  < max_rates:
            try:
                self.eta0_search_calls  += 1
                rates_tried += 1
                # increase eta0 until we see divergence
                eta0 = eta0*change_factor
                print('testing eta0 = {}.  (Try # {})'.format(eta0, self.eta0_search_calls))
                # Test high learning rates until the model diverges.
                model.reset_model()
                # reset weights (can't assert!)
                model.zero_weights()
                model.eta0 = eta0
                model.eta = eta0
                model.max_epochs = max_expochs # make sure it fails pretty fast.
                model.run(
                    max_divergence_streak_length=max_divergence_streak_length)
                if model.epochs < model.max_epochs and \
                                model.epochs == model.max_epochs:
                    passed = False
                # If that passed without exception, passed = True
            except ModelFitException as e:
                print("Model training raised an exception.")
                passed = False
        assert rates_tried >= 1, "\n eta0 didn't change; start lower"
        print("Exploration for good eta0 started at {}; stopped passing when "
              "eta0  grew to {}".format(starting_eta0, eta0))

        if self.eta0_search_calls == max_rates:
            print("search for eat0 tried {} values and failed to converge."
                  "".format(max_rates))
            raise ModelFitException("eta0 search failed")

        if rates_tried == 1:
            print("--- eta0 didn't change; start 125x lower --- \n")
            self.eta0_search_start = self.eta0_search_start/5**3
            self.find_good_learning_rate()
        else:
            # return an eta almost as high as the biggest one one that
            # didn't cause divergence
            self.eta0 = eta0/change_factor
            self.eta = self.eta0
            print("===== eta0 search landed on {}, using {} points ===="
                  "".format(self.eta0, num_pts))

    def apply_weights(self, X):
        """
        Calculate the prediction matrix: Y_bar = XW.  No bias.
        """
        # Todo: change to self.W   This was leftover from prev code
        return X.dot(self.get_weights())

    def step(self, X, Y):
        """
        Update the weights and bias, using X which *has* been transformed
        by the kernel
        """
        if np.isnan(X).any():
            print("warning: X has some nan in it")
        if np.isnan(Y).any():
            print("warning: Y has some nan in it")

        n, d = X.shape  # n and d of the sub-sample of X
        assert n == Y.shape[0]
        assert X.shape == (n, self.kernel.d)

        gradient = -(1./n)*X.T.dot(Y - X.dot(self.W))
        if np.isnan(gradient).any():
            print("gradient might have gotten too large")
            print(gradient)
            raise ModelFitException("Model gradient must have gotten too large")
        assert gradient.shape == (self.kernel.d, self.C)

        assert self.eta is not None
        self.W += -(self.eta/n)*gradient
        assert self.W.shape == (self.kernel.d ,self.C), \
            "shape of W is {}".format(self.W.shape)
        self.steps += 1

    def calc_Yhat(self, chunk_size=10, calc_for_W_bar = True):
        """
        Produce an (NxC) array of classes predictions on X, which has *not*
        been transformed by the kernel.
        """
        assert self.W is not None, "Can't calc hat{Y} without weights, W."
        X = self.X

        if X.shape[0] < chunk_size:
            chunk_size = X.shape[0]
        N = X.shape[0]
        n = 0
        if calc_for_W_bar:
            assert len(self.W_sums_for_epoch) > 0, "need weights for bar{W}"
            Wbar = self.calc_W_bar()
            assert Wbar is not None
            assert not np.isnan(Wbar).any()

        def build_up_Yhat(X_chunk, Yhat, weights):
            assert weights is not None
            Yhat_chunk = X_chunk.dot(weights)
            assert not np.isnan(Yhat_chunk).any()
            assert Yhat_chunk.shape == (X_chunk.shape[0], self.C)
            if Yhat is None:
                Yhat = Yhat_chunk
            else:
                Yhat = np.vstack([Yhat, Yhat_chunk])
            return Yhat

        Yhat = None
        Yhat_Wbar = None

        # for each chunk of X, transform to kernel and find Yhat.
        num_iter = 0
        while n < N:
            # Find kernel-version of a chunk of X
            X_chunk = X[n: n+chunk_size, ]
            kernel_chunk = self.kernel.transform(X_chunk)
            assert kernel_chunk.shape == (X_chunk.shape[0], self.kernel.d)

            assert not np.isnan(self.W).any()

            Yhat = build_up_Yhat(X_chunk=kernel_chunk, Yhat=Yhat, weights=self.W)
            if calc_for_W_bar:
                Yhat_Wbar = build_up_Yhat(X_chunk=kernel_chunk, Yhat=Yhat_Wbar,
                                          weights=Wbar)

            n += X_chunk.shape[0]

            num_iter += 1
            if num_iter%100 == 0:
                sys.stdout.write(",")
        print(" (done calculating hat{Y})") # line break after , printing

        assert Yhat.shape == (N, self.C)
        if calc_for_W_bar:
            assert Yhat_Wbar.shape == (N, self.C)

        # Yhat_Wbar mis None if you didn't ask for it
        return Yhat, Yhat_Wbar

    def predict(self):
        """
        Predict for the entire X matrix.  We only calc 0/1 loss on whole set.
        :return:
        """
        assert self.Yhat is not None, \
            "Compute Yhat before calling predict, but don't compute too often!"

        classes = np.argmax(self.Yhat, axis=1)
        return classes

    def square_loss(self):
        assert self.Yhat is not None, \
            "Compute Yhat before calling predict, but don't compute too often!"
        Yhat = self.Yhat

        errors = self.Y - Yhat
        avg_err = np.sum(np.absolute(errors))/self.N/self.C

        if self.verbose:
            print("average error: {}.  (step = {})".format(avg_err, self.steps))
        if avg_err > 5:
            print("The sum of errors is concerningly big: {}".format(avg_err))

        # element-wise squaring:
        errors_squared = np.multiply(errors, errors)
        squares_sum = errors_squared.sum()
        assert not math.isnan(squares_sum)
        return squares_sum

    def shrink_eta(self, s_exp=0.5):
        """
        Scale eta by the number of steps, in a way that is independent of
        the batch size.
        :param s: steps so far
        :param s_exp: exponential rate
        :return:
        """
        epochs = (self.steps - self.fast_steps)/(self.N) + 1
        self.eta = self.eta0/(epochs**s_exp)

    def results_row(self):
        """
        Return a dictionary that can be put into a Pandas DataFrame.

        Expensive!  Computes stuff for the (Nxd) X matrix.
        """
        # append on logistic regression-specific results
        if len(self.W_sums_for_epoch) >= 1 and \
                self.check_W_bar_fit_during_fitting:
            calc_for_W_bar = True
        else:
            # can't calculate anything before the weights have been set once.
            calc_for_W_bar = False

        self.Yhat, self.Yhat_Wbar = \
            self.calc_Yhat(calc_for_W_bar = calc_for_W_bar)

        # call parent class for universal metrics
        row = super(LeastSquaresSGD, self).results_row()

        square_loss = self.square_loss()
        more_details = {
            "eta0":[self.eta0],
            "eta": [self.eta],  # learning rate
            "(square loss), training": [square_loss],
            "(square loss)/N, training": [square_loss/self.N],
            "step": [self.steps],
            "epoch": [self.epochs],
            "epoch (fractional)": [(self.steps - self.fast_steps)/(self.N) + 1],
            "batch size": [self.batch_size],
            "points": [self.points_sampled]
            }
        row.update(more_details)
        kernel_info = self.kernel.info()
        row.update(kernel_info)
        self.Yhat = None  # wipe it so it can't be used incorrectly later
        self.Yhat_Wbar = None  # wipe it so it can't be used incorrectly later
        return row

    def calc_W_bar(self):
        """
        bar{W} is the average of the weight vectors for the epoch
        """
        return self.W_sums_for_epoch/self.W_vectors_in_sum

    def add_W_to_epoch_W_sum(self, weight_array):
        """
        \bar{W} is the average weights over the last n fittings

        It is built from a tuple of previous weights, stored in
        self.last_n_weights.
        """
        assert weight_array.shape == self.W_sums_for_epoch.shape
        self.W_sums_for_epoch = \
            np.add(self.W_sums_for_epoch, weight_array)

    def run(self, max_divergence_streak_length=7, rerun=False):

        # To support running model longer, need to retrieve
        if rerun:
            print("Before re-run, epochs = {}, steps = {}".format(
                self.epochs, self.steps))
            if self.results is None:
                self.results = pd.concat([self.results, self.observe_fit()], axis=0)
        else:
            self.results = pd.concat([self.results, self.observe_fit()], axis=0)
            self.W_sums_for_epoch = \
                np.zeros(shape=(self.kernel.d, self.C))

        # initialize the statistic for tracking variance
        # Should be zero if weights are initially zero.
        old_W_bar = self.calc_W_bar()
        self.W_vectors_in_sum = 0

        old_square_loss_norm = \
                self.results.tail(1).reset_index()['(square loss)/N, training'][0]

        # Step until converged
        while self.epochs < self.max_epochs:
            if self.converged:
                print('model already converged.')

            epoch_start_time = datetime.datetime.now()

            # Reset the weights for the epoch at the epoch's start.
            self.W_sums_for_epoch = np.zeros(shape=(self.kernel.d, self.C))
            self.W_vectors_in_sum = 0

            if self.verbose:
                print('Begin epoch {}'.format(self.epochs))
            # Shuffle each time we loop through the entire data set.
            X, Y = self.shuffle(self.X.copy(), self.Y.copy())

            # loop over ~all of the data points in little batches.
            num_pts = 0
            iter = 0
            epoch_iters = 0
            while num_pts < self.N :

                iter += 1

                idx_start = num_pts
                idx_stop = num_pts + self.batch_size
                X_sample = X[idx_start:idx_stop, ] # works even if you ask for too many rows.
                # apply the kernel transformation
                X_sample = self.kernel.transform(X_sample)
                Y_sample = Y[idx_start:idx_stop, ]

                # update W
                self.step(X_sample, Y_sample)

                # Add weight to total, which will be divided by N at the end.
                self.add_W_to_epoch_W_sum(self.W)
                self.W_vectors_in_sum += 1

                # get ready for next loop
                num_pts += X_sample.shape[0]  # loop-scoped count
                self.points_sampled += X_sample.shape[0]  # every point ever

                self.steps += 1
                epoch_iters += 1
                if epoch_iters%100 == 0:
                    # one dot per 10 steps
                    # Doesn't print anything for small N, but those don't take
                    # long anyway.
                    sys.stdout.write(".")

            print(" (epoch complete)") # line break after . printing

            # --- EPOCH IS OVER ---
            epoch_stop_time = datetime.datetime.now()
            if self.verbose:
                print("Epoch iteration time: {}.".format(
                    self.time_delta(epoch_start_time, epoch_stop_time)))

            self.epochs += 1
            W_bar = self.calc_W_bar()
            self.W_bar = W_bar

            # EPOCH IS OVER.  RECORD FIT STATS
            start_time = datetime.datetime.now()
            epoch_results = self.observe_fit()
            epoch_results['bar{W} update variance'] = \
                self.W_bar_update_variance(old_W_bar, W_bar)
            self.results = pd.concat([self.results, epoch_results], axis=0)
            if self.verbose:
                stop_time = datetime.datetime.now()
                print("fit observation done: {}.".format(
                    self.time_delta(start_time, stop_time)))

            # TEST FOR CONVERGENCE
            square_loss_norm = \
                self.results.tail(1).reset_index()['(square loss)/N, training'][0]
            assert square_loss_norm is not None, \
                "square loss shouldn't be None"
            assert not math.isnan(square_loss_norm), "square loss can't be nan"

            if square_loss_norm/self.N > 1e3:
                s = "square loss/N/N grew to {}".format(
                    square_loss_norm/self.N)
                raise ModelFitException(s)

            if self.epochs > 1:
                square_loss_percent_improvement = self.percent_change(
                    new = square_loss_norm, old = old_square_loss_norm)

                if self.verbose:
                    print(square_loss_norm)
                if self.percent_metrics_converged(square_loss_percent_improvement):
                    print("Loss optimized.  Old/N: {}, new/N:{}. Eta: {}"
                          "".format(old_square_loss_norm, square_loss_norm,
                                    self.eta))
                    self.converged = True
                    break
                    return
                elif self.test_divergence(n=max_divergence_streak_length):
                    raise ModelFitException(
                        "\nSquare loss grew {} measurements in a row!"
                        "".format(max_divergence_streak_length))

            old_W_bar = W_bar
            old_square_loss_norm = square_loss_norm

            if self.epochs == self.max_epochs:
                print('\n!!! Max epochs ({}) reached. !!!'.format(self.max_epochs))

            # shrink learning rate
            #self.shrink_eta(self.epochs - fast_convergence_epochs + 1)
            self.shrink_eta()

        print('final normalized training (square loss): {}'.format(square_loss_norm))
        self.results.reset_index(drop=True, inplace=True)

    def run_longer(self, epochs,
                   delta_percent=None, max_divergence_streak_length=None,
                   fast_steps=None):
        print("Run model (currently with with {} steps) longer."
              "".format(self.steps))
        if self.converged:
            print("Don't run a previously converged model longer without"
                  "changing the convergence criteria.")
            self.converged = False

        # increase the learning rate by specifying a number of fast steps.
        # This is subtracted from self.steps when scaling eta by the number of
        # steps
        if fast_steps is not None:
            if self.fast_steps > 0 and self.fast_steps is not None:
                print("Warning: number of fast steps was already set to {}.  "
                      "Changing it to {}".format(self.fast_steps, fast_steps))
            self.fast_steps = fast_steps

        self.max_epochs = self.max_epochs + epochs

        if delta_percent is not None:
            self.delta_percent = delta_percent

        if max_divergence_streak_length is not None:
            self.run(max_divergence_streak_length=max_divergence_streak_length)
        else:
            self.run(rerun=True)

    def W_bar_update_variance(self, old_W_bar, new_W_bar):
        difference = np.subtract(new_W_bar, old_W_bar)
        return np.var(difference)

    def observe_fit(self):
        row_results = pd.DataFrame(self.results_row())
        assert row_results.shape[0] == 1, "row_results should have 1 row"

        if self.check_W_bar_fit_during_fitting:
            W_bar_results = self.observe_fit_using_W_bar()
            assert W_bar_results.shape[0] == 1, \
                "\bar{W} results should have 1 row"
            merged_results = pd.merge(row_results, W_bar_results)
            assert merged_results.shape[0] == 1

        # also find the square loss & 0/1 loss using test data.
        if self.assess_test_data_during_fitting:
            assert (self.test_X is not None) and (self.test_y is not None), \
                "Asked for test results but no test data was provided."
            test_results = self.observe_fit_on_test_data()
            merged_results = pd.merge(merged_results, test_results)
            assert merged_results.shape[0] == 1
            row_results = merged_results # merge worked

        return row_results
        #self.results = pd.concat([self.results, row_results], axis=0)

    def observe_fit_using_W_bar(self):
        """
        Note: this has nothing to do with model fitting.
        It is only for reporting and gaining intuition.
        """

        # Make a copy of the model and replace the weights with the W_bar
        # weights
        assert len(self.W_sums_for_epoch) > 0, "Need weights to do bar{W} stuff"
        model = self.copy(reset=False)
        model.W = self.calc_W_bar()

        # Get the results, the usual way:
        all_W_bar_results = model.results_row()
        results = {re.sub("training", "training (bar{W})", k): v
                   for k, v in all_W_bar_results.items()}

        # don't need step if merging on.
        columns = [c for c in results.keys() if 'bar{W}' in c]
        columns.append('step')
        results_df = pd.DataFrame(results)[columns]
        assert results_df.shape[0] == 1, "Results for bar{W} should be length 1"
        return results_df

    def observe_fit_on_test_data(self):
        """
        Note: this has nothing to do with model fitting.
        It is only for reporting and gaining intuition.
        """
        new_model = self.copy(reset=False)
        new_model.replace_X_and_y(self.test_X, self.test_y)
        # Get the results, the usual way:
        all_W_bar_results = new_model.results_row()
        results = {re.sub("training", "testing", k): v
                   for k, v in all_W_bar_results.items()}
        if self.results is not None:
            print_cols = [c for c in results.keys() if "0/1 loss" in c]
            print(pd.DataFrame(results)[print_cols].reset_index(drop=True).T)

        W_bar_results = new_model.observe_fit_using_W_bar()
        # already pluggs in the extra bar{W} string
        W_bar_results = {re.sub("training", "testing", k): v
                   for k, v in W_bar_results.items()}
        results.update(W_bar_results)

        # don't need step if merging on.
        columns = [c for c in results.keys() if 'test' in c]
        columns.append('step')
        results_df = pd.DataFrame(results)[columns]
        assert results_df.shape[0] == 1, "Results for bar{W} should be length 1"
        return results_df

    def test_divergence(self, n):
        """
        Check stats from last n pulses and return True if they are ascending.
        """
        last_square_losses = \
            self.results.tail(n)['(square loss)/N, training']
        if len(last_square_losses) < n:
            return False
        # Check for monotonic increase:
        # http://stackoverflow.com/questions/4983258/python-how-to-check-list-monotonicity
        return all(x < y for x, y in
                   zip(last_square_losses, last_square_losses[1:]))

    def percent_change(self, new, old):
        # todo: move to parent class.
        return (new - old)/old*100.

    def percent_metrics_converged(self, *args):
        # check whether all of the metrics are less than delta_percent
        for metric in args:
            if abs(metric) > self.delta_percent:
                return False
        else:
            return True

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)

    def plot_01_loss(self, filename=None, logx=False, head_n=None, tail_n=None):
        super(LeastSquaresSGD, self).plot_01_loss(
            y="training (0/1 loss)/N", filename=filename, logx=logx,
            head_n=head_n, tail_n=tail_n)

    def plot_square_loss(self, filename=None, logx=False, logy=False,
                         head_n=None, tail_n=None):
        self.plot_ys(x='step', y1="(square loss)/N, training",
                     ylabel="(square loss)/N", logx=logx, logy=logy,
                     filename=filename, head_n=head_n, tail_n=tail_n)

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

    def plot_W_bar_update_variance(self):
        x = 'epoch'
        y1 = 'bar{W} update variance'
        self.plot_ys(x=x, y1=y1, y2=None,
                     ylabel= "bar{W} update variance",
                     y0_line=True, logx=False, logy=False,
                     colors=None, figsize=(4, 3))

    def plot_loss_of_both_W_arrays(self, loss='square', style='lines'):
        """
        One plot showing the both the squared loss after every epoch

        Plot the loss of both wt and the average weight vector wτ .
        You shave both the training and test losses on the same plot
        (so there should be four curves)
        """
        fig, ax = plt.subplots(1, 1, figsize=(4,3))

        if loss == 'square':
            y_vars = ['(square loss)/N, training', '(square loss)/N, training (bar{W})',
                      '(square loss)/N, testing (bar{W})', '(square loss)/N, testing']
            ylabel = "square loss, normalized"
        elif loss == '0/1':
            y_vars = ['training (0/1 loss)/N', 'training (bar{W}) (0/1 loss)/N',
                      'testing (0/1 loss)/N', 'testing (bar{W}) (0/1 loss)/N']
            ylabel = "0/1 loss, normalized"
        else:
            print("Doesn't handle plot of tpe {}.  Did you mean 'square' or '0/1'?")
        x = self.results['epoch']

        colors = ['#bcbddc', '#756bb1', '#74c476', '#006d2c']
        if style == 'lines':
            s_vals = ['--', '-', '--', '-']
        elif style == 'dots':
            s_vals = [10, 5, 10, 5]

        for y, c, s in zip(y_vars, colors, s_vals):
            if style == 'lines':
                plt.plot(x, self.results[y], linestyle=s, linewidth=3, color=c)
            elif style == 'dots':
                plt.plot(x, self.results[y], linestyle='--',
                         marker='o', markersize=s, color=c, alpha=0.7)
            plt.legend(loc = 'best')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel("epoch")

        ax.set_ylim(bottom=0)
        plt.ylabel(ylabel)
        plt.tight_layout()
        return fig

    @staticmethod
    def datetime():
        now = datetime.datetime.now()
        return now.strftime("%Y-%m-%d %H:%M:%S")

    def time_delta(self, start, stop):
        total_seconds = (stop - start).total_seconds()
        hours, remainder = divmod(total_seconds,60*60)
        minutes, seconds = divmod(remainder,60)
        seconds = int(np.round(seconds))
        if hours > 0:
            return '{}:{}:{}'.format(hours,minutes,seconds)
        else:
            return '{}:{}'.format(minutes,seconds)

