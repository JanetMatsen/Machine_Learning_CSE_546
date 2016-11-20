from functools import partial
import datetime
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
                 eta0_search_start=0.1, # gets normalized by N
                 max_epochs=50,  # of times passing through N pts
                 batch_size=10,
                 progress_monitoring_freq=15000,
                 delta_percent=0.01, verbose=False,
                 check_W_bar_vitals=True,
                 test_X=None, test_y=None,
                 assess_test_data_during_fitting=True):

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
        self.max_epochs = max_epochs
        self.delta_percent = delta_percent
        self.steps = 0
        self.fast_steps = 0
        self.verbose = verbose

        self.batch_size = batch_size
        assert progress_monitoring_freq%batch_size == 0, \
            "need to monitor at frequencies that are multiples of the " \
            "mini-batch size."
        print("Remember not to check the loss too often.  Super $$expensive$$!")
        self.progress_monitoring_freq = progress_monitoring_freq
        self.epochs = 1
        self.points_sampled = 0
        self.converged = False # Set True if converges.

        # keep track of last n sets of weights to compute \bar(w)
        self.last_n_weights = []
        self.W_bar_variance_df = pd.DataFrame()
        self.W_bar = None
        if check_W_bar_vitals:
            print("Checking bar{W} as we go.  Adds expense!")
            self.check_W_bar_vitals = check_W_bar_vitals

        self.eta0_search_start = eta0_search_start
        if eta0 is None:
            self.eta0_search_calls = 0
            self.find_good_learning_rate()
        else:
            self.eta0 = eta0
        self.eta = self.eta0
        # \bar{Y} is expensive to calc, so share it across functions
        self.Yhat = None

    def copy(self, reset=True):
        model = super(LeastSquaresSGD, self).copy(reset=reset)
        return model

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
        self.last_n_weights = [] # erase old weights
        self.W_bar = None
        self.results = None

    def find_good_learning_rate(self, max_divergence_streak_length=3,
                                max_expochs=5, max_pts = 1000):
        # TODO: bump back up from 5, and raise max_pts
        """
        Follow Sham's advice of cranking up learning rate until the model
        diverges, then cutting it back down 50%.

        The final learning rate that is found via Sham's advice is dependent
        on how much you crank up the learning rate each time, and
        how you define divergence.

        My tool defines divergence by having a string of sequential
        diverging update steps.
        """
        # First scale the eta0 value by the number of points in the training set.
        eta0 = self.eta0_search_start/self.N
        starting_eta0 = eta0
        change_factor = 5
        eta0 = eta0/change_factor  # so we don't skip first value

        num_pts = min(max_pts, self.N)
        random_indices = np.random.choice(self.N, num_pts, replace=False)
        X = self.X[random_indices]
        y = self.y[random_indices]
        model = self.copy()
        model.assess_test_data_during_fitting = False
        model.check_W_bar_vitals = False
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
                model.W = np.zeros(model.W.shape)
                model.progress_monitoring_freq = model.N
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
            print("--- eta0 didn't change; start 100x lower --- \n")
            self.eta0_search_start = self.eta0_search_start/5**3
            self.find_good_learning_rate()
        else:
            # return an eta almost as high as the biggest one one that
            # didn't cause divergence
            self.eta0 = eta0/change_factor
            self.eta = self.eta0
            print("===== eta0 search landed on {} ====".format(self.eta0))

    def apply_weights(self, X):
        """
        Calculate the prediction matrix: Y_bar = XW.  No bias.
        """
        return X.dot(self.get_weights())

    def step(self, X, Y):
        """
        Update the weights and bias, using X which *has* been transformed
        by the kernel
        """
        n, d = X.shape  # n and d of the sub-sample of X
        assert n == Y.shape[0]
        assert X.shape == (n, self.kernel.d)
        # TODO: be positive I use W for all the points so far.

        # TODO: apply kernel, or before the step.  Then assert it's right dim
        gradient = -(1./n)*X.T.dot(Y - X.dot(self.W))
        assert gradient.shape == (self.kernel.d, self.C)

        # TODO: do I scale eta by N still?
        # TODO: subtract the gradient for grad descent (?)
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
        assert self.W is not None, "Can't calc \hat{Y} without weights, W."
        X = self.X

        if X.shape[0] < chunk_size:
            chunk_size = X.shape[0]
        N = X.shape[0]
        n = 0
        if calc_for_W_bar:
            assert len(self.last_n_weights) > 0, "need weights for \bar{W}"
            Wbar = self.calc_W_bar()
            assert Wbar is not None

        def build_up_Yhat(X_chunk, Yhat, weights):
            assert weights is not None
            Yhat_chunk = X_chunk.dot(weights)
            if Yhat_chunk.shape != (X_chunk.shape[0], self.C):
                import pdb; pdb.set_trace()
            assert Yhat_chunk.shape == (X_chunk.shape[0], self.C)
            if Yhat is None:
                Yhat = Yhat_chunk
            else:
                Yhat = np.vstack([Yhat, Yhat_chunk])
            return Yhat

        Yhat = None
        Yhat_Wbar = None

        # for each chunk of X, transform to kernel and find Yhat.
        while n < N:
            # Find kernel-version of a chunk of X
            X_chunk = X[n: n+chunk_size, ]
            kernel_chunk = self.kernel.transform(X_chunk)
            assert kernel_chunk.shape == (X_chunk.shape[0], self.kernel.d)

            Yhat = build_up_Yhat(X_chunk=kernel_chunk, Yhat=Yhat, weights=self.W)
            if calc_for_W_bar:
                Yhat_Wbar = build_up_Yhat(X_chunk=kernel_chunk, Yhat=Yhat_Wbar,
                                          weights=Wbar)

            n += X_chunk.shape[0]

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
        return errors_squared.sum()

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
        if len(self.last_n_weights) >= 1 and self.check_W_bar_vitals:
            calc_for_W_bar = True
        else:
            # can't calculate anything before the weights have been set once.
            calc_for_W_bar = False

        if self.W is None:
            calc_Yhat = False
        else:
            calc_Yhat = True

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
            "points": [self.points_sampled],
            }
        row.update(more_details)
        kernel_info = self.kernel.info()
        row.update(kernel_info)
        self.Yhat = None  # wipe it so it can't be used incorrectly later
        self.Yhat_Wbar = None  # wipe it so it can't be used incorrectly later
        return row

    def calc_W_bar(self):
        """
        \bar{W} is the average weights over the last n fittings
        """
        assert len(self.last_n_weights) > 0, "need weights to calc \bar{W}"
        return np.array(self.last_n_weights).sum(axis=0)/\
               len(self.last_n_weights)

    def update_W_bar(self, weight_array, n=5):
        """
        \bar{W} is the average weights over the last n fittings

        It is built from a tuple of previous weights, stored in
        self.last_n_weights.
        """
        # Variance of new weights minus the recent average:

        if len(self.last_n_weights) >= n:
            self.last_n_weights.pop(0)
        self.last_n_weights.append(weight_array)
        self.W_bar = self.calc_W_bar()

    def run(self, max_divergence_streak_length=7, rerun=False):

        # To support running model longer, need to retrieve
        if rerun:
            old_square_loss_norm = \
                self.results.tail(1).reset_index()['(square loss)/N, training'][0]
            print("Before re-run, epochs = {}, steps = {}".format(
                self.epochs, self.steps))
        if len(self.last_n_weights) == 0:
            W_bar_available = False
        elif len(self.last_n_weights) >= 1:
            W_bar_available = True


        # Step until converged
        while self.epochs < self.max_epochs and not self.converged:

            if self.verbose:
                print('Begin epoch {}'.format(self.epochs))
            # Shuffle each time we loop through the entire data set.
            X, Y = self.shuffle(self.X.copy(), self.Y.copy())

            # initialize the statistic for tracking variance
            # Should be zero if weights are initially zero.
            if W_bar_available:
                old_W_bar_variance = self.W_bar_variance()
            else:
                old_W_bar_variance = 0

            # loop over ~all of the data points in little batches.
            num_pts = 0
            iter = 0
            while num_pts < self.N :
                iter += 1
                if self.points_sampled%self.progress_monitoring_freq == 0:
                    take_pulse = True
                # add extra monitoring for first few steps; this gives extra
                # awareness of model divergence.
                elif rerun and num_pts == 0:
                    take_pulse = True
                else:
                    take_pulse = False

                idx_start = num_pts
                idx_stop = num_pts + self.batch_size
                X_sample = X[idx_start:idx_stop, ] # works even if you ask for too many rows.
                # apply the kernel transformation
                X_sample = self.kernel.transform(X_sample)
                Y_sample = Y[idx_start:idx_stop, ]

                self.step(X_sample, Y_sample)

                num_pts += X_sample.shape[0]  # loop-scoped count
                last_pass = num_pts == self.N # True if last loop in epoch
                self.points_sampled += X_sample.shape[0]  # every point ever

                # assess \bar{W} every N points
                # or the first pass of a re-run
                if last_pass or (rerun and iter==1):
                    # update the average of recent weight vectors
                    W_bar_variance, W_bar_percent_improvement = \
                        self.W_bar_vitals(old_W_bar_variance)

                # take the more expensive pulse, using Yhat, which
                # requires kernel transformatin of all of X.
                if take_pulse:
                    start_time = datetime.datetime.now()
                    self.record_vitals(W_bar_available=W_bar_available)
                    if self.verbose:
                        stop_time = datetime.datetime.now()
                        print("Vitals done: {}.".format(
                            self.time_delta(start_time, stop_time)))

                    square_loss_norm = \
                        self.results.tail(1).reset_index()['(square loss)/N, training'][0]
                    assert square_loss_norm is not None, \
                        "square loss shouldn't be None"

                    if square_loss_norm/self.N > 1e3:
                        s = "square loss/N/N grew to {}".format(
                            square_loss_norm/self.N)
                        raise ModelFitException(s)

                if take_pulse and self.epochs > 1:
                    square_loss_percent_improvement = self.percent_change(
                        new = square_loss_norm, old = old_square_loss_norm)
                    if self.verbose:
                        print(square_loss_norm)
                    if self.test_convergence(square_loss_percent_improvement,
                                             W_bar_percent_improvement):
                        print("Loss optimized.  Old/N: {}, new/N:{}. Eta: {}"
                              "".format(old_square_loss_norm, square_loss_norm,
                                        self.eta))
                        self.converged = True
                        break
                    elif self.test_divergence(n=max_divergence_streak_length):
                        raise ModelFitException(
                            "\nSquare loss grew {} measurements in a row!"
                            "".format(max_divergence_streak_length))

                old_square_loss_norm = square_loss_norm

                if last_pass:
                    # record variables for next loop
                    old_W_bar_variance = W_bar_variance

            self.epochs +=1
            sys.stdout.write(".") # one dot per pass through ~ N pts
            if self.epochs == self.max_epochs:
                print('\n!!! Max epochs ({}) reached. !!!'.format(self.max_epochs))

            # shrink learning rate
            #self.shrink_eta(self.epochs - fast_convergence_epochs + 1)
            self.shrink_eta()

        print('final normalized training (square loss): {}'.format(square_loss_norm))
        self.results.reset_index(drop=True, inplace=True)

    def run_longer(self, epochs, progress_monitoring_freq=None,
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

        if progress_monitoring_freq is not None:
            self.progress_monitoring_freq = progress_monitoring_freq

        if max_divergence_streak_length is not None:
            self.run(max_divergence_streak_length=max_divergence_streak_length)
        else:
            self.run(rerun=True)

    def W_bar_variance(self):
        assert len(self.last_n_weights) > 0, "Need weights to calc weight var"
        return np.var(self.calc_W_bar())

    def W_bar_vitals(self, old_W_bar_variance):
        self.update_W_bar(self.W, n=5)
        new_W_bar_variance = self.W_bar_variance()

        W_bar_percent_improvement = self.percent_change(
            new=new_W_bar_variance, old=old_W_bar_variance)

        # record the improvement:
        W_bar_improvement = pd.DataFrame(
            {'epoch':[self.epochs],
             '\bar{W} variance % change': [W_bar_percent_improvement]})
        # record it in our tracker.
        self.W_bar_variance_df = pd.concat([self.W_bar_variance_df,
                                            W_bar_improvement], axis=0)

        return new_W_bar_variance, W_bar_percent_improvement

    def record_vitals(self, W_bar_available):
        row_results = pd.DataFrame(self.results_row())
        if row_results.shape[0] != 1:
            import pdb; pdb.set_trace()
        assert row_results.shape[0] == 1, "row_results should have 1 row"

        if self.check_W_bar_vitals and W_bar_available:
            W_bar_results = self.assess_model_using_W_bar()
            if W_bar_results.shape[0] != 1:
                import pdb; pdb.set_trace()
            assert W_bar_results.shape[0] == 1, \
                "\bar{W} results should have 1 row"

        # also find the square loss & 0/1 loss using test data.
        if self.assess_test_data_during_fitting:
            assert (self.test_X is not None) and (self.test_y is not None), \
                "Asked for test results but no test data was provided."
            test_results = self.assess_model_on_test_data()
            merged_results = pd.merge(row_results, test_results)
            assert merged_results.shape[0] == 1
            if merged_results.shape[0] != 1:
                import pdb; pdb.set_trace()
            row_results = merged_results # merge worked

        self.results = pd.concat([self.results, row_results], axis=0)

    def assess_model_using_W_bar(self):
        """
        Note: this has nothing to do with model fitting.
        It is only for reporting and gaining intuition.
        """

        # Make a copy of the model and replace the weights with the W_bar
        # weights
        assert len(self.last_n_weights) > 0, "Need weights to do \\bar{W} stuff"
        model = self.copy(reset=False)
        model.W = self.calc_W_bar()

        # Get the results, the usual way:
        all_W_bar_results = model.results_row()
        results = {re.sub("training", "bar{W}", k): v
           for k, v in all_W_bar_results.items()}
        # don't need step if merging on.
        columns = [c for c in results.keys() if 'bar{W}' in c]
        results = pd.DataFrame(results)
        assert results.shape[0] == 1, "Results for bar{W} should be length 1"
        return results[columns]

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

    def test_convergence(self, square_loss_percent_improvement,
                         W_bar_percent_improvement):
        if self.percent_metrics_converged(square_loss_percent_improvement,
                                          W_bar_percent_improvement):
            # record convergence status
            self.converged = True # flag that it converged.

            return True

        else:
            return False

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

    def plot_W_bar_history(self):
        x = 'epoch'
        y1 = '\bar{W} variance % change'
        self.plot_ys(df=self.W_bar_variance_df, x=x, y1=y1, y2=None,
                     ylabel= "bar{W} variance % change",
                     y0_line=True, logx=False, logy=False,
                     colors=None, figsize=(4, 3))

    def plot_loss_of_both_W_arrays(self):
        """
        One plot showing the both the squared loss after every epoch

        Plot the loss of both wt and the average weight vector wτ .
        You shave both the training and test losses on the same plot
        (so there should be four curves)
        """
        pass

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

