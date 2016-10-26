from functools import partial
import itertools

import pandas as pd


class HyperparameterSweep:
    def __init__(self, X, y, model, validation_split=None,
                 test_X=None, test_y=None):
        # model_partial is a model that's missing one or more parameters.
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        self.summary = pd.DataFrame()
        # split off a validation set from X
        if validation_split is not None:
            split_index = int(self.N*(1-validation_split))
            self.train_X = X[0:split_index, :]
            self.validation_X = X[split_index:, :]
            self.train_y = y[0:split_index]
            self.validation_y = y[split_index:]
            self.model_partial = \
                partial(model, X=self.train_X, y=self.train_y,
                        lam=0, max_iter=2000)
        if test_X is not None:
            self.test_X = test_X
        if test_y is not None:
            self.test_y = test_y
        # keep track of model numbers.
        self.num_models = 0
        self.models = {}

    def train_model(self, eta0, plot_trajectory=True):
        # train model
        try:
            m = self.model_partial(eta0=eta0)
            self.num_models += 1
            m.run()
            # save outcome
            self.models[self.num_models] = m
            # get results
            outcome = m.results_row()
            outcome['model number'] = [self.num_models]
            outcome = pd.DataFrame(outcome)
            self.summary = pd.concat([self.summary, outcome])
            if plot_trajectory:
                m.plot_log_loss_normalized_and_eta()
        except:
            # save failure message.
            pass


