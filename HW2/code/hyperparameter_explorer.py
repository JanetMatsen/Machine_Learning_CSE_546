from functools import partial
import matplotlib.pyplot as plt
import scipy.sparse as sp
import re

import pandas as pd


class HyperparameterExplorer:
    def __init__(self, X, y, model, score_name, validation_split=None,
                 test_X=None, test_y=None, use_prev_best_weights=True):
        # model_partial is a model that's missing one or more parameters.
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        self.summary = pd.DataFrame()
        # split off a validation set from X
        if validation_split is not None:
            split_index = int(self.N*(1-validation_split))
            print("{} of {} points from training are reserved for "
                  "validation".format(self.N - split_index, self.N))
            self.train_X = X[0:split_index, :]
            self.validation_X = X[split_index:, :]
            self.train_y = y[0:split_index]
            self.validation_y = y[split_index:]
            self.model = partial(model, X=self.train_X, y=self.train_y)
            # self.model_partial = \
            #     partial(model, X=self.train_X, y=self.train_y,
            #             lam=0, max_iter=2000)
        if test_X is not None:
            self.test_X = test_X
        if test_y is not None:
            self.test_y = test_y
        # keep track of model numbers.
        self.num_models = 0
        self.models = {}
        # the score that will be used to determine which model is best.
        self.score_name = score_name
        self.validation_score_name = re.sub("training", "validation", score_name)
        self.use_prev_best_weights = use_prev_best_weights

    def train_model(self, **kwargs):
        # train model
        # check that model was made
        try:
            m = self.model(**kwargs)
            # set weights to the best found so far
            # Note: this is silly for non-iterative solvers like Ridge.
            if self.use_prev_best_weights:
                # TODO: this wasn't working for Ridge Multiclass...
                if "W" in m.__dict__.keys() and len(self.models) > 0:
                    m.W = self.best_weights(self.score_name).copy()
                    if m.sparse:
                        m.W = sp.csc_matrix(m.W)
                elif ("w" in m.__dict__.keys()) and len(self.models) >0:
                    m.w = self.best_weights(self.score_name).copy()
        except NameError:
            print("model failed for {}".format(**kwargs))

        self.num_models += 1
        m.run()
        # save outcome of fit.  Includes training data 0/1 loss, etc.
        self.models[self.num_models] = m
        # get results
        outcome = m.results_row()
        if len(outcome) < 1:
            print("model didn't work..?")
        # Save the model number for so we can look up the model later
        outcome['model number'] = [self.num_models]

        # calculate the outcome for the validation data.
        # need a new model to do this.
        validation_model = self.model(X=self.validation_X,
                                      y=self.validation_y,
                                      # won't use lam b/c not training
                                      lam=None)
        # give the new model the trained model's weights.
        if "W" in m.__dict__.keys():
            validation_model.W = m.W.copy()
            assert type(validation_model.W) == sp.csc_matrix, \
                "type of W is {}".format(type(validation_model.W))
            # ensure everything is sparse if necessary (Ridge)
            if validation_model.sparse:
                validation_model.X = sp.csc_matrix(validation_model.X)
                validation_model.W = sp.csc_matrix(validation_model.W)
        elif "w" in m.__dict__.keys():
            validation_model.w = m.w.copy()
        validation_results = validation_model.results_row()

        # pick out the good results.
        outcome[self.validation_score_name] = \
            validation_results[self.score_name][0]
        outcome = pd.DataFrame(outcome)

        # Append this new model's findings onto the old model.
        self.summary = pd.concat([self.summary, outcome])

        # Plot log loss vs time if applicable.
        if "log loss" in self.summary.columns:
            m.plot_log_loss_normalized_and_eta()

    def best_model(self):
        """
        Find the best model according to the validation data
        via the Pandas DataFrame.
        """
        # get the index of the model with the best score
        i = self.summary[[self.validation_score_name]].idxmin()
        best_score = self.summary[self.validation_score_name].iloc(i)
        model = self.models[1]
        print("best {} = {}; found in model {}".format(
            self.validation_score_name, best_score, i))
        return model

    def best_weights(self):
        '''
        get best weights using the validation set.
        '''
        best_model = self.best_model(self.validation_score_name)
        return best_model.get_weights()

    def plot_fits(self):
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        plot_data = self.summary.sort('lambda')
        plt.semilogx(plot_data['lambda'], plot_data['validation RMSE'],
                    linestyle='--', marker='o', c='g')
        plt.semilogx(plot_data['lambda'], plot_data['training RMSE'],
                    linestyle='--', marker='o', c='grey')
        plt.legend(loc='best')
        plt.xlabel('lambda')
        plt.ylabel('RMSE')
        ax.axhline(y=0, color='k')



