from functools import partial
import itertools

import pandas as pd


class HyperparameterExplorer:
    def __init__(self, X, y, model, score_name, validation_split=None,
                 test_X=None, test_y=None):
        # model_partial is a model that's missing one or more parameters.
        self.X = X
        self.y = y
        self.N, self.d = X.shape
        self.summary = pd.DataFrame()
        # split off a validation set from X
        if validation_split is not None:
            split_index = int(self.N*(1-validation_split))
            print("{} of {} points from training are reserved for "
                  "validation".format(split_index, self.N))
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

    def train_model(self, **kwargs):
        # train model
        # check that model was made
        try:
            m = self.model(**kwargs)
            # set weights to the best found so far
            # Note: this is silly for non-iterative solvers like Ridge.
            if "W" in m.__dict__.keys() and len(self.models) > 0:
                m.W = self.best_weights(self.score_name)
            elif ("w" in m.__dict__.keys()) and len(self.models) >0:
                m.w = self.best_weights(self.score_name)
        except NameError:
            print("model failed for {}".format(**kwargs))

        try:
            self.num_models += 1
            m.run()
            # save outcome
            self.models[self.num_models] = m
            # get results
            outcome = m.results_row()
            # Save the model
            outcome['model number'] = [self.num_models]
            # calculate the outcome for the validation data.
            # need a new model to do this.
            validation_model = self.model(X=self.validation_X,
                                          y=self.validation_y,
                                          # won't use lam b/c not training
                                          lam=None)
            if "W" in m.__dict__.keys():
                validation_model.W = m.W.copy()
            elif "w" in m.__dict__.keys():
                validation_model.w = m.w.copy()
            validation_results = validation_model.results_row()
            #
            outcome['validation ' + self.score_name] = \
                validation_results[self.score_name][0]
            outcome = pd.DataFrame(outcome)
            self.summary = pd.concat([self.summary, outcome])
            if "log loss" in self.summary.columns:
                m.plot_log_loss_normalized_and_eta()

        except:
            # save failure message.
            pass

    def best_model(self, score_name):
        """
        Find the best model according to the validation data
        via the Pandas DataFrame.
        """
        # get the index of the model with the best score
        i = self.summary[['validation ' + score_name]].idxmin()
        best_score = self.summary[score_name].iloc(i)
        model = self.models[1]
        print("best {} = {}; found in model {}".format(
            score_name, best_score, i))
        return model

    def best_weights(self, score_name=None):
        '''
        get best weights using the validation set.
        '''
        if score_name is None:
            score_name = self.score_name
        best_model = self.best_model(score_name)
        return best_model.get_weights()



