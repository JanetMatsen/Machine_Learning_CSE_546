from functools import partial
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import re

import pandas as pd


class HyperparameterExplorer:
    def __init__(self, X, y, classifier, score_name, primary_hyperparameter,
                 validation_split=0.1, test_X=None, test_y=None,
                 # only used for final training on all of the training data.
                 assess_test_data_during_fitting=True,
                 use_prev_best_weights=True):

        # check data
        assert X.shape[0] == y.shape[0]
        # Sometimes we check the loss of test_X and test_y during fitting
        if test_X is None and test_y is None:
            print("No test data was provided.")
        if test_X is not None and test_y is not None:
            assert test_X.shape[0] == test_y.shape[0]

        # model_partial is a model that's missing one or more parameters.
        self.all_training_X = X  # reserved for final training after hyper sweep.
        self.all_training_y = y  # reserved for final training after hyper sweep.
        self.N, self.d = X.shape
        self.summary = pd.DataFrame()
        self.primary_hyperparameter = primary_hyperparameter

        # split off a validation set from X
        split_index = int(self.N*(1-validation_split))
        print("{} of {} points from training are reserved for "
              "validation".format(self.N - split_index, self.N))
        self.train_X = X[0:split_index, :]
        self.validation_X = X[split_index:, :]
        self.train_y = y[0:split_index]
        self.validation_y = y[split_index:]
        print('variances of all training data: {}'.format(np.var(y)))
        print('variances of split-off training & validation '
              'data: {}, {}'.format(np.var(self.train_y),
                                    np.var(self.validation_y)))

        if test_X is not None and test_y is not None:
            self.assess_test_data_during_fitting = assess_test_data_during_fitting
            self.test_X = test_X
            self.test_y = test_y
            self.model = partial(classifier,
                                 X=self.train_X, y=self.train_y,
                                 test_X=test_X, test_y=test_y)
        else:
            self.model = partial(classifier, X=self.train_X, y=self.train_y)

        # keep track of model numbers.
        self.num_models = 0
        self.models = {}

        # the score that will be used to determine which model is best.
        self.training_score_name = score_name
        self.score_name = re.sub("training", "", score_name)
        self.validation_score_name = re.sub("training", "validation", score_name)
        self.use_prev_best_weights = use_prev_best_weights

    def train_model(self, kernel_kwargs, model_kwargs, assess_test_data_during_fitting=False):
        # train model
        # check that model was made
        try:
            model_kwargs['assess_test_data_during_fitting'] = assess_test_data_during_fitting
            m = self.model(kernel_kwargs=kernel_kwargs, **model_kwargs)
            # set weights to the best found so far
            # Note: this is silly for non-iterative solvers like Ridge.
            if self.use_prev_best_weights and self.summary.shape[0] >=1:
                print('Starting with best weights from previous model(s)')
                best_weights = self.best('weights')
                if best_weights is not None:
                    m.replace_weights(best_weights.copy())

        except NameError:
            print("model construction failed for {}".format(**model_kwargs))

        self.num_models += 1
        self.model_number = self.num_models
        m.run()
        # save outcome of fit.  Includes training data 0/1 loss, etc.
        self.models[self.num_models] = m
        print("saved as model # {}".format(self.num_models))

        # Check if model converged, and assess it if so
        if m.converged:
            print("Model converged; assess it using validation data.")

        # assess the model even if it didn't converge
        self.assess_model(m)

    def assess_model(self, model):
        m = model
        # get results
        outcome = m.results.tail(1).reset_index()
        if len(outcome) < 1:
            print("model didn't work..?")
        # Save the model number for so we can look up the model later
        outcome['model number'] = [self.num_models]
        outcome['converged'] = [m.converged]
        print('{}:{}'.format(self.training_score_name,
                             outcome[self.training_score_name][0]))

        validation_results = m.apply_model(X=self.validation_X,
                                           y=self.validation_y,
                                           data_name='validation')
        validation_results = pd.DataFrame(validation_results)
        v_columns = [c for c in validation_results.columns
                     if 'validation' in c or self.primary_hyperparameter == c]
        outcome = pd.merge(pd.DataFrame(outcome),
                           validation_results[v_columns])

        # Append this new model's findings onto the old model.
        self.summary = pd.concat([self.summary, outcome])
        # Oh, silly Pandas:
        self.summary.reset_index(drop=True, inplace=True)

        # Plot log loss vs time if applicable.
        try:
            m.plot_loss_normalized_and_eta()
            m.plot_w_hat_history()
        except:
            print("not all plotting calls worked.")

    def best(self, value='model'):
        """
        Find the best model according to the validation data
        via the Pandas DataFrame.

        :param value: a string describing what you want from the best model.
        """
        # get the index of the model with the best score
        i = self.summary[[self.validation_score_name]].idxmin()[0]
        i = self.summary['model number'][i]
        if value == 'model number':
            return i

        summary_row = self.summary[self.summary['model number'] == i]
        if value == 'summary':
            return summary_row.T

        best_score = summary_row[self.validation_score_name]
        if value == 'score':
            return best_score

        model = self.models[i]
        if value == 'model':
            return model

        if value == 'weights':
            return model.get_weights()

        print("best {} = {}; found in model {}".format(
            self.validation_score_name, best_score, i))
        return model

    def best_results_across_hyperparams(self):
        """
        Group summary results by lambda and return a summary of the best
        validation score result for each lambda tested so far.
        """
        if self.summary.shape[0] == 0:
            return None
        # best losses at each lambda:
        p = self.primary_hyperparameter
        idx = self.summary.groupby([p])[self.validation_score_name].\
                  transform(min) == self.summary[self.validation_score_name]
        return self.summary[idx]

    def plot_fits(self, df=None, x=None, y1=None, y2=None,
                  filename=None, xlim=None, ylim=None, logx=True):
        if x is None:
            x = self.primary_hyperparameter
        if df is None:
            df = self.summary
        if y1 == None:
            y1 = self.validation_score_name
        if y2 == None:
            y2 = self.training_score_name
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        plot_data = df.sort(x)
        if logx:
            plt.semilogx(plot_data[x], plot_data[y1],
                        linestyle='--', marker='o', c='g')
            plt.semilogx(plot_data[x], plot_data[y2],
                         linestyle='--', marker='o', c='grey')
        else:
            plt.plot(plot_data[x], plot_data[y1],
                         linestyle='--', marker='o', c='g')
            plt.plot(plot_data[x], plot_data[y2],
                         linestyle='--', marker='o', c='grey')

        plt.legend(loc='best')
        plt.xlabel(x)
        plt.ylabel(self.score_name)
        ax.axhline(y=0, color='k')
        if xlim:
            ax.set_xlim([xlim[0],xlim[1]])
        if ylim:
            ax.set_ylim([ylim[0],ylim[1]])

        plt.tight_layout()
        if filename is not None:
            fig.savefig(filename + '.pdf')

    def train_on_whole_training_set(self, max_steps=None, delta_percent=None):
        # get the best model conditions from the hyperparameter exploration,
        # and print it to ensure the user's hyperparameters match the best
        # models's.:
        #print("best cross-validation model's info:")
        #print(self.best('summary'))
        print("getting best model.")
        self.final_model = self.best('model').copy()
        self.final_model.assess_test_data_during_fitting = self.assess_test_data_during_fitting
        print(self.final_model.results.tail(1))

        # replace the smaller training sets with the whole training set.
        self.final_model.replace_X_and_y(self.all_training_X,
                                         self.all_training_y)
        if max_steps:
            self.final_model.max_steps = max_steps
        if delta_percent:
            self.delta_percent = delta_percent

        # find the best weights using all the data
        self.final_model.run()

    def evaluate_test_data(self):
        test_results = self.final_model.apply_model(
            X=self.test_X, y=self.test_y, data_name="test")
        print(pd.DataFrame(test_results).T)


