from functools import partial
import itertools
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse as sp
import re

import pandas as pd
import seaborn as sns

from NeuralNet import NeuralNet

class HyperparameterExplorer:
    def __init__(self, X, y, hiddenTF, outputTF,
                 learning_rate, epochs, minibatch_size,
                 test_X, test_y):
        self.X = X
        self.y = y
        self.hidden_TF = hiddenTF
        self.output_TF = outputTF
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.minibatch_size=minibatch_size
        self.test_X = test_X
        self.test_y = test_y
        self.num_models = 0
        self.models = dict()
        self.summary = pd.DataFrame()

    def train_model(self, hiddenTF_kwargs, outputTF_kwargs, verbose=True):
        self.num_models += 1
        try:
            print('eta0: {}'.format(self.learning_rate))
            m = NeuralNet(X=self.X, y=self.y,
                          X_test=self.test_X, y_test=self.test_y,
                          eta0=self.learning_rate,
                          hiddenTF = self.hidden_TF,
                          outputTF = self.output_TF,
                          hidden_nodes=500,  # todo: generalize
                          minibatch_size = self.minibatch_size,
                          hiddenTF_kwargs=hiddenTF_kwargs,
                          outputTF_kwargs=outputTF_kwargs,
                          verbose=verbose)
            m.run(epochs=self.epochs)
            summary = m.results.tail(1).copy() # copy before appending
            summary['model #'] = self.num_models
            summary["scale_W1"] = int(hiddenTF_kwargs['scale_W1'])
            summary["scale_W2"] = int(outputTF_kwargs['scale_W2'])
            summary["log10(scale_W1)"] = \
                np.log10(int(hiddenTF_kwargs['scale_W1']))
            summary["log10(scale_W2)"] = \
                np.log10(int(outputTF_kwargs['scale_W2']))
            self.summary = pd.concat([self.summary, summary], axis=0)
            m.model_number = self.num_models

        except:
            print("=== ** model construction failed for ** ===")
            print(hiddenTF_kwargs)
            print(outputTF_kwargs)
            print("=== ** === ** ===")


        self.models[self.num_models] = m
        print("saved as model # {}".format(self.num_models))

    def test_tuples_of_W_init_scales(self, tuple_list):
        for t in tuple_list:
            print("testing with scale_W1={}, scale_W2={}".format(t[0], t[1]))
            hiddenTF_kwargs={"scale_W1":t[0]}
            outputTF_kwargs={"scale_W2":t[1]}
            self.train_model(hiddenTF_kwargs=hiddenTF_kwargs,
                             outputTF_kwargs=outputTF_kwargs)

    def test_combo_of_tuples(self, scale_list, scale_listW2=None):
        """
        specify a W1 scale set and a W2 scale set separately if desired.
        """
        if scale_listW2 is None:
            # do all combinations
            value_tuples = list(itertools.product(scale_list, scale_list))
        else:
            # do all combinations of list 1 as W1 and list2 as W2
            value_tuples = list(itertools.product(scale_list, scale_listW2))
        print(value_tuples)
        self.test_tuples_of_W_init_scales(value_tuples)

    def plot_tests_of_W1_init_scaling(self):
        piv = self.summary.pivot(index='log10(scale_W1)',
                              columns='log10(scale_W2)',
                              values='(0/1 loss)/N, testing')
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        sns.heatmap(piv, ax=ax, cmap='Greens_r')
        plt.title('hyperparameter exploration: eta={}, epoochs={}'
                  ''.format(self.learning_rate, self.epochs))
        plt.tight_layout()
        return fig

