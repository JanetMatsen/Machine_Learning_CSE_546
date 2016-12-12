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
    def __init__(self, X, y, hidden_TF, output_TF, test_X, test_y):
        self.X = X
        self.y = y
        self.hidden_TF = hidden_TF
        self.output_TF = output_TF
        self.test_X = test_X
        self.test_y = test_y
        self.num_models = 0
        self.models = dict()
        self.summary = pd.DataFrame()

    def train_model(self, hiddenTF_kwargs, outputTF_kwargs, eta0=1e-3,
                    epochs=3, verbose=True):
        self.num_models += 1
        try:
            m = NeuralNet(X=self.X, y=self.y,
                          X_test=self.test_X, y_test=self.test_y,
                          eta0=eta0,
                          hiddenTF = self.hidden_TF,
                          outputTF = self.output_TF,
                          hidden_nodes=500,  # todo: generalize
                          minibatch_size = 10,  # todo: generalize
                          hiddenTF_kwargs=hiddenTF_kwargs,
                          outputTF_kwargs=outputTF_kwargs,
                          verbose=verbose)
            m.run(epochs=epochs)
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

        except NameError:
            print("model construction failed for {}, {}"
                  "".format(**hiddenTF_kwargs, **outputTF_kwargs))


        self.models[self.num_models] = m
        print("saved as model # {}".format(self.num_models))

    def test_tuples_of_W_init_scales(self, tuple_list):
        for t in tuple_list:
            print("testing with scale_W1={}, scale_W2={}".format(t[0], t[1]))
            print(t)
            print(type(t))
            hiddenTF_kwargs={"scale_W1":t[0]}
            outputTF_kwargs={"scale_W2":t[1]}
            self.train_model(hiddenTF_kwargs=hiddenTF_kwargs,
                             outputTF_kwargs=outputTF_kwargs)

    def test_combo_of_tuples(self, scale_list):
        value_tuples = list(itertools.product(scale_list, scale_list))
        print(value_tuples)
        self.test_tuples_of_W_init_scales(value_tuples)

    def plot_tests_of_W1_init_scaling(self):
        piv = self.summary.pivot(index='log10(scale_W1)',
                              columns='log10(scale_W2)',
                              values='(0/1 loss)/N, testing')
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        sns.heatmap(piv, ax=ax)
        plt.tight_layout()
        return fig

