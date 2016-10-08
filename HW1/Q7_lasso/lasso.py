import numpy as np


class lasso:
    def __init__(self, X, lam, delta, wo=0):
        """

        :param X: a numpy array of features
        :param wo: initial weight to use for bias
        """
        self.wo = wo
        self.num_points = None
        self.lam = lam

    def calculate_yhat(self):
        pass

    def update_wo(self):
        pass

    def update_yhat(self):
        pass

    def update_wk(self):
        pass

    @staticmethod
    def test_convergence(old_w, new_w, delta):
        # stop when no element of w changes by more than some small delta
        # return True if it is time to stop.
        pass

    @staticmethod
    def check_for_objective_increase():
        # need a nonincreasing objective value
        pass


class regularization_path:
    def __init__(self):
        pass

    def determine_smallest_lambda(self):
        pass
