import numpy as np
import scipy.sparse as sp

class Lasso:
    def __init__(self, X, y, lam, w, w0=0, delta=0.01):
        """

        :param X: A scipy.csc matrix (sparse matrix) of features.
                  Rows are training data and columns are features.
        :param wo: initial weight to use for bias.  Not sparse.
        """
        assert type(X) == sp.csc_matrix
        self.X = X
        assert type(y) == np.ndarray
        self.y = y
        assert w.shape[0] == self.X.shape[1]
        self.w = w        # sp.csc_matrix
        self.w0 = w0*1.
        self.lam = lam
        self.delta = delta
        self.yhat = None  # sp.csc_matrix

    def optimize_weights(self):
        print("begin optimizing weights")

        # have to initialize old_w before entering the while loop.
        old_w = self.w + \
                sp.csc_matrix(np.ones(self.w.shape[0])).T*2*self.delta #sparse

        assert(self.test_convergence(old_w) is False)

        # check that the weights haven't converged.
        while not self.test_convergence(old_w):
            old_w = self.w   #sparse

            # calculate predictions to avoid numerical drift.
            old_yhat = self.calculate_yhat()  #not sparse

            # update the bias.
            old_w0 = self.w0  # get it before we write over it
            self.update_w0(old_yhat)  # updates self.w0

            # update our predictions, yhat, using the new w0.
            self.update_yhat(old_yhat, old_w0)  # updates self.yhat

            # iterate over the d features
            for k in range(1, self.w.shape[0]):
                self.update_wk(k)

                # update the values of w, w0 for the next loop.

            # check that the objective function decreased
            old_objective_fun_val = None  #todo: calc
            new_objective_fun_val = None  #todo: calc
            #assert(new_objective_fun_val < old_objective_fun_val)

        print("Lasso optimized.")
        print("w: {}, w0: {}".format(self.w.toarray(), self.w0))

        return

    def calculate_yhat(self):
        # multiply X*w + w_o
        # returns vector of predictions, yhat.
        print("Calc X*w + w_o for w = {}, w_o={}".format(self.w, self.w0))

        # don't want to store this yhat.  Temporary.
        yhat = self.X.dot(self.w) + self.w0
        yhat = yhat.toarray()
        assert yhat.shape[1] == 1
        return yhat

    def update_w0(self, old_yhat):
        new_w0 = sum(self.y - old_yhat) + self.w0
        assert type(new_w0*1.0 == float)  # sometimes I had int.
        return new_w0

    def update_yhat(self, old_yhat, old_w0):
        new_yhat = old_yhat + old_w0 - self.w0
        assert type(new_yhat == np.array)
        assert new_yhat.shape[1] == 1
        self.yhat = new_yhat

    def update_wk(self, k):
        Xik = self.X[:, k-1]  # array  (slice of sparse matrix)
        wk = self.w[k-1].toarray()[0][0]  # scalar

        ak = 2*Xik.sum()

        tmp = self.y - self.yhat + Xik.toarray()*wk
        assert tmp.shape[1] == 1  # column vector
        tmp2 = Xik.T.dot(tmp)
        assert tmp2.shape == (1,1)
        ck = 2*(tmp2)[0][0]

        # apply the update rule.
        print("ck: {}, lambda: {}".format(ck, self.lam))
        if ck < - self.lam:
            self.w[k-1] = (ck + self.lam)/ak
        elif ck > self.lam:
            self.w[k-1] = (ck - self.lam)/ak
        else:
            # todo: assert in range
            self.w[k-1] = 0

    @staticmethod
    def calc_objective_fun(X, w, w0, y):
        pass

    def test_convergence(self, old_w):
        # stop when no element of w changes by more than some small delta
        # return True if it is time to stop.
        # see HW pg 8.

        print("Testing convergence.")
        print("Old w: {}, new w: {}".format(old_w, self.w))

        # make a vector of the differences in each element
        delta_w = old_w - self.w

        for w_i in delta_w :
            if w_i > self.delta:
                return False
            else:
                return True

    @staticmethod
    def check_for_objective_increase():
        # need a nonincreasing objective value
        pass



class RegularizationPath:
    def __init__(self):
        pass

    def determine_smallest_lambda(self):
        pass
