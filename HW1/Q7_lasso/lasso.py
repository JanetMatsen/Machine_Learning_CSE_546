import numpy as np
import scipy.sparse as sp
# analyze my solution by comparing objective functions
from sklearn import linear_model
import pandas as pd


class SparseLasso:
    def __init__(self, X, y, lam, w=None, w0=0, delta=0.0001,
                 verbose = False, max_iter = 100000):
        """

        :param X:
        :param y: don't pass in a transposed y
        :param lam:
        :param w: don't pass in a transposed w
        :param w0:
        :param delta:
        :param verbose:
        :param max_iter:
        """
        # TODO: make w if it is not provided

        self.X = sp.csc_matrix(X)
        self.y = sp.csc_matrix([y]).T
        self.w = sp.csc_matrix([w]).T
        self.w0 = w0
        self.lam = lam
        self.delta = delta
        self.N, self.d = self.X.shape
        self.verbose = verbose
        self.max_iter = max_iter

    def w0_array(self):
        return np.ones((self.N, 1))*self.w0

    def objective(self):
        error_v = self.X.dot(self.w) + self.w0_array() - self.y
        return self.extract_scalar(error_v.T.dot(error_v)) + \
                self.lam*self.l1_norm(self.w)

    def step(self):
        yhat = self.X.dot(self.w) + self.w0_array()
        old_w0 = self.w0
        self.w0 += (self.y - yhat).sum()/self.N
        yhat += self.w0 - old_w0

        for k in range(0, self.d):
            Xk = self.X[:, k]
            ak = 2 * self.extract_scalar(Xk.T.dot(Xk))
            ck = 2 * \
                self.extract_scalar(Xk.T.dot(self.y - yhat + Xk*self.w[k, 0]))
            old_wk = self.w[k, 0]
            if ck < - self.lam:
                self.w[k, 0] = (ck + self.lam)/ak
            elif ck > self.lam:
                self.w[k, 0] = (ck - self.lam)/ak
            else:
                self.w[k, 0] = 0.
            yhat += Xk*(self.w[k, 0] - old_wk)

    def run(self):
        for s in range(0, self.max_iter):
            old_objective = self.objective()
            old_w = self.w.copy()
            self.step()
            if(old_objective - self.objective() <= -1e-5):
                print(old_objective)
                print(self.objective())
            assert(old_objective - self.objective() > -1e-5)
            if abs(old_w - self.w).max() < self.delta:
                break

        print(self.objective())
        print(self.w)

    @staticmethod
    def l1_norm(v):
        assert(v.shape[1] == 1)
        return abs(v).sum()

    @staticmethod
    def extract_scalar(m):
        assert(m.shape == (1,1))
        return m[0,0]

    def calc_yhat(self):
        return self.X.dot(self.w) + self.w0_array()


def sklearn_comparison(X, y, lam, sparse = False):
    alpha = lam/(2.*X.shape[0])
    clf = linear_model.Lasso(alpha)
    clf.fit(X, y)
    # store solutions in my Lasso class so I can look @ obj fun
    dummy_weights =  X[1,:].T  # will write over this
    assert dummy_weights.shape == (X.shape[1], 1)
    skl_lasso = Lasso(X, y, lam, w=dummy_weights,
                      w0=0, verbose = False, sparse=sparse)
    skl_lasso.w = sp.csc_matrix(clf.coef_).T
    skl_lasso.w0 = clf.intercept_

    skl_objective_fun_value = skl_lasso.calc_objective_fun()

    return({"objective": skl_objective_fun_value,
            "weights": clf.coef_,
            "intercept": clf.intercept_})


def generate_random_data(N, d, sigma, k=5):
    assert(d > N)

    # generate w0
    w0 = 0

    # generate X
    X = np.reshape(np.random.normal(0, 1, N*d),
                   newshape = (N, d), order='C')
    assert X.shape == (N, d)

    # generate w* with the first k elements being nonzero.
    # todo: k is hard coded for now.
    w = np.zeros((d, 1), dtype=float)
    w[0] = 10
    w[1] = -10
    w[2] = 10
    w[3] = -10
    w[4] = 10
    assert w.shape == (d, 1)

    # generate error
    e = np.reshape(np.random.normal(0, sigma**2, N), newshape = (N, 1))
    assert e.shape == (N, 1)

    # generate noisy Y
    Y = X.dot(w) + w0 + e
    Y.reshape(N, 1)
    assert Y.shape == (N, 1)

    assert X.shape == (N, d)
    assert Y.shape == (N, 1)
    assert w.shape == (d, 1)
    return X, Y, w


class paramSweepRandData():
    def __init__(self, N, d, sigma, init_lam, frac_decrease, k=5):
        self.N = N
        self.d = d
        self.sigma = sigma
        self.k = k
        self.init_lam = init_lam
        self.frac_decrease = frac_decrease
        X, Y, true_weights = generate_random_data(N=N, d=d,
                                                  sigma=sigma, k=k)
        self.X = X
        self.Y = Y
        self.true_weights = true_weights

    def sklearn_weights(self, lam):

        # compute the "correct" answer:
        #sklearn_weights = sklearn_comparison(X, Y, lam)['weights']
        alpha = lam/self.X.shape[0]
        clf = linear_model.Lasso(alpha)
        clf.fit(self.X, self.Y)
        return clf.coef_

    def loop_lambda(self):
        # protect the first value of lambda.
        lam = self.init_lam/self.frac_decrease

        # initialize a dataframe to store results in
        results = pd.DataFrame()
        for c in range(0,5):
            lam = lam*self.frac_decrease

            sklearn_weights = self.sklearn_weights(lam)

            # Compute my (hopefully correct) answer:
            result = Lasso(self.X, self.Y, lam)
            assert result.w.shape == (self.d, 1) # check before we slice out
            regression_weights = result.w.toarray()[:,0]

            precision, recall = \
                self.calc_precision_and_recall(regression_weights)

            one_val = pd.DataFrame({"sigma":[self.sigma], "lam":[lam],
                                    "precision":[precision],
                                    "recall":[recall],
                                    "sklearn weights":[sklearn_weights],
                                    "my weights":[regression_weights]})
            results = pd.concat([results, one_val])

            self.results = results

    def calc_precision_and_recall(self, regression_weights, z=0.001):
        # True array for regression weight ~ 0:
        true_weights_array = self.true_weights.reshape(1, self.d)
        abs_weights = np.absolute(true_weights_array)
        nonzero_weights = abs_weights > z

        reg_weight_array = regression_weights.reshape(1, self.d)
        abs_reg_weights = np.absolute(reg_weight_array)
        nonzero_reg_weights = abs_reg_weights > z

        agreement = np.bitwise_and(nonzero_weights, nonzero_reg_weights)
        # precision = (# correct nonzeros in w^hat)/(num zeros in w^hat)
        precision = agreement.sum()/nonzero_reg_weights.sum()
        # recall = (number of correct nonzeros in w^hat)/k
        recall = agreement.sum()/self.k
        return precision, recall


class RegularizationPath:
    def __init__(self):
        pass

    def determine_smallest_lambda(self):
        pass
