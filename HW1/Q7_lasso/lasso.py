import numpy as np
import scipy.sparse as sp
# analyze my solution by comparing objective functions
from sklearn import linear_model
import pandas as pd


class SparseLasso:
    def __init__(self, X, y, lam, w=None, w0=0, delta=0.01,
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

        self.X = sp.csc_matrix(X)
        self.N, self.d = self.X.shape
        self.y = sp.csc_matrix([y]).T
        assert self.y.shape == (self.N, 1)

        if w is None:
            self.w = sp.csc_matrix(np.ones(self.d)).T
        elif type(w) == np.ndarray:
            self.w = sp.csc_matrix([w]).T
        else:
            assert False, "w is not None or a numpy array."
        assert self.w.shape == (self.d ,1), \
            "shape of w is {}".format(self.w.shape)
        self.w0 = w0
        self.lam = lam
        self.delta = delta
        self.verbose = verbose
        self.max_iter = max_iter

    def w0_array(self):
        return np.ones((self.N, 1))*self.w0

    def sse(self):
        # SSE is sum of residuals squared
        error_v = self.X.dot(self.w) + self.w0_array() - self.y
        return self.extract_scalar(error_v.T.dot(error_v))

    def rmse(self):
        # RMSE = root mean square error
        # the square root of the variance of the residuals
        mse = self.sse()/self.N  # /N for the M in RMSE
        return mse**0.5 # **0.5 for the R in the RMSE

    def objective(self):
        return self.sse() + self.lam * self.l1_norm(self.w)

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
        # todo: calculate the a_k array here because it's the same for
        # each call to self.step()
        for s in range(0, self.max_iter):
            old_objective = self.objective()
            old_w = self.w.copy()
            self.step()
            assert not self.has_increased_significantly(
                    old_objective, self.objective()), \
                "objective: {} --> {}".format(old_objective, self.objective())
            if abs(old_w - self.w).max() < self.delta:
                break

        if self.verbose:
            print(self.objective())
            print(self.w)

    @staticmethod
    def has_increased_significantly(old, new, sig_fig=10**(-4)):
       """
       Return if new is larger than old in the `sig_fig` significant digit.
       """
       return(new > old and np.log10(1.-old/new) > -sig_fig)

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
    skl_lasso = SparseLasso(X, y, lam, w0=0, verbose = False)
    skl_lasso.w = sp.csc_matrix(clf.coef_).T
    skl_lasso.w0 = clf.intercept_

    skl_objective_fun_value = skl_lasso.objective()

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
    w = np.zeros(d, dtype=float)
    w[0] = 10.
    w[1] = -10
    w[2] = 10
    w[3] = -10
    w[4] = 10
    assert w.shape == (d, )

    # generate error
    e = np.random.normal(0, sigma, N)
    assert e.shape == (N, )

    # generate noisy Y
    Y = X.dot(w) + w0 + e
    Y.reshape(N, )

    assert X.shape == (N, d)
    assert Y.shape == (N, )
    assert w.shape == (d, )
    return X, Y, w


class RegularizationPath:
    def __init__(self, X, y, lam_max, frac_decrease, steps, delta):
        self.X = X
        self.y = y
        self.N, self.d = self.X.shape
        self.lam_max = lam_max
        self.frac_decrease = frac_decrease
        self.steps = steps
        self.delta = delta

    def analyze_lam(self, lam, w):
        sl = SparseLasso(self.X, self.y, lam, w=w, delta=self.delta)
        sl.run()
        assert sl.w.shape == (self.d, 1) # check before we slice out
        return sl.w.toarray()[:,0], sl.w0

    def walk_path(self):
        # protect the first value of lambda.
        lam = self.lam_max/self.frac_decrease
        w_prev = None

        # initialize a dataframe to store results in
        results = pd.DataFrame()
        for c in range(0, self.steps):
            print("Loop {}: solving weights.".format(c+1))
            lam = lam*self.frac_decrease

            w, w0 = self.analyze_lam(lam, w=w_prev)

            one_val = pd.DataFrame({"lam":[lam],
                                    "weights":[w],
                                    "w0": [w0]})
            results = pd.concat([results, one_val])
            w_prev = w

        self.results_df = results


class SyntheticDataRegPath():
    def __init__(self, N, d, sigma, lam_max, frac_decrease, k=5, steps=10):
        self.N = N
        self.d = d
        self.sigma = sigma
        self.k = k
        self.init_lam = lam_max
        self.frac_decrease = frac_decrease
        X, y, true_weights = generate_random_data(N=N, d=d,
                                                  sigma=sigma, k=k)
        self.X = X
        self.y = y
        self.true_weights = true_weights
        self.lam_max = lam_max
        reg_path = RegularizationPath(X=self.X, y=self.y,
                                      lam_max=self.lam_max,
                                      frac_decrease=self.frac_decrease,
                                      steps=steps)
        reg_path.walk_path()
        self.results_df = reg_path.results_df

    def analyze_path(self):
        # update self.reg_path_results dataframe
        self.results_df['precision'] = \
            self.results_df['weights'].apply(self.calc_precision)
        self.results_df['recall'] = \
            self.results_df['weights'].apply(self.calc_recall)

    def weight_agreement(self, regression_weights, z):
        # True array for regression weight ~ 0:
        true_weights_array = self.true_weights.reshape(1, self.d)
        abs_weights = np.absolute(true_weights_array)
        nonzero_weights = abs_weights > z

        reg_weight_array = regression_weights.reshape(1, self.d)
        abs_reg_weights = np.absolute(reg_weight_array)
        nonzero_reg_weights = abs_reg_weights > z
        disagreement = np.bitwise_xor(nonzero_weights, nonzero_reg_weights)

        return (disagreement, nonzero_weights, nonzero_reg_weights)

    def calc_precision(self, regression_weights, z=0.001):
        agreement, nonzero_weights, nonzero_reg_weights = \
            self.weight_agreement(regression_weights, z)
        # precision = (# correct nonzeros in w^hat)/(num zeros in w^hat)
        return agreement.sum() / nonzero_reg_weights.sum()

    def calc_recall(self, regression_weights, z=0.001):
        agreement, nonzero_weights, nonzero_reg_weights = \
            self.weight_agreement(regression_weights, z)
        # recall = (number of correct nonzeros in w^hat)/k
        return agreement.sum() / self.k


class RegularizationPathTrainTest:
    def __init__(self, X_train, y_train, lam_max, X_val, y_val,
                 steps=10, frac_decrease=0.1, delta=0.01):
        self.X_train = X_train
        self.y_train = y_train
        self.lam_max = lam_max
        self.X_val = X_val
        self.y_val = y_val
        self.steps = steps
        self.frac_decrease = frac_decrease
        reg_path = RegularizationPath(X=self.X_train, y=self.y_train,
                                      lam_max=self.lam_max,
                                      frac_decrease=self.frac_decrease,
                                      steps=self.steps,
                                      delta=delta)
        reg_path.walk_path()
        self.results_df = reg_path.results_df

    def analyze_path(self):
        self.results_df['RMSE (training)'] = \
            self.results_df.apply(
                lambda x: self.rmse_train(x['weights'], x['w0']), axis=1)
        self.results_df['RMSE (validation)'] = \
            self.results_df.apply(
                lambda x: self.rmse_test(x['weights'], x['w0']), axis=1)

        self.results_df['# nonzero coefficients'] = \
            self.results_df['weights'].apply(self.num_nonzero_coefs)

    def calc_rmse(self, X, w, w0, y):
        # re-use the formula implemented in SparseLasso.
        # put in a random lam b/c it isn't used.
        sl = SparseLasso(X, y, lam=0, verbose=False)
        # store solutions in my Lasso class so I can look @ obj fun
        sl.w = w.reshape(w.shape[0], 1)
        sl.w0 = w0
        return sl.rmse()

    def rmse_train(self, w, w0):
        return self.calc_rmse(X=self.X_train, w=w, w0=w0, y=self.y_train)

    def rmse_test(self, w, w0):
        return self.calc_rmse(X=self.X_val, w=w, w0=w0, y=self.y_val)

    def num_nonzero_coefs(self, w, z=0.001):
        nonzero_weights = np.absolute(w) > z
        return nonzero_weights.sum()

