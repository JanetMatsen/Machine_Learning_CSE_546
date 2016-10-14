import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splin


class Ridge:
    def __init__(self, X, y, lam):

        assert type(X) == sp.csc_matrix or type(X) == sp.csr_matrix
        assert type(lam*1.0) == float
        assert type(y) == sp.csr_matrix or type(y) == sp.csc_matrix

        self.X = X
        self.N = X.shape[0]
        self.y = y
        self.lam = lam
        self.w = None

    def solve(self):

        d = self.X.shape[1]  # d = number of features/columns
        # find lambda*I_D + X^T*X
        piece_to_invert = sp.identity(d)*self.lam + self.X.T.dot(self.X)

        inverted_piece = splin.inv(piece_to_invert)

        solution = inverted_piece.dot(self.X.T)
        solution = solution.dot(self.y)

        self.w = solution

    def sse(self):
        # sse = RSS
        error_v = self.X.dot(self.w) - self.y
        return self.extract_scalar(error_v.T.dot(error_v))

    def rmse(self):
        return(self.sse()/self.N)**0.5

    @staticmethod
    def extract_scalar(m):
        assert(m.shape == (1, 1))
        return m[0, 0]


class RidgeRegularizationPath:
    def __init__(self, train_X, train_y, lam_max, frac_decrease, steps,
                 val_X, val_y):
        self.train_X = train_X
        self.train_y = train_y
        self.train_N, self.train_d = train_X.shape
        self.lam_max = lam_max
        self.frac_decrease = frac_decrease
        self.steps = steps
        self.val_X = val_X
        self.val_y = val_y

    def train_with_lam(self, lam):
        rr = Ridge(self.train_X, self.train_y, lam=lam)
        rr.solve()
        sse_train = rr.sse()
        # replace the y values with the validation y and get the val sss
        rr.X = self.val_X
        rr.y = self.val_y
        sse_val = rr.sse()
        assert rr.w.shape == (self.train_d, 1) # check before we slice out
        return rr.w.toarray()[:,0], sse_train, sse_val

    def walk_path(self):
        # protect the first value of lambda.
        lam = self.lam_max/self.frac_decrease

        # initialize a dataframe to store results in
        results = pd.DataFrame()
        for c in range(0, self.steps):
            lam = lam*self.frac_decrease
            print("Loop {}: solving weights.  Lambda = {}".format(c+1, lam))

            w, sse_train, sse_val = self.train_with_lam(lam)

            one_val = pd.DataFrame({"lam":[lam],
                                    "weights":[w],
                                    "SSE (training)": [sse_train],
                                    "SSE (validaton)": [sse_val]})
            results = pd.concat([results, one_val])

        self.results_df = results
