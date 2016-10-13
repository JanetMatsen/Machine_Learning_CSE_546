import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splin

class Ridge:
    def __init__(self, X, y, lam):

        assert type(X) == sp.csc_matrix or type(X) == sp.csr_matrix
        assert type(lam*1.0) == float
        #assert type(y) == np.ndarray
        assert type(y) == sp.csr_matrix or type(y) == sp.csc_matrix

        self.X = X
        self.y = y
        self.lam = lam
        #self.cutoff = cutoff

    def solve(self):

        D = self.X.shape[1]  # d = number of features/columns
        # find lambda*I_D + X^T*X
        piece_to_invert = sp.identity(D)*self.lam + self.X.T.dot(self.X)

        #inverted_piece = piece_to_invert.linalg.inv()
        inverted_piece = splin.inv(piece_to_invert)

        solution = inverted_piece.dot(self.X.T)
        solution = solution.dot(self.y)

        self.w = solution
        self.y_preds = self.X.dot(self.w).toarray()[:, 0]

    def calc_square_loss(self):
        differences = self.y - self.y_preds
        self.square_loss = None


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

    def train_with_lam(self, lam, w):
        rr = Ridge(self.train_X, self.train_y, lam=lam)
        rr.solve()
        assert rr.w.shape == (self.train_d, 1) # check before we slice out
        return rr.w.toarray()[:,0]

    def walk_path(self):
        # protect the first value of lambda.
        lam = self.lam_max/self.frac_decrease
        w_prev = None

        # initialize a dataframe to store results in
        results = pd.DataFrame()
        for c in range(0, self.steps):
            print("Loop {}: solving weights.".format(c+1))
            lam = lam*self.frac_decrease

            w = self.train_with_lam(lam, w=w_prev)

            one_val = pd.DataFrame({"lam":[lam],
                                    "weights":[w]})
            results = pd.concat([results, one_val])
            w_prev = w

        self.results_df = results
