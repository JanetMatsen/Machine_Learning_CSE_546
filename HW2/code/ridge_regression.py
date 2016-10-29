import numpy as np
import pandas as pd
import scipy.sparse as sp
import scipy.sparse.linalg as splin

import time;


from classification_base import ClassificationBase

class RidgeMulti(ClassificationBase):
    """
    Train multiple ridge models.
    """
    def __init__(self, X, y, lam, W=None, verbose=False, sparse=True):
        super(RidgeMulti, self).__init__(X=X, y=y, W=W, sparse=sparse)
        self.sparse = sparse
        if self.sparse:
            assert lam != 0, "can't invert the big stuff with lambda = 0."
            self.X = sp.csc_matrix(self.X)
            self.Y = sp.csc_matrix(self.Y)
        self.lam = lam
        self.W = None # don't want to have W before solving!
        self.matrix_work = None
        self.verbose = verbose

    def get_weights(self):
        if self.sparse:
            return self.W.toarray()
        if not self.sparse:
            return self.W

    def apply_weights(self):
        # Check that weights are the right dims.

        # Apply weights
        if self.sparse:
            assert type(self.W) == sp.csc_matrix, \
                "type of W is {}".format(type(self.W))
            assert type(self.X) == sp.csc_matrix, \
                "type of W is {}".format(type(self.X))
            prod = self.X.dot(self.W)
            if type(prod) == sp.csc_matrix:
                return prod.toarray()
            else:
                return prod
        else:
            return self.X.dot(self.W)

    def optimize(self):
        # When solving multiclass, (X^TX + lambdaI)-1X^T is shared
        # solve it once and share it with all the regressors.
        # find lambda*I_D + X^T*X
        if self.verbose: print("optimize: multiply matrices before inversion.")

        # Get (X^TX + lambdaI)
        if self.sparse:
            piece_to_invert = sp.csc_matrix(sp.identity(self.d)*self.lam) + \
                              self.X.T.dot(self.X)
        else:
            piece_to_invert = np.identity(self.d)*self.lam + self.X.T.dot(self.X)
        assert piece_to_invert.shape == (self.d, self.d)

        # Invert (X^TX + lambdaI)
        if self.verbose:
            print("invert matrix:")
            print("time: {}".format(time.asctime(time.localtime(time.time()))))
        if self.sparse:
            inverted_piece = splin.inv(piece_to_invert)
        else:
            inverted_piece = np.linalg.inv(piece_to_invert)

        # Dot with X^T
        if self.verbose:
            print("time: {}".format(time.asctime(time.localtime(time.time()))))
            print("dot with X^T:")
        self.matrix_work = inverted_piece.dot(self.X.T)
        assert self.matrix_work.shape == (self.d, self.N)

        if self.verbose:
            print("train the {} classifiers:".format(self.C))
        # Train C classifiers.
        self.W = self.matrix_work.dot(self.Y)
        if self.verbose:
            print("done generating weights.")
        assert self.W.shape == (self.d, self.C)
        return self.W

    def predict(self):
        if self.verbose:
            print("predict:")
        if self.W is None:
            self.optimize()

        Yhat = self.apply_weights()
        assert type(Yhat) == np.ndarray
        classes = np.argmax(Yhat, axis=1)
        if self.sparse:
            yhat = np.multiply(self.Y.toarray(), Yhat)
        else:
            yhat = np.multiply(self.Y, Yhat)
        # collapse it into an Nx1 array:
        self.yhat = np.amax(yhat, axis=1)
        return classes

    def run(self):
        self.predict()
        self.results = pd.DataFrame(self.results_row())

    def loss_01(self):
        return self.pred_to_01_loss(self.predict())

    def results_row(self):
        """
        Return a dictionary that can be put into a Pandas DataFrame.
        """
        results_row = super(RidgeMulti, self).results_row()

        # append on Ridge regression-specific results
        more_details = {
            "lambda":[self.lam],
            "training SSE":[self.sse()],
            "training RMSE":[self.rmse()],
            }
        results_row.update(more_details)
        return results_row

    def sse(self):
        """
        Calculate the sum of squared errors.

        In class on 10/26, Sham coached us to include errors for all
        classifications in our RMSE (and thus SSE) calculations.
        For y = [0, 1], Y=[[0, 1], [1, 0]], Yhat = [[0.01, 0.95], [0.99, 0.03]],
        SSE = sum(0.01**2 + 0.05**2 + 0.01**2 + 0.03**2) = RSS
        Note: this would not be equivalent to the binary classifier, which
        would only sum (0.05**2 + 0.03**2)

        My formula before only used the errors for the correct class:
            error = self.apply_weights() - self.Y
            error = np.multiply(error, self.Y)
            error = np.amax(np.abs(error), axis=1)
            return error.T.dot(error)

        :return: sum of squared errors for all classes for each point (float)
        """
        if self.sparse:
            error = self.apply_weights() - self.Y.toarray()
            assert type(error) == np.ndarray
        else:
            error = self.apply_weights() - self.Y
        return np.multiply(error, error).sum()

    def rmse(self):
        """
        For the binary classifier, RMSE = (SSE/N)**0.5.
        For the multiclass one, SSE is counting errors for all classifiers.
        We could use (self.sse()/self.N/self.C)**0.5 to make the RMSE
        calcs more similar between the binary and multi-class classifiers,
        but they still are not the same, so I won't.

        :return: RMSE (float)
        """

        return(self.sse()/self.N)**0.5


class RidgeBinary(ClassificationBase):
    """
    Train *one* ridge model.
    """
    def __init__(self, X, y, lam, w=None):

        self.X = X
        self.N, self.d = X.shape
        self.y = y
        self.lam = lam
        if w is None:
            self.w = np.zeros(self.d)
        assert self.w.shape == (self.d, )
        self.threshold = None

    def get_weights(self):
        return self.w

    def apply_weights(self):
        return self.X.dot(self.w)

    def run(self):

        # find lambda*I_D + X^T*X
        piece_to_invert = np.identity(self.d)*self.lam + self.X.T.dot(self.X)

        inverted_piece = np.linalg.inv(piece_to_invert)

        solution = inverted_piece.dot(self.X.T)
        solution = solution.dot(self.y)

        solution = np.squeeze(np.asarray(solution))
        assert solution.shape == (self.d, )
        self.w = solution
        self.results = pd.DataFrame(self.results_row())

    def predict(self, threshold):
        # TODO: having a default cutoff is a terrible idea!
        Yhat = self.X.dot(self.w)
        classes = np.zeros(self.N)
        classes[Yhat > threshold] = 1
        return classes

    def loss_01(self, threshold=None):
        if threshold is None:
            threshold=0.5
            print("WARNING: 0/1 loss is calculated for threshold=0.5, which "
                  "is very likely to be a poor choice!!")
        return self.pred_to_01_loss(self.predict(threshold))

    def results_row(self):
        """
        Return a dictionary that can be put into a Pandas DataFrame.
        """
        results_row = super(RidgeBinary, self).results_row()

        # append on logistic regression-specific results
        more_details = {
            "lambda":[self.lam],
            "SSE":[self.sse()],
            "RMSE":[self.rmse()],
            }
        results_row.update(more_details)
        return results_row

    def sse(self):
        # sse = RSS
        error = self.apply_weights() - self.y
        return error.T.dot(error)

    def rmse(self):
        return(self.sse()/self.N)**0.5



class RidgeRegularizationPath:
    """ DEPRECATED """
    # TODO: refactor so it uses HyperparameterSweep class
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
        rr = RidgeBinary(self.train_X, self.train_y, lam=lam)
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
