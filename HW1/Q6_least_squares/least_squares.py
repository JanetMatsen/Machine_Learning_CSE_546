import numpy as np
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

    def solve_coeffs(self):

        D = self.X.shape[1]  # d = number of features/columns
        # find lambda*I_D + X^T*X
        piece_to_invert = sp.identity(D)*self.lam + self.X.T.dot(self.X)

        #inverted_piece = piece_to_invert.linalg.inv()
        inverted_piece = splin.inv(piece_to_invert)

        solution = inverted_piece.dot(self.X.T)
        solution = solution.dot(self.y)

        self.solution_coeffs = solution
        self.y_preds = self.X.dot(self.solution_coeffs).toarray()[:, 0]
