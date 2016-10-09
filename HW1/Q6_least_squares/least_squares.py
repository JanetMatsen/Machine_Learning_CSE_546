import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splin

def ridge(X, y, lam):

    assert type(X) == sp.csc_matrix or type(X) == sp.csr_matrix
    assert type(lam*1.0) == float
    #assert type(y) == np.ndarray
    assert type(y) == sp.csr_matrix or type(y) == sp.csc_matrix


    D = X.shape[1]  # d = number of features/columns
    # find lambda*I_D + X^T*X
    piece_to_invert = sp.identity(D)*lam + X.T.dot(X)

    #inverted_piece = piece_to_invert.linalg.inv()
    inverted_piece = splin.inv(piece_to_invert)

    solution = inverted_piece.dot(X.T)
    solution = solution.dot(y)

    return solution

def analyze_results(X, y, cutoff):

    import pdb; pdb.set_trace()
    assert type(X) == sp.csc_matrix or type(X) == sp.csr_matrix
    assert type(y) == sp.csr_matrix or type(y) == sp.csc_matrix

    y_predictions = X.dot(sol).toarray()[:, 0]
    truth = y.toarray()[:,0]

    # categorize the info
    ys_for_3s = y_predictions[truth ==1]
    ys_for_other_numbers = y_predictions[truth == 0]

    true_positives =
