import matplotlib.pyplot as plt
import numpy as np
import sys
import time

import pandas as pd


class Pca:
    def __init__(self, X, dimensions, center=True, verbose=False):
        self.X = X
        self.X_center = X.sum(axis=0)/X.shape[0]
        self.N, self.d = X.shape
        # dimensions = "dimensions which best reconstruct the data"
        self.dimensions = dimensions
        self.fractional_reconstruction_df = None
        self.center = center
        self.verbose = verbose

    def calc_sigma(self):
        """
        sigma = covariance matrix.
        """
        sigma = np.zeros(shape=(self.d, self.d))
        num_done = 0
        if self.center:
            X = self.X - self.X_center
        else:
            X = self.X

        if self.verbose:
            print("Iterate over x_i in X to get Sigma: {}".format(
                time.asctime(time.localtime(time.time()))))
        # Build up the covariance matrix.
        for i in range(self.N):
            xi = np.array([X[i,:]])
            if self.verbose:
                print("xi: {}".format(xi))
            #print("xi for i={}: {}".format(i, xi))
            dot_prod = xi.T.dot(xi)
            if self.verbose:
                print("dot_prod: {}".format(dot_prod))
            #print("dot shape: {}".format(dot_prod.shape))
            #print("dot: {}".format(dot_prod))
            sigma = np.add(sigma, dot_prod)
            if self.verbose:
                print("sigma: \n{}".format(sigma))
            num_done += 1
            if num_done%100 == 0:
                sys.stdout.write(".")
        if self.verbose:
            print("sigma before dividing by N = {}".format(self.N))
            print(sigma)
        self.sigma = sigma*1./self.N
        if self.verbose:
            print("sigma after dividing by N = {}".format(self.N))
            print(self.sigma)
        if self.verbose:
            print("Done iterating to get Sigma: {}".format(
                time.asctime(time.localtime(time.time()))))
            print("Sigma: \n{}".format(self.sigma))

    def calc_eigen_stuff(self):
        self.calc_sigma()
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eigh.html
        # The eigenvalues in ascending order
        # The column v[:, i] is the normalized eigenvector corresponding to
        # the eigenvalue w[i].
        self.eigenvals, self.eigenvects = np.linalg.eigh(self.sigma)
        # Order them the with the big ones first other way:

        if self.verbose:
            print("before: \n {}".format(self.eigenvals))
        self.eigenvals = self.eigenvals[::-1]
        if self.verbose:
            print("after: \n{}".format(self.eigenvals))

        # Order the matrix the same way
        if self.verbose:
            print("before: \n{}".format(self.eigenvects))
        # flip the matrix left to right.
        self.eigenvects = np.fliplr(self.eigenvects)
        if self.verbose:
            print("after: \n{}".format(self.eigenvects))

    def sum_of_top_eigenvalues(self):
        return np.sum(self.eigenvals[0:self.dimensions])

    def fractional_reconstruction_error(self):
        # total reconstruction error is measured as the average squared
        # length of the corresponding red lines.
        # http://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
        eigenvalues = self.eigenvals
        summary = pd.DataFrame()
        vector_sum = np.sum(eigenvalues)
        if self.verbose:
            print('vector sum: {}'.format(vector_sum))
        for i in range(1, len(eigenvalues)):
            numerator_sum = np.sum(eigenvalues[0:i])
            if self.verbose:
                print(numerator_sum)
            row = {'k':[i],
                   'fractional reconstruction': [1-numerator_sum/vector_sum]}
            row = pd.DataFrame(row)
            summary = pd.concat([summary, row])
        self.fractional_reconstruction_df = summary

    def plot_fractional_reconstruction_error(self, start=None, stop=None,
                                             title=None):
        if self.fractional_reconstruction_df is None:
            self.fractional_reconstruction_error()
        if start is None:
            self.start = 0
        if stop is None:
            self.stop = self.dimensions
        plot_data = self.fractional_reconstruction_df[
            (self.fractional_reconstruction_df.k >= start) &
            (self.fractional_reconstruction_df.k <= stop)]
        x = plot_data.k
        y = plot_data['fractional reconstruction']
        fig, ax = plt.subplots(1, 1, figsize=(3.5,3))
        plt.plot(x, y, linestyle='--', marker='o', color='b')
        plt.legend(loc = 'best')
        plt.xlabel("k")
        plt.ylabel("fractional reconstrction error")
        if title is None:
            title = "Q-1-2-2"
        plt.title(title)
        plt.tight_layout()
        fig.savefig("../figures/Q-1-2-2.pdf")
        return fig

    def save_sigma(self, filename):
        np.save(filename + '.npy', self.sigma)


def make_image(data, path=None):
    plt.figure(figsize=(1,1))
    p=plt.imshow(data.reshape(28, 28), origin='upper')
    p.set_cmap('gray_r')
    plt.axis('off')
    if path is not None:
        plt.savefig(path)
        plt.close()

