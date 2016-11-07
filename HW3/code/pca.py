import matplotlib.pyplot as plt
import numpy as np
import sys
import time

class Pca:
    def __init__(self, X, dimensions):
       self.X = X
       self.N, self.d = X.shape
       # dimensions = "dimensions which best reconstruct the data"
       self.dimensions = dimensions

    def calc_sigma(self):
        sigma = np.zeros(shape=(self.d, self.d))
        num_done = 0
        print("Iterate over x_i in X to get Sigma: {}".format(
            time.asctime(time.localtime(time.time()))))
        for i in range(self.N):
            xi = self.X[i,:]
            #print("xi for i={}: {}".format(i, xi))
            dot_prod = xi.T.dot(xi)
            #print("dot shape: {}".format(dot_prod.shape))
            #print("dot: {}".format(dot_prod))
            sigma = np.add(sigma, dot_prod)
            num_done += 1
            if num_done%100 == 0:
                sys.stdout.write(".")
        self.sigma = sigma/1./self.N
        print("Done iterating to get Sigma: {}".format(
            time.asctime(time.localtime(time.time()))))

    def calc_eigen_stuff(self):
        self.calc_sigma()
        self.eigenvals, self.eigenvects = np.linalg.eig(self.sigma)
        # note that the eigenvectors are rows:
        # The normalized (unit “length”) eigenvectors, such that the
        # column v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
        # https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.eig.html

    def top_eigenvalue_indices(self):
        abs_eigenvalues = np.abs(self.eigenvals)
        return abs_eigenvalues.argsort()[-self.dimensions:][::-1]

    def top_eigenvectors(self):
        best_indices = self.top_eigenvalue_indices()
        return self.eigenvects[:, best_indices]

    def sum_of_top_eigenvectors(self):
        best_eigenvectors = self.top_eigenvectors()
        vectors_summed = np.sum(best_eigenvectors, axis=1)
        assert vectors_summed.shape == (self.d, )
        return vectors_summed

    def fractional_reconstruction_error(self):
        # total reconstruction error is measured as the average squared
        # length of the corresponding red lines.
        # http://stats.stackexchange.com/questions/2691/making-sense-of-principal-component-analysis-eigenvectors-eigenvalues
        pass

    def plot_fractional_reconstruction_error(self):
        # TODO: incomplete
        x = None
        y= None
        fig, ax = plt.subplots(1, 1, figsize=(3.5,3))
        plt.plot(self.results[x], self.results[y],
                     linestyle='--', marker='o', color='b')
        plt.legend(loc = 'best')
        plt.xlabel("k")
        plt.ylabel("fractional reconstrction error")
        plt.title("Q-1-2-2: Frac. Reconstruction Error")
        plt.tight_layout()
        return fig





