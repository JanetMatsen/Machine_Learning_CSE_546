import numpy as np
from scipy.spatial.distance import pdist

class Fourier:
    def __init__(self, X, k=60000, sigma=None):
        self.X = X
        self.k = k
        self.N = X.shape[0]
        self.d = k
        if sigma is None:
            sample_size = min(self.N, max(1000, int(X.shape[0]/10)))
            self.set_sigma(sample_size)
        self.generate_feature_vectors()
        pass

    def set_sigma(self, sample_size):
        # About 2000 is good.
        print('determine kernel bandwidth using {} points.'.format(sample_size))

        X_sample_indices = np.random.choice(self.N, sample_size, replace=False)
        X_sample = self.X[X_sample_indices]
        assert X_sample is not None

        pairwise_distances = pdist(X_sample)
        # TODO: try mean instead of median.
        median_dist = np.median(pairwise_distances)
        print("median distance for {} samples from N: {}".format(
            sample_size, median_dist))
        self.sigma = median_dist

    def generate_feature_vectors(self):
        """
        independently sample every coordinate for every vector from a
        standard normal distribution (with unit variance).
        """
        n = self.X.shape[1]
        self.vectors = np.random.randn(n, self.k)

    def transform(self, X):
        dot_prod = X.dot(self.vectors)
        return np.exp(dot_prod/self.sigma)

    def info(self):
        return {'sigma':[self.sigma]}


class RBFKernel:
    def __init__(self, X, sigma=None):
        self.X = X
        self.N = X.shape[0]
        self.d = X.shape[0] # N by d --> N by N
        if sigma is None:
            sample_size = min(self.N, max(1000, int(X.shape[0]/10)))
            self.set_sigma(sample_size)
        else:
            self.sigma = sigma
        self.name = 'radial basis function'

    def set_sigma(self, sample_size):
        """
        setting σ is often done with the ’median trick’, which is the median
        of the pairwise distances (between the x’s) in your dataset.

        Randomly grab a few pairs of points and estimate the mean distance
        between a random pair of points rather than the median).

        Then multiplicatively cut it down by some factor (maybe 2, 4, 8, ...
        depending on the problem).

        :param sample_size: number of samples to chose the median based on
        :return:
        """
        print('determine RBF kernel bandwidth using {} points.'.format(
            sample_size))

        X_sample_indices = np.random.choice(self.N, sample_size, replace=False)
        X_sample = self.X[X_sample_indices]
        assert X_sample is not None

        pairwise_distances = pdist(X_sample)
        # TODO: try mean instead of median.
        median_dist = np.median(pairwise_distances)
        print("median distance for {} samples from N: {}".format(
            sample_size, median_dist))
        self.sigma = median_dist

    def transform_vector(self, xi):
        """
        transforms a single point
        """
        dist = np.linalg.norm(self.X - xi, axis=1)
        dist_squared = np.multiply(dist, dist)
        return np.exp(dist_squared/(-2.)/self.sigma**2)

    def transform(self, X):
        """
        Transforms a matrix, which isn't necessarily self.X
        """
        # TODO: could apply only to the first 1/2 of point (I think)
        return np.apply_along_axis(func1d=self.transform_vector, axis=1, arr=X)

    def info(self):
        return {'sigma':[self.sigma]}


class NoKernel:
    def __init__(self, X):
        self.X = X
        self.N, self.d = X.shape

    def transform(self, X):
        # The point of this kernel is to not transform anything at all.
        return X

    def info(self):
        return {"null kernel":[None]}
