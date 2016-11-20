import numpy as np
from scipy.spatial.distance import cdist

class KMeans:
    def __init__(self, k, train_X, train_y, eigenvectors=None,
                 max_iter = 10,
                 test_X=None, test_y=None):
        self.k = k
        self.X = train_X
        self.y = train_y
        self.test_X = test_X
        self.test_y = test_y
        # each row is a center.
        self.eigenvectors = eigenvectors

        self.num_iter = 0
        self.max_iter = max_iter

        # intitialize centers by drawing from X without replacement.
        self.centers = self.choose_random_points(train_X, k)

        # mark the model as converged once it is
        self.converged = False

    @staticmethod
    def choose_random_points(X, n):
        indices = np.random.choice(X.shape[0], n, replace=False)
        return X[indices]

    def recenter_each_center(self):
        """
        For each center, find the points that are assigned to it, and
        """
        for c in range(self.k):
            old_center = self.centers[c]
            print("recenter cluster {}".format(c))
            points = self.X[self.assignments == c]
            center = np.sum(points, axis=0)/points.shape[0]
            self.centers[c] = center
            print("old center: {}".format(old_center))
            print("new center: {}".format(center))

    def assign_points(self, X=None):
        if X is None:
            X = self.X
        distances = cdist(X, self.centers)
        self.assignments = np.argmin(distances, axis=1)

    def run(self):
        self.assign_points()

        while (self.converged == False) and (self.num_iter < self.max_iter):
            self.num_iter += 1

            # Store old state to assess convergence.
            old_centers = self.centers.copy()
            old_assignments = self.assignments.copy()

            # Update
            self.assign_points()
            # After re-centering the cluster centers, re-assign the points to
            # the nearest center.
            self.recenter_each_center()

            # Test for convergence
            centers_converged = self.test_convergence_of_arrays(
                old_centers, self.centers)
            assignments_converged = self.test_convergence_of_arrays(
                old_assignments, self.assignments)
            if centers_converged and assignments_converged:
                print("Both the centers and assignments converged after {} "
                      "iterations.".format(self.num_iter))
                self.converged = True

    def test_convergence_of_arrays(self, before, after):
        # Start by testing for identity.  Later can downgrade to small percent difference.
        if np.sum(np.abs(before - after)) < 0.001:
            return True

    def majority_label_for_each_center(self):
        """
        Find the center of the points assigned to this cluster
        """
        pass

    def classify(self):
        # find label for each center, and classify all points.
        pass

    def num_assignments_per_cluster(self):
        pass

    def squared_reconstruction_error(self):
        pass

    def visualize_single_image(self):
        # todo: put in MNIST_helper file?
        pass

    def visualize_center(self):
        # First transform back into image space.
        pass

    def loss_01(self, set="train"):
        if set == "train":
            pass
        if set == "test":
            pass
        pass

