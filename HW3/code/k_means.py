from collections import Counter
import numpy as np
import pandas as pd
from scipy.stats import mode as scipy_mode
from scipy.spatial.distance import cdist
import subprocess
import sys

from pca import Pca
import matplotlib.pyplot as plt

class KMeans:
    def __init__(self, k, train_X, train_y, pca_obj,
                 max_iter = 10,
                 test_X=None, test_y=None,
                 verbose=False):
        self.k = k
        self.X = train_X
        self.N, self.d = train_X.shape
        self.C = len(set(train_y)) # number of classes based on y
        self.y = train_y
        self.test_X = test_X
        self.test_y = test_y
        # each row is a center.
        # Go back to pickled PCA object to get them back into image space
        self.pca = pca_obj

        self.num_iter = 0
        self.max_iter = max_iter

        # intitialize centers by drawing from X without replacement.
        self.center_coordinates, self.cluster_labels = \
            self.choose_random_points(train_X, train_y, k)

        # mark the model as converged once it is
        self.converged = False
        self.verbose = verbose

        # model characteristics
        self.assignments = None # cluster assignment.  Does not know about labels.
        self.predictions = None # label assignment.  Does not know about cluster.

        self.results_df = None
        # todo: rename
        self.results_df_cluster_assignment_counts = None


    @staticmethod
    def choose_random_points(X, y, n):
        assert X.shape[0] == y.shape[0], \
            'must sample from arrays with same length'
        indices = np.random.choice(X.shape[0], n, replace=False)
        return X[indices], y[indices]

    def recenter_each_center(self):
        """
        For each center, find the points that are assigned to it, and
        """
        total_points_assigned = 0 # will use to check

        for c in range(self.k):
            old_center = self.center_coordinates[c]
            if self.verbose:
                print("recenter cluster {}".format(c))
            if np.isnan(old_center).any():
                import pdb; pdb.set_trace()
            points = self.X[self.assignments == c]
            center = np.sum(points, axis=0)/points.shape[0]
            self.center_coordinates[c] = center
            if self.verbose:
                print("old center: {}".format(old_center))
                print("new center: {}".format(center))

            total_points_assigned += points.shape[0]

        assert total_points_assigned == self.N

    def set_point_assignments(self, X=None):
        """
        Assign points to a cluster.
        Blind to the label eventually associated with that cluster.
        :return:
        """
        if np.isnan(self.center_coordinates).any():
            import pdb; pdb.set_trace()
        if X is None:
            X = self.X
        distances = cdist(X, self.center_coordinates)
        self.assignments = np.nanargmin(distances, axis=1)

    def run(self):
        # assign points
        self.set_point_assignments()

        while (self.converged == False) and (self.num_iter < self.max_iter):
            #print("--- begin epoch {} ---".format(self.num_iter))
            sys.stdout.write(".") # one dot per pass

            self.num_iter += 1

            # Store old state to assess convergence.
            old_centers = self.center_coordinates.copy()
            old_assignments = self.assignments.copy()

            # Update
            self.recenter_each_center()
            # After re-centering the cluster centers, re-assign the points to
            # the nearest center.
            self.set_point_assignments()

            # Test for convergence
            centers_converged = self.test_convergence_of_arrays(
                old_centers, self.center_coordinates)
            assignments_converged = self.test_convergence_of_arrays(
                old_assignments, self.assignments)
            if centers_converged and assignments_converged:
                print("")
                print("Both the centers and assignments converged after {} "
                      "iterations.".format(self.num_iter))
                self.converged = True

            self.set_point_assignments()
            self.set_centers_classes()
            self.set_predicted_labels()

            if self.num_iter > 1:
                # Also update the df of points assigned to each mean
                self.record_count_of_assignments_to_each_mean()

                # track fitting parameters
                self.record_fit_statistics()

                self.re_seed_empty_clusters()

    def test_convergence_of_arrays(self, before, after):
        # Start by testing for identity.  Later can downgrade to small percent difference.
        difference = np.abs(before - after)
        # TODO: make sure I like this handling
        if np.isnan(difference).any():
            return False
        if np.sum(difference)/difference.size < 1e-10:
            return True
        else:
            return False

    def set_centers_classes(self):
        """
        Find the center of the points assigned to this cluster
        """
        cluster_labels = []
        for c in range(self.k):
            labels = self.y[self.assignments == c]
            # return the mode, or the lowest # if two modes.
            mode_and_count = scipy_mode(labels)
            if len(mode_and_count[0]) == 0:
                import pdb; pdb.set_trace()
                mode = None
                cluster_labels.append(mode)
            # Not sure this is the right place to put re-seeding, but do
            # it for now
            #    print("re-seed cluster # {}, which was empty".format(c))
            #    cluster_labels.append(None)
            #    # re-seed the cluster
            #    self.center_coordinates[c] = self.sample_random_point()
            else:
                mode = mode_and_count[0][0] # picking from (mode, count)
                cluster_labels.append(mode)

            counts = Counter(labels)
            if self.verbose:
                print("majority label for center {}, with {} points: {}.  Counts "
                      "of each item: {}".format(c, len(labels), mode, counts))

        self.cluster_labels = cluster_labels

    def sample_random_point(self):
        new_center_index = np.random.choice(self.X.shape[0], 1)
        new_center = self.X[new_center_index]
        return new_center

    def re_seed_empty_clusters(self):
        # access last set of results.
        last_counts = \
            self.results_df_cluster_assignment_counts.tail(1).reset_index()
        count_colnames = [c for c in last_counts.columns
                          if type(c) is np.uint8]
        for column in last_counts[count_colnames]:
            center_index = column
            num_pts = last_counts[column][0]
            if num_pts is 0:
                print("Re-seeding center (index={}) with new point"
                      "".format(center_index))
                self.center_coordinates[center_index] = \
                    self.sample_random_point()

    def predicted_label_for_number(self, digit_index):
        assignment = self.assignments[digit_index]
        prediction = self.cluster_labels[assignment]
        return prediction

    def set_predicted_labels(self):
        """
        """
        # Assumes self.set_center_classes() has been called previously.
        classes = []
        for i in range(self.X.shape[0]):
            classes.append(self.predicted_label_for_number(i))

        self.predictions = classes

    def loss_01(self):

        y = self.y.copy()
        y = y.reshape(1, self.N)
        return self.N - np.equal(y, self.predictions).sum()

    def loss_01_normalized(self):
        return self.loss_01()/self.N

    def num_assignments_per_cluster(self):
        # assumes assign_points() has been called just prior.
        clusters = self.assignments
        counts = Counter(clusters)
        return dict(counts)

    def center_for_number(self, digit_index):
        center_number = self.assignments[digit_index]
        center_coordinate = self.center_coordinates[center_number]
        return center_coordinate

    def squared_reconstruction_error(self):
        """
        Squared reconstruction error = squared distance between the
        data point and its center.
        Sum for all N points.
        """
        squared_distance_sum = 0
        for i in range(self.X.shape[0]):
            center = self.center_for_number(i)
            dist = np.linalg.norm(self.X[i] - center)
            dist_squared = dist**2
            squared_distance_sum += dist_squared
        return squared_distance_sum

    def record_fit_statistics(self):
        # Record
        squared_reconstruction_error = self.squared_reconstruction_error()

        results = {'iteration':self.num_iter,
                   'squared reconstruction error':
                       squared_reconstruction_error,
                   '(squared reconstruction error)/N':
                        squared_reconstruction_error/self.N,
                   '0/1 loss':self.loss_01(),
                   '(0/1 loss)/N':self.loss_01_normalized()}
        result_df_row = pd.DataFrame.from_dict(results, orient='index').T
        self.results_df = pd.concat([self.results_df, result_df_row], axis=0)

    def record_count_of_assignments_to_each_mean(self):
        counts = self.num_assignments_per_cluster()
        counts['iteration'] = self.num_iter
        row_results = pd.DataFrame.from_dict(counts, orient='index').T
        self.results_df_cluster_assignment_counts = \
            pd.concat([self.results_df_cluster_assignment_counts, row_results])

    @staticmethod
    def make_image(data, path=None):
        plt.figure(figsize=(0.7,0.7))
        p=plt.imshow(data.reshape(28, 28), origin='upper',
                     interpolation='none')
        p.set_cmap('gray_r')
        plt.axis('off')
        if path is not None:
            plt.savefig(path)
            plt.close()

    def visualize_center(self, x, path=None):
        # First transform back into image space.
        # Use PCA object's methods.
        assert type(self.pca) == Pca, \
            "Need a Pca object to go back to image space."
        image_space = self.pca.transform_number_up(x, center=True)
        assert image_space.shape[0] == 784
        return self.make_image(image_space, path=path)

    def clusters_by_num_in_cluster(self):
        cts = self.results_df_cluster_assignment_counts.tail(1)
        del cts['iteration']
        cts = cts.T
        cts.sort_values(by=0, ascending=False, inplace=True)
        return cts

    def top_n_centers(self, n):
        cts = self.clusters_by_num_in_cluster()
        cts = cts[0]
        indices = cts.nlargest(n).index.values
        centers = []
        for i in indices:
            centers.append(self.center_coordinates[i])
        assert len(centers) == n, "need to give back n centers."
        return centers

    def visualize_top_n_centers(self, n=16):
        """
        Visualize the 16 centers that you learned, and display them in an
        order in that corresponds to the frequency in which they were assigned
        """
        import pdb; pdb.set_trace()
        centers_list = self.top_n_centers(n)
        assert len(centers_list) == n, "expected {} coordinates".format(n)
        paths = []
        path_base = '../figures/k_means/'
        for i, center in enumerate(centers_list):
            path = path_base + \
                   'k_{}_top_{}_centers_{}.pdf'.format(self.k, n, i)
            print('saving to {}'.format(path))

            # save image
            self.visualize_center(center, path=path)
            paths.append(path)

        # stitch together w/ shell command
        photo_paths = ' '.join(paths)
        print('photo paths: \n{}'.format(photo_paths))
        stitched_path = path_base + 'k_{}_top_{}_centers.pdf'.format(self.k, n)
        shell_stitch_command = "convert +append {} {} ".format(photo_paths, stitched_path)
        print(shell_stitch_command)
        subprocess.call(shell_stitch_command, shell=True)

    def plot_results(self, x_var, y_var, ylabel, color=None):
        if self.results_df is None:
            print("no data to plot")
            return

        if color is None:
            color = '#feb24c' # orange

        x = self.results_df[x_var]
        y = self.results_df[y_var]

        fig, ax = plt.subplots(1, 1, figsize=(4,3))

        plt.plot(x, y, linestyle='-', marker='o', markersize=4, color=color)
        plt.legend(loc = 'best')
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        plt.xlabel(x_var)
        plt.ylabel(ylabel)
        ax.set_ylim(bottom=0)
        plt.tight_layout()
        return fig

    def plot_squared_reconstruction_error(self):
        x = 'iteration'
        y = 'squared reconstruction error'
        color = '#feb24c' # orange
        return self.plot_results(x_var=x, y_var=y, ylabel=y, color=color)

    def plot_squared_reconstruction_error_normalized(self):
        x = 'iteration'
        y = '(squared reconstruction error)/N'
        color = '#feb24c' # orange
        return self.plot_results(x_var=x, y_var=y, ylabel=y, color=color)

    def plot_0_1_loss(self):
        x = 'iteration'
        y = '(0/1 loss)/N'
        color = '#31a354' # dark green
        return self.plot_results(x_var=x, y_var=y, ylabel=y, color=color)

    def plot_num_assignments_for_each_center(self):
        """
        Plot the number of assignments for each center in descending order
        """
        cts = self.clusters_by_num_in_cluster()
        c = cts[0]

        fig, ax = plt.subplots(1, 1, figsize=(8,2.2))
        plt.bar(range(c.shape[0]), c, align="center")
        plt.xlim([0,len(c)])
        plt.xlabel("cluster number (ordered)")
        plt.ylabel("points in cluster")

        return fig

