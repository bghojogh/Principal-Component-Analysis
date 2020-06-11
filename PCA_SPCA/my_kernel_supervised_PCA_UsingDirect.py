import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv
from my_generalized_eigen_problem import My_generalized_eigen_problem

class My_kernel_supervised_PCA_UsingDirect:

    def __init__(self, n_components=None, kernel_on_labels=None, kernel=None):
        self.n_components = n_components
        self.X = None
        self.mean_of_X = None
        self.Theta = None
        self.Lambda = None
        if kernel_on_labels != None:
            self.kernel_on_labels = kernel_on_labels
        else:
            self.kernel_on_labels = "linear"
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def fit_transform(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed

    def delta_kernel(self, Y):
        Y = Y.ravel()
        n_samples = len(Y)
        delta_kernel = np.zeros((n_samples, n_samples))
        for sample_index_1 in range(n_samples):
            for sample_index_2 in range(n_samples):
                if Y[sample_index_1] == Y[sample_index_2]:
                    delta_kernel[sample_index_1, sample_index_2] = 1
                else:
                    delta_kernel[sample_index_1, sample_index_2] = 0
        return delta_kernel

    def fit(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.X = X
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        kernel_X_X = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        # kernel_Y_Y = pairwise_kernels(X=Y.T, Y=Y.T, metric=self.kernel_on_labels)
        kernel_Y_Y = self.delta_kernel(Y=Y)
        A = kernel_X_X.dot(H).dot(kernel_Y_Y).dot(H).dot(kernel_X_X)
        my_generalized_eigen_problem = My_generalized_eigen_problem(A=A, B=kernel_X_X)
        self.Theta, self.Lambda = my_generalized_eigen_problem.solve()

    def transform(self, X, Y=None):
        # n = X.shape[1]
        # H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        kernel_X_X = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        # X_transformed = (self.Theta.T).dot(H).dot(kernel_X_X).dot(H)
        X_transformed = (self.Theta.T).dot(kernel_X_X)
        return X_transformed

    def transform_outOfSample_all_together(self, X):
        # X: rows are features and columns are samples
        # X = X - self.mean_of_X
        # n = self.X.shape[1]
        # H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        kernel_outOfSample = pairwise_kernels(X=self.X.T, Y=X.T, metric=self.kernel)
        # kernel_outOfSample_centered = self.center_kernel_of_outOfSample(kernel_of_outOfSample=kernel_outOfSample, matrix_or_vector="matrix")
        # X_transformed = (self.Theta.T).dot(kernel_outOfSample_centered)
        X_transformed = (self.Theta.T).dot(kernel_outOfSample)
        return X_transformed

    def center_kernel_of_outOfSample(self, kernel_of_outOfSample, matrix_or_vector="matrix"):
        n_training_samples = self.X.shape[1]
        kernel_X_X_training = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
        if matrix_or_vector == "matrix":
            n_outOfSample_samples = kernel_of_outOfSample.shape[1]
            kernel_of_outOfSample_centered = kernel_of_outOfSample - (1 / n_training_samples) * np.ones((n_training_samples, n_training_samples)).dot(kernel_of_outOfSample) \
                                             - (1 / n_training_samples) * kernel_X_X_training.dot(np.ones((n_training_samples, n_outOfSample_samples))) \
                                             + (1 / n_training_samples**2) * np.ones((n_training_samples, n_training_samples)).dot(kernel_X_X_training).dot(np.ones((n_training_samples, n_outOfSample_samples)))
        elif matrix_or_vector == "vector":
            kernel_of_outOfSample = kernel_of_outOfSample.reshape((-1, 1))
            kernel_of_outOfSample_centered = kernel_of_outOfSample - (1 / n_training_samples) * np.ones((n_training_samples, n_training_samples)).dot(kernel_of_outOfSample) \
                                             - (1 / n_training_samples) * kernel_X_X_training.dot(np.ones((n_training_samples, 1))) \
                                             + (1 / n_training_samples**2) * np.ones((n_training_samples, n_training_samples)).dot(kernel_X_X_training).dot(np.ones((n_training_samples, 1)))
        return kernel_of_outOfSample_centered

    def center_the_matrix(self, the_matrix, mode="double_center"):
        n_rows = the_matrix.shape[0]
        n_cols = the_matrix.shape[1]
        vector_one_left = np.ones((n_rows,1))
        vector_one_right = np.ones((n_cols, 1))
        H_left = np.eye(n_rows) - ((1/n_rows) * vector_one_left.dot(vector_one_left.T))
        H_right = np.eye(n_cols) - ((1 / n_cols) * vector_one_right.dot(vector_one_right.T))
        if mode == "double_center":
            the_matrix = H_left.dot(the_matrix).dot(H_right)
        elif mode == "remove_mean_of_rows_from_rows":
            the_matrix = H_left.dot(the_matrix)
        elif mode == "remove_mean_of_columns_from_columns":
            the_matrix = the_matrix.dot(H_right)
        return the_matrix