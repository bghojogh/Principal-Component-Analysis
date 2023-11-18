import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels

class My_supervised_PCA:

    def __init__(self, n_components=None, kernel_on_labels=None):
        self.n_components = n_components
        self.U = None
        self.mean_of_X = None
        if kernel_on_labels != None:
            self.kernel_on_labels = kernel_on_labels
        else:
            self.kernel_on_labels = "linear"

    def fit_transform(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.fit(X, Y)
        X_transformed = self.transform(X, Y)
        return X_transformed

    def delta_kernel(self, Y):
        Y = Y.ravel()
        grid_x, grid_y = np.meshgrid(Y, Y)
        delta_kernel = np.where(grid_x == grid_y, 1., 0.)
        return delta_kernel

    def fit(self, X, Y):
        # X: rows are features and columns are samples
        # Y: rows are dimensions of labels (usually 1-dimensional) and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        n = X.shape[1]
        H = np.eye(n) - ((1/n) * np.ones((n,n)))
        # B = pairwise_kernels(X=Y.T, Y=Y.T, metric=self.kernel_on_labels)
        B = self.delta_kernel(Y=Y)
        eig_val, eig_vec = LA.eigh( X.dot(H).dot(B).dot(H).dot(X.T) )
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            self.U = eig_vec[:, :self.n_components]
        else:
            self.U = eig_vec

    def transform(self, X, Y=None):
        # X_centered = self.center_the_matrix(the_matrix=X, mode="remove_mean_of_columns_from_columns")
        # X_transformed = (self.U.T).dot(X_centered)
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def transform_outOfSample_all_together(self, X):
        # X: rows are features and columns are samples
        # X = X - self.mean_of_X
        x_transformed = (self.U.T).dot(X)
        return x_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        # X_centered = self.center_the_matrix(the_matrix=X, mode="remove_mean_of_columns_from_columns")
        # X_transformed = (U.T).dot(X_centered)
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        # X_reconstructed = X_reconstructed + self.mean_of_X
        return X_reconstructed

    def reconstruct_outOfSample_all_together(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        # x = x - self.mean_of_X
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        # X_reconstructed = X_reconstructed + self.mean_of_X
        return X_reconstructed

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