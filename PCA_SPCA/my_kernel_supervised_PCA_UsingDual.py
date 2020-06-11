import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv

class My_kernel_supervised_PCA_UsingDual:

    def __init__(self, n_components=None, kernel_on_labels=None, kernel=None):
        self.n_components = n_components
        self.S = None
        self.V = None
        self.X = None
        self.Delta = None
        self.mean_of_X = None
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
        # kernel_Y_Y = pairwise_kernels(X=Y.T, Y=Y.T, metric=self.kernel_on_labels)
        kernel_Y_Y = self.delta_kernel(Y=Y)
        Q, omega, Qh = LA.svd(kernel_Y_Y, full_matrices=True)
        omega = np.asarray(omega)
        Omega_square_root = np.diag(omega**0.5)
        self.Delta = Q.dot(Omega_square_root)
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        kernel_X_X = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        Psi_transpose_Phi = (self.Delta.T).dot(H).dot(kernel_X_X).dot(H).dot(self.Delta)
        eig_val, eig_vec = LA.eigh(Psi_transpose_Phi)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        if self.n_components is not None:
            V = eig_vec[:, :self.n_components]
            s = eig_val[:self.n_components]
        else:
            V = eig_vec
            s = eig_val
        s = np.asarray(s)
        S = np.diag(s)
        self.S = S
        self.V = V

    def transform(self, X, Y=None):
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        kernel_X_X = pairwise_kernels(X=X.T, Y=X.T, metric=self.kernel)
        # X_transformed = (inv(self.S)).dot(self.V.T).dot(self.Delta.T).dot(H).dot(kernel_X_X).dot(H)  #--> rather than X_centered, we can use X.dot(H)
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        X_transformed = (inv(S)).dot(self.V.T).dot(self.Delta.T).dot(H).dot(kernel_X_X)
        return X_transformed

    def transform_outOfSample_all_together(self, X):
        # X: rows are features and columns are samples
        # X = X - self.mean_of_X
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        kernel_X_X = pairwise_kernels(X=self.X.T, Y=X.T, metric=self.kernel)
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        x_transformed = (inv(S)).dot(self.V.T).dot(self.Delta.T).dot(H).dot(kernel_X_X)
        return x_transformed

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