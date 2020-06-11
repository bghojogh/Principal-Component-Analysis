import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv
from sklearn.preprocessing import StandardScaler

class My_dual_supervised_PCA:

    def __init__(self, n_components=None, kernel_on_labels=None):
        self.n_components = n_components
        self.U = None
        self.S = None
        self.V = None
        self.X = None
        self.Delta = None
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
        self.mean_of_X = self.X.mean(axis=1).reshape((-1, 1))
        n = self.X.shape[1]
        # B = pairwise_kernels(X=Y.T, Y=Y.T, metric=self.kernel_on_labels)
        B = self.delta_kernel(Y=Y)
        Q, omega, Qh = LA.svd(B, full_matrices=True)
        omega = np.asarray(omega)
        Omega_square_root = np.diag(omega**0.5)
        self.Delta = Q.dot(Omega_square_root)
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        Psi = (self.X).dot(H).dot(self.Delta)
        U, s, Vh = LA.svd(Psi, full_matrices=False)  # ---> in dual SPCA, the S should be square so --> full_matrices=False
        V = Vh.T
        if self.n_components != None:
            U = U[:, :self.n_components]
            s = s[:self.n_components]
            V = V[:, :self.n_components]
        s = np.asarray(s)
        S = np.diag(s)
        self.U = U
        self.S = S
        self.V = V

    def transform(self, X, Y=None):
        #X_centered = self.center_the_matrix(the_matrix=X, mode="remove_mean_of_columns_from_columns")
        # mean_of_X = X.mean(axis=1)
        # mean_of_X = np.reshape(mean_of_X, (-1,1))
        # X_centered = X - mean_of_X
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        # X_transformed = (inv(self.S)).dot(self.V.T).dot(self.Delta.T).dot(H).dot(X.T).dot(X_centered)  #--> rather than X_centered, we can use X.dot(H)
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        X_transformed = (inv(S)).dot(self.V.T).dot(self.Delta.T).dot(H).dot(X.T).dot(X)
        return X_transformed

    def transform_outOfSample_all_together(self, X):
        # X: rows are features and columns are samples
        # X = X - self.mean_of_X
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        X_transformed = (inv(S)).dot(self.V.T).dot(self.Delta.T).dot(H).dot(self.X.T).dot(X)
        return X_transformed

    # def transform_outOfSample_2(self, X_test):
    #     # X_test: rows are features and columns are samples
    #     n = self.X.shape[1]
    #     H = np.eye(n) - ((1 / n) * np.ones((n, n)))
    #     K = (self.X.T).dot(X_test)
    #     K = self.center_the_matrix(the_matrix=K, mode="double_center")
    #     X_test_transformed = (inv(self.S)).dot(self.V.T).dot(self.Delta.T).dot(K)
    #     return X_test_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        # X_centered = self.center_the_matrix(the_matrix=X, mode="remove_mean_of_columns_from_columns")
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
            S = self.S[0:using_howMany_projection_directions, 0:using_howMany_projection_directions]
        else:
            V = self.V
            S = self.S
        n = X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        # X_reconstructed = X.dot(H).dot(self.Delta).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.Delta.T).dot(H).dot(X.T).dot(X_centered) #--> rather than X_centered, we can use X.dot(H)
        # X_reconstructed = X_reconstructed + self.mean_of_X
        diag_S = np.diag(S) + 0.0000001
        S = np.diag(diag_S)
        X_reconstructed = X.dot(H).dot(self.Delta).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.Delta.T).dot(H).dot(X.T).dot(X)
        return X_reconstructed

    def reconstruct_outOfSample_all_together(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        # X = X - self.mean_of_X
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
            S = self.S[0:using_howMany_projection_directions, 0:using_howMany_projection_directions]
        else:
            V = self.V
            S = self.S
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        # x_reconstructed = (self.X).dot(H).dot(self.Delta).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.Delta.T).dot(H).dot(self.X.T).dot(x)
        # x_reconstructed = x_reconstructed + self.mean_of_X
        diag_S = np.diag(S) + 0.0000001
        S = np.diag(diag_S)
        X_reconstructed = (self.X).dot(H).dot(self.Delta).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.Delta.T).dot(H).dot(self.X.T).dot(X)
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