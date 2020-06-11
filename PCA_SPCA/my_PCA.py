import numpy as np
from numpy import linalg as LA


class My_PCA:

    def __init__(self, n_components=None):
        self.n_components = n_components
        self.X = None
        self.U = None
        self.S = None
        self.V = None
        self.mean_of_X = None

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def fit(self, X):
        # X: rows are features and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1,1))
        X = X - self.mean_of_X
        self.X = X
        U, s, Vh = LA.svd(self.X, full_matrices=True)
        V = Vh.T
        if self.n_components != None:
            U = U[:,:self.n_components]
            s = s[:self.n_components]
            V = V[:,:self.n_components]
        s = np.asarray(s)
        S = np.diag(s)
        self.U = U
        self.S = S
        self.V = V

    def transform(self, X):
        # X: rows are features and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        X = X - self.mean_of_X
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def transform_outOfSample(self, x):
        # x: a vector
        x = np.reshape(x,(-1,1))
        x = x - self.mean_of_X
        x_transformed = (self.U.T).dot(x)
        return x_transformed

    def transform_outOfSample_all_together(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        X = X - self.mean_of_X
        X_transformed = (self.U.T).dot(X)
        return X_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X = X - self.mean_of_X
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        X_reconstructed = X_reconstructed + self.mean_of_X
        if scaler is not None:
            X_reconstructed = scaler.inverse_transform(X=X_reconstructed.T)
            X_reconstructed = X_reconstructed.T
        return X_reconstructed

    def reconstruct_outOfSample(self, x, x_means=None, using_howMany_projection_directions=None):
        # x: a vector
        x = np.reshape(x, (-1, 1))
        x = x - self.mean_of_X
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        x_transformed = (U.T).dot(x)
        x_reconstructed = U.dot(x_transformed)
        x_reconstructed = x_reconstructed + self.mean_of_X
        if x_means is not None:
            x_reconstructed = x_reconstructed + x_means
        return x_reconstructed

    def reconstruct_outOfSample_all_together(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        X = X - self.mean_of_X
        if using_howMany_projection_directions != None:
            U = self.U[:, 0:using_howMany_projection_directions]
        else:
            U = self.U
        X_transformed = (U.T).dot(X)
        X_reconstructed = U.dot(X_transformed)
        X_reconstructed = X_reconstructed + self.mean_of_X
        if scaler is not None:
            X_reconstructed = scaler.inverse_transform(X=X_reconstructed.T)
            X_reconstructed = X_reconstructed.T
        return X_reconstructed
