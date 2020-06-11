import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
import os
import pickle


class My_dual_PCA:

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
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        X = X - self.mean_of_X
        self.X = X
        U, s, Vh = LA.svd(self.X, full_matrices=False)  #---> in dual PCA, the S should be square so --> full_matrices=False
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
        X_transformed = (self.S).dot(self.V.T)
        return X_transformed

    def transform_outOfSample(self, x):
        # x: a vector
        x = np.reshape(x,(-1,1))
        x = x - self.mean_of_X
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        x_transformed = (inv(self.S)).dot(self.V.T).dot(H).dot(self.X.T).dot(x)
        return x_transformed

    def transform_outOfSample_all_together(self, X, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        X = X - self.mean_of_X
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        X_transformed = (inv(self.S)).dot(self.V.T).dot(H).dot(self.X.T).dot(X)
        return X_transformed

    def get_projection_directions(self):
        return self.U

    def reconstruct(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        self.mean_of_X = X.mean(axis=1).reshape((-1, 1))
        X = X - self.mean_of_X
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
        else:
            V = self.V
        X_reconstructed = X.dot(V).dot(V.T)
        X_reconstructed = X_reconstructed + self.mean_of_X
        if scaler is not None:
            X_reconstructed = scaler.inverse_transform(X=X_reconstructed.T)
            X_reconstructed = X_reconstructed.T
        return X_reconstructed

    def reconstruct_outOfSample(self, x, using_howMany_projection_directions=None):
        # x: a vector
        x = np.reshape(x, (-1, 1))
        x = x - self.mean_of_X
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
            S = self.S[0:using_howMany_projection_directions, 0:using_howMany_projection_directions]
        else:
            V = self.V
            S = self.S
        x_reconstructed = (self.X).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.X.T).dot(x)
        x_reconstructed = x_reconstructed + self.mean_of_X
        return x_reconstructed

    def reconstruct_outOfSample_all_together(self, X, scaler=None, using_howMany_projection_directions=None):
        # X: rows are features and columns are samples
        X = X - self.mean_of_X
        if using_howMany_projection_directions != None:
            V = self.V[:, 0:using_howMany_projection_directions]
            S = self.S[0:using_howMany_projection_directions, 0:using_howMany_projection_directions]
        else:
            V = self.V
            S = self.S
        X_reconstructed = (self.X).dot(V).dot(inv(S)).dot(inv(S)).dot(V.T).dot(self.X.T).dot(X)
        X_reconstructed = X_reconstructed + self.mean_of_X
        if scaler is not None:
            X_reconstructed = scaler.inverse_transform(X=X_reconstructed.T)
            X_reconstructed = X_reconstructed.T
        return X_reconstructed