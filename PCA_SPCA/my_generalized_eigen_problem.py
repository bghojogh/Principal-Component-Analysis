import numpy as np
from numpy import linalg as LA
from sklearn.metrics.pairwise import pairwise_kernels
from numpy.linalg import inv

class My_generalized_eigen_problem:

    def __init__(self, A, B):
        # A Phi = B Phi Lambda --> Phi: eigenvectors, Lambda: eigenvalues
        self.A = A
        self.B = B

    def solve(self):
        Phi_B, Lambda_B = self.eigen_decomposition(matrix=self.B)
        lambda_B = Lambda_B.diagonal()
        a = lambda_B**0.5
        a = np.nan_to_num(a) + 0.0001
        # Lambda_B_squareRoot = np.diag(lambda_B**0.5)
        Lambda_B_squareRoot = np.diag(a)
        Phi_B_hat = Phi_B.dot(inv(Lambda_B_squareRoot))
        A_hat = (Phi_B_hat.T).dot(self.A).dot(Phi_B_hat)
        Phi_A, Lambda_A = self.eigen_decomposition(matrix=A_hat)
        Lambda = Lambda_A
        Phi = Phi_B_hat.dot(Phi_A)
        return Phi, Lambda

    def eigen_decomposition(self, matrix):
        eig_val, eig_vec = LA.eigh(matrix)
        idx = eig_val.argsort()[::-1]  # sort eigenvalues in descending order (largest eigenvalue first)
        eig_val = eig_val[idx]
        eig_vec = eig_vec[:, idx]
        Eigenvectors = eig_vec
        eigenvalues = eig_val
        eigenvalues = np.asarray(eigenvalues)
        Eigenvalues = np.diag(eigenvalues)
        return Eigenvectors, Eigenvalues