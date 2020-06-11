import numpy as np
from numpy import linalg as LA
from numpy.linalg import inv
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import kneighbors_graph as KNN   # https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.kneighbors_graph.html
import os
import pickle

# ----- python fast kernel matrix:
# http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_kernels.html
# https://stackoverflow.com/questions/7391779/fast-kernel-matrix-computation-python
# https://stats.stackexchange.com/questions/15798/how-to-calculate-a-gaussian-kernel-effectively-in-numpy
# https://stackoverflow.com/questions/36324561/fast-way-to-calculate-kernel-matrix-python?rq=1

# ----- python fast scatter matrix:
# https://stackoverflow.com/questions/31145918/fast-weighted-scatter-matrix-calculation


class My_kernel_PCA:

    def __init__(self, n_components=None, kernel=None):
        self.n_components = n_components
        self.X = None
        self.S = None
        self.V = None
        self.scaler = None
        self.X_mean = None
        if kernel is not None:
            self.kernel = kernel
        else:
            self.kernel = 'linear'

    def fit_transform(self, X):
        # X: rows are features and columns are samples
        self.fit(X)
        X_transformed = self.transform(X)
        return X_transformed

    def fit(self, X):
        # X: rows are features and columns are samples
        self.X = X
        self.X_mean = (self.X.mean(axis=1)).reshape((-1,1))
        # self.scaler = StandardScaler().fit(self.X.T)
        # self.X = (self.scaler.transform(self.X.T)).T
        kernel_X_X = pairwise_kernels(X=self.X.T, Y=self.X.T, metric=self.kernel)
        kernel_X_X = self.center_the_matrix(the_matrix=kernel_X_X, mode="double_center")
        eig_val, eig_vec = LA.eigh(kernel_X_X)
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
        s = s ** 0.5
        s = np.nan_to_num(s)
        S = np.diag(s)
        self.S = S
        self.V = V

    def transform(self, X):
        # X: rows are features and columns are samples
        X_transformed = (self.S).dot(self.V.T)
        return X_transformed

    def transform_outOfSample(self, x):
        # x: a vector
        x = np.reshape(x,(-1,1))
        kernel_outOfSample = pairwise_kernels(X=self.X.T, Y=x.T, metric=self.kernel)
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        kernel_outOfSample_centered = self.center_kernel_of_outOfSample(kernel_of_outOfSample=kernel_outOfSample, matrix_or_vector="vector")
        x_transformed = (inv(S)).dot(self.V.T).dot(kernel_outOfSample_centered)
        return x_transformed

    def transform_outOfSample_all_together(self, X):
        # X: rows are features and columns are samples
        kernel_outOfSample = pairwise_kernels(X=self.X.T, Y=X.T, metric=self.kernel)
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        kernel_outOfSample_centered = self.center_kernel_of_outOfSample(kernel_of_outOfSample=kernel_outOfSample, matrix_or_vector="matrix")
        x_transformed = (inv(S)).dot(self.V.T).dot(kernel_outOfSample_centered)
        return x_transformed

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

    def transform_outOfSample_matrix(self, X):
        X = (self.scaler.transform(X.T)).T
        kernel_X_x = pairwise_kernels(X=self.X.T, Y=X.T, metric=self.kernel)
        n = self.X.shape[1]
        H = np.eye(n) - ((1 / n) * np.ones((n, n)))
        diag_S = np.diag(self.S) + 0.0000001
        S = np.diag(diag_S)
        kernel_X_x = self.center_the_matrix(the_matrix=kernel_X_x, mode="double_center")
        x_transformed = (inv(S)).dot(self.V.T).dot(kernel_X_x)
        # x_transformed = 10 * x_transformed
        # print(self.X_mean.shape)
        # print(x_transformed.shape)
        # print(self.X.shape)
        # x_transformed = x_transformed - self.X_mean
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

    def classify_distortion_trainingSet(self):
        n_samples = self.X.shape[1]
        X_projected = (self.S).dot(self.V.T)
        # --- KNN:
        estimated_distortion_class = np.zeros((n_samples, 2))
        connectivity_matrix = KNN(X=X_projected.T, n_neighbors=1, mode='connectivity', include_self=False, n_jobs=-1)
        connectivity_matrix = connectivity_matrix.toarray()
        for image_index in range(n_samples):
            index_of_neighbor = int(np.argwhere(connectivity_matrix[image_index, :] == 1))
            if index_of_neighbor == 0:  # "original"
                estimated_distortion_class[image_index, 0] = 0
            elif index_of_neighbor >= 1 and index_of_neighbor <= 20:  # "contrast_stretched"
                estimated_distortion_class[image_index, 0] = 1
            elif index_of_neighbor >= 21 and index_of_neighbor <= 40:  # "Gaussian_noise"
                estimated_distortion_class[image_index, 0] = 2
            elif index_of_neighbor >= 41 and index_of_neighbor <= 60:  # "enhanced_luminance"
                estimated_distortion_class[image_index, 0] = 3
            elif index_of_neighbor >= 61 and index_of_neighbor <= 80:  # "Gaussian_blurring"
                estimated_distortion_class[image_index, 0] = 4
            elif index_of_neighbor >= 81 and index_of_neighbor <= 100:  # "impulse_noise"
                estimated_distortion_class[image_index, 0] = 5
            elif index_of_neighbor >= 101 and index_of_neighbor <= 120:  # "jpeg_distortion"
                estimated_distortion_class[image_index, 0] = 6
            estimated_distortion_class[image_index, 1] = image_index
        path_to_save = './output/kernel_PCA/' + self.kernel + '/classification/'
        self.save_variable(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        self.save_np_array_to_txt(variable=estimated_distortion_class, name_of_variable="estimated_distortion_class", path_to_save=path_to_save)
        return estimated_distortion_class[:, 0]

    def save_variable(self, variable, name_of_variable, path_to_save='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.pckl'
        f = open(file_address, 'wb')
        pickle.dump(variable, f)
        f.close()

    def load_variable(self, name_of_variable, path='./'):
        # https://stackoverflow.com/questions/6568007/how-do-i-save-and-restore-multiple-variables-in-python
        file_address = path + name_of_variable + '.pckl'
        f = open(file_address, 'rb')
        variable = pickle.load(f)
        f.close()
        return variable

    def save_np_array_to_txt(self, variable, name_of_variable, path_to_save='./'):
        if type(variable) is list:
            variable = np.asarray(variable)
        # https://stackoverflow.com/questions/22821460/numpy-save-2d-array-to-text-file/22822701
        if not os.path.exists(path_to_save):  # https://stackoverflow.com/questions/273192/how-can-i-create-a-directory-if-it-does-not-exist
            os.makedirs(path_to_save)
        file_address = path_to_save + name_of_variable + '.txt'
        np.set_printoptions(threshold=np.inf, linewidth=np.inf)  # turn off summarization, line-wrapping
        with open(file_address, 'w') as f:
            f.write(np.array2string(variable, separator=', '))