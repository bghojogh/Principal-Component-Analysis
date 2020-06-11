from my_PCA import My_PCA
from my_dual_PCA import My_dual_PCA
from my_kernel_PCA import My_kernel_PCA
from my_supervised_PCA import My_supervised_PCA
from my_dual_supervised_PCA import My_dual_supervised_PCA
from my_kernel_supervised_PCA_UsingDual import My_kernel_supervised_PCA_UsingDual
from my_kernel_supervised_PCA_UsingDirect import My_kernel_supervised_PCA_UsingDirect
from sklearn.preprocessing import StandardScaler
import pickle
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from skimage.transform import resize
from matplotlib import offsetbox
import pandas as pd
import scipy.io
import csv
import scipy.misc
import os
import math
from sklearn.model_selection import train_test_split   #--> https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html




def main():
    # ---- settings:
    dataset = "ATT"  #--> Frey, ATT, ATT_glasses
    manifold_learning_method = "supervised_PCA" #--> PCA, dual_PCA, kernel_PCA, supervised_PCA, dual_supervised_PCA, kernel_SPCA_UsingDual, kernel_SPCA_UsingDirect, MDS, Isomap, Laplacian_eigenmap
    kernel = "linear"  #kernel over data (X) --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    kernel_on_labels_in_SPCA = "linear"  #kernel over labels (Y) --> ‘rbf’, ‘sigmoid’, ‘polynomial’, ‘poly’, ‘linear’, ‘cosine’ --> if None, it is linear
    n_neighbors_in_KNN = 5

    split_to_train_and_test = False
    save_projection_directions_again = False
    reconstruct_using_howMany_projection_directions = None  # --> an integer >= 1, if None: using all "specified" directions when creating the python class

    process_out_of_sample_all_together = True
    project_out_of_sample = True
    convert_mat2csv_again = False
    show_an_image = False
    save_image_dataset_again = False
    n_projection_directions_to_save = 10 #--> an integer >= 1, if None: save all "specified" directions when creating the python class
    save_reconstructed_images_again = False
    save_reconstructed_outOfSample_images_again = False
    if dataset == "Frey":
        indices_reconstructed_images_to_save = [100, 120]  #--> list of two indices (start and end), e.g. [100,120] --> if None, save all of them
        outOfSample_indices_reconstructed_images_to_save = [100, 120]
    elif dataset == "ATT":
        indices_reconstructed_images_to_save = None
        outOfSample_indices_reconstructed_images_to_save = None
    plot_projected_pointsAndImages_again = True
    which_dimensions_to_plot_inpointsAndImagesPlot = [0,1] #--> list of two indices (start and end), e.g. [1,3] or [0,1]


    if dataset == "Frey":
        labels = None
        path_dataset = "./Frey_dataset/frey_rawface"
        # ---- read mat dataset and convert to csv:
        if convert_mat2csv_again:
            convert_mat_to_csv(path_mat=path_dataset, path_to_save=path_dataset)
        # ---- read the csv dataset:
        data = read_csv_file(path=path_dataset+".csv")
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data = data/255
        data_notNormalized = data
        # ---- normalize (standardation):
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T
        # ---- image settings:
        n_samples = data.shape[1]
        image_height = 28
        image_width = 20
        # ---- show one of the images:
        if show_an_image:
            an_image = data[:, 0].reshape((image_height, image_width))
            plt.imshow(an_image, cmap='gray')
            plt.colorbar()
            plt.show()
        # ---- save the images in a folder:
        if save_image_dataset_again:
            for image_index in range(n_samples):
                an_image = data[:, image_index].reshape((image_height, image_width))
                # scale (resize) image array:
                an_image = scipy.misc.imresize(arr=an_image, size=500)  #--> 5 times bigger
                # save image:
                save_image(image_array=an_image, path_without_file_name="./Frey_dataset/images/", file_name=str(image_index)+".png")
    elif dataset == "ATT":
        path_dataset = "./Att_dataset/"
        # n_samples = 400
        n_samples = 30
        scaler = None
        image_height = 112
        image_width = 92
        data = np.zeros((image_height*image_width, n_samples))
        labels = np.zeros((1, n_samples))
        for image_index in range(n_samples):
            img = load_image(address_image=path_dataset+str(image_index+1)+".jpg")
            data[:, image_index] = img.ravel()
            labels[:, image_index] = math.floor(image_index / 10) + 1
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data = data / 255
        data_notNormalized = data
        # ---- normalize (standardation):
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T
        # ---- show one of the images:
        if show_an_image:
            an_image = data[:, 0].reshape((image_height, image_width))
            plt.imshow(an_image, cmap='gray')
            plt.colorbar()
            plt.show()
    elif dataset == "ATT_glasses":
        path_dataset = "./Att_glasses/"
        n_samples = 90 * 2
        scaler = None
        image_height = 112
        image_width = 92
        data = np.zeros((image_height * image_width, n_samples))
        labels = np.zeros((1, n_samples))
        image_index = -1
        for class_index in range(2):
            for filename in os.listdir(path_dataset + "class" + str(class_index+1) + "/"):
                image_index = image_index + 1
                if image_index >= n_samples:
                    break
                img = load_image(address_image=path_dataset + "class" + str(class_index+1) + "/" + filename)
                data[:, image_index] = img.ravel()
                labels[:, image_index] = class_index
        # ---- cast dataset from string to float:
        data = data.astype(np.float)
        # ---- change range of images from [0,255] to [0,1]:
        data = data / 255
        data_notNormalized = data
        # ---- normalize (standardation):
        scaler = StandardScaler(with_mean=True, with_std=True).fit(data.T)
        data = (scaler.transform(data.T)).T

    # ---- split the train and test:
    if dataset == "Frey":
        if split_to_train_and_test:
            X_train, X_test = train_test_split(data.T, test_size = 0.2, random_state = 42)
            X_train_notNormalized = scaler.inverse_transform(X=X_train)
            X_test_notNormalized = scaler.inverse_transform(X=X_test)
            X_train = X_train.T
            X_test = X_test.T
            X_train_notNormalized = X_train_notNormalized.T
            X_test_notNormalized = X_test_notNormalized.T
        else:
            X_train = data
            X_train_notNormalized = data_notNormalized
    elif dataset == "ATT":
        if split_to_train_and_test:
            n_classes = len(np.unique(labels))
            X_train = np.empty((data.shape[0], 0))
            X_test = np.empty((data.shape[0], 0))
            Y_train = []
            X_train_notNormalized = np.empty((data.shape[0], 0))
            X_test_notNormalized = np.empty((data.shape[0], 0))
            for class_index in range(1, n_classes+1):
                mask = (labels.ravel().astype(int) == class_index)
                X_class = data[:, mask==True]
                X_train = np.column_stack((X_train, X_class[:, :6]))
                X_test = np.column_stack((X_test, X_class[:, 6:]))
                Y_train.extend([class_index] * 6)
                X_class_notNormalized = data_notNormalized[:, mask == True]
                X_train_notNormalized = np.column_stack((X_train_notNormalized, X_class_notNormalized[:, :6]))
                X_test_notNormalized = np.column_stack((X_test_notNormalized, X_class_notNormalized[:, 6:]))
            Y_train = np.asarray(Y_train)
        else:
            X_train = data
            X_train_notNormalized = data_notNormalized
            Y_train = labels
    elif dataset == "ATT_glasses":
        if split_to_train_and_test:
            pass
        else:
            X_train = data
            X_train_notNormalized = data_notNormalized
            Y_train = labels

    # ---- apply manifold learning (fit + project training data + projection directions):
    if manifold_learning_method == "PCA":
        my_manifold_learning = My_PCA(n_components=None)
        data_transformed = my_manifold_learning.fit_transform(X=X_train)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "dual_PCA":
        my_manifold_learning = My_dual_PCA(n_components=None)
        data_transformed = my_manifold_learning.fit_transform(X=X_train)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "kernel_PCA":
        my_manifold_learning = My_kernel_PCA(n_components=None, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=X_train)
    elif manifold_learning_method == "supervised_PCA":
        my_manifold_learning = My_supervised_PCA(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "dual_supervised_PCA":
        my_manifold_learning = My_dual_supervised_PCA(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)
        projection_directions = my_manifold_learning.get_projection_directions()
    elif manifold_learning_method == "kernel_SPCA_UsingDual":
        my_manifold_learning = My_kernel_supervised_PCA_UsingDual(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)
    elif manifold_learning_method == "kernel_SPCA_UsingDirect":
        my_manifold_learning = My_kernel_supervised_PCA_UsingDirect(n_components=None, kernel_on_labels=kernel_on_labels_in_SPCA, kernel=kernel)
        data_transformed = my_manifold_learning.fit_transform(X=X_train, Y=Y_train)

    # ---- save projection directions:
    if save_projection_directions_again:
        if n_projection_directions_to_save == None:
            n_projection_directions_to_save = projection_directions.shape[1]
        for projection_direction_index in range(n_projection_directions_to_save):
            an_image = projection_directions[:, projection_direction_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            save_image(image_array=an_image, path_without_file_name="./output/"+manifold_learning_method+"/directions/", file_name=str(projection_direction_index)+".png")

    # ---- save reconstructed images:
    if save_reconstructed_images_again and not split_to_train_and_test:
        X_reconstructed = my_manifold_learning.reconstruct(X=X_train, scaler=scaler, using_howMany_projection_directions=reconstruct_using_howMany_projection_directions)
        if indices_reconstructed_images_to_save == None:
            indices_reconstructed_images_to_save = [0, X_reconstructed.shape[1]]
        for image_index in range(indices_reconstructed_images_to_save[0], indices_reconstructed_images_to_save[1]):
            an_image = X_reconstructed[:, image_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            # save image:
            if reconstruct_using_howMany_projection_directions is not None:
                tmp = "_using" + str(reconstruct_using_howMany_projection_directions) + "Directions"
            else:
                tmp = "_usingAllDirections"
            save_image(image_array=an_image, path_without_file_name="./output/"+manifold_learning_method+"/reconstructed_train"+tmp+"/", file_name=str(image_index)+".png")

    # ---- project out-of-sample data:
    if split_to_train_and_test and project_out_of_sample:
        if not process_out_of_sample_all_together:
            if manifold_learning_method == "kernel_PCA" or manifold_learning_method == "kernel_SPCA_UsingDual" or manifold_learning_method == "kernel_SPCA_UsingDirect":
                data_transformed_test = np.zeros((X_train.shape[1], X_test.shape[1])) #--> because in kernel methods, dimension will be n_train and not d
            else:
                data_transformed_test = np.zeros(X_test.shape)
            for test_sample_index in range(X_test.shape[1]):
                print("processing image " + str(test_sample_index) + " out of " + str(X_test.shape[1]))
                data_transformed_test[:, test_sample_index] = my_manifold_learning.transform_outOfSample(x=X_test[:, test_sample_index]).ravel()
        else:
            data_transformed_test = my_manifold_learning.transform_outOfSample_all_together(X=X_test)

    # ---- save reconstructed out-of-sample images:
    if split_to_train_and_test and save_reconstructed_outOfSample_images_again:
        if not process_out_of_sample_all_together:
            X_test_reconstructed = np.zeros(X_test.shape)
            for test_sample_index in range(X_test.shape[1]):
                X_test_reconstructed[:, test_sample_index] = my_manifold_learning.reconstruct_outOfSample(x=X_test[:, test_sample_index], using_howMany_projection_directions=reconstruct_using_howMany_projection_directions).ravel()
        else:
            X_test_reconstructed = my_manifold_learning.reconstruct_outOfSample_all_together(X=X_test, scaler=scaler, using_howMany_projection_directions=reconstruct_using_howMany_projection_directions)
        if outOfSample_indices_reconstructed_images_to_save == None:
            outOfSample_indices_reconstructed_images_to_save = [0, X_test_reconstructed.shape[1]]
        for image_index in range(outOfSample_indices_reconstructed_images_to_save[0], outOfSample_indices_reconstructed_images_to_save[1]):
            an_image = X_test_reconstructed[:, image_index].reshape((image_height, image_width))
            an_image_original = X_test_notNormalized[:, image_index].reshape((image_height, image_width))
            # scale (resize) image array:
            an_image = scipy.misc.imresize(arr=an_image, size=500)  # --> 5 times bigger
            an_image_original = scipy.misc.imresize(arr=an_image_original, size=500)  # --> 5 times bigger
            # save image:
            if reconstruct_using_howMany_projection_directions is not None:
                tmp = "_using" + str(reconstruct_using_howMany_projection_directions) + "Directions"
            else:
                tmp = "_usingAllDirections"
            save_image(image_array=an_image, path_without_file_name="./output/"+manifold_learning_method+"/reconstructed_test"+tmp+"/", file_name=str(image_index)+".png")
            save_image(image_array=an_image_original, path_without_file_name="./output/"+manifold_learning_method+"/test_to_reconstruct/", file_name=str(image_index)+".png")


    # Plotting the embedded data:
    if dataset == "Frey":
        scale = 5
    elif dataset == "ATT" or dataset == "ATT_glasses":
        scale = 1
    if plot_projected_pointsAndImages_again:
        if not split_to_train_and_test:
            dataset_notReshaped = np.zeros((n_samples, image_height*scale, image_width*scale))
            for image_index in range(n_samples):
                image = data_notNormalized[:, image_index]
                image_not_reshaped = image.reshape((image_height, image_width))
                image_not_reshaped_scaled = scipy.misc.imresize(arr=image_not_reshaped, size=scale*100)
                dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
            fig, ax = plt.subplots(figsize=(10, 10))
            # only take two dimensions to plot:
            if dataset != "ATT_glasses":
                plot_components(X_projected=data_transformed, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                                images=255-dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.07, cmap='gray_r')
                # plot_components_by_colors(X_projected=data_transformed, y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot, ax=None, markersize=20)
            else:
                plot_components_2(X_projected=data_transformed, Y_projected=Y_train, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                                images=255-dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.01, cmap='gray_r')
        elif split_to_train_and_test:
            training_dataset_notReshaped = np.zeros((X_train.shape[1], image_height*scale, image_width*scale))
            for image_index in range(X_train.shape[1]):
                image = X_train_notNormalized[:, image_index]
                image_not_reshaped = image.reshape((image_height, image_width))
                image_not_reshaped_scaled = scipy.misc.imresize(arr=image_not_reshaped, size=scale*100)
                training_dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
            test_dataset_notReshaped = np.zeros((X_train.shape[1], image_height*scale, image_width*scale))
            for image_index in range(X_test.shape[1]):
                image = X_test_notNormalized[:, image_index]
                image_not_reshaped = image.reshape((image_height, image_width))
                image_not_reshaped_scaled = scipy.misc.imresize(arr=image_not_reshaped, size=scale*100)
                test_dataset_notReshaped[image_index, :, :] = image_not_reshaped_scaled
            fig, ax = plt.subplots(figsize=(10, 10))
            # only take two dimensions to plot:
            plot_components_with_test(X_projected=data_transformed, X_test_projected=data_transformed_test, which_dimensions_to_plot=which_dimensions_to_plot_inpointsAndImagesPlot,
                            images=255-training_dataset_notReshaped, images_test=255-test_dataset_notReshaped, ax=ax, image_scale=0.25, markersize=10, thumb_frac=0.07, cmap='gray_r')


def convert_mat_to_csv(path_mat, path_to_save):
    # https://gist.github.com/Nixonite/bc2f69b0c4430211bcad
    data = scipy.io.loadmat(path_mat)
    for i in data:
        if '__' not in i and 'readme' not in i:
            np.savetxt((path_to_save + i + ".csv"), data[i], delimiter=',')

def read_csv_file(path):
    # https://stackoverflow.com/questions/46614526/how-to-import-a-csv-file-into-a-data-array
    with open(path, newline='') as csvfile:
        data = list(csv.reader(csvfile))
    # convert to numpy array:
    data = np.asarray(data)
    return data

def load_image(address_image):
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.open(address_image).convert('L')
    img_arr = np.array(img)
    return img_arr

def save_image(image_array, path_without_file_name, file_name):
    if not os.path.exists(path_without_file_name):
        os.makedirs(path_without_file_name)
    # http://code.activestate.com/recipes/577591-conversion-of-pil-image-and-numpy-array/
    img = Image.fromarray(image_array)
    img = img.convert("L")
    img.save(path_without_file_name + file_name)

def show_image(img):
    plt.imshow(img)
    plt.gray()
    plt.show()

def plot_components_by_colors(X_projected, y_projected, which_dimensions_to_plot, ax=None, markersize=10):
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    n_classes = len(np.unique(y_projected))
    colors = get_spaced_colors(n=n_classes)
    for sample_index in range(X_projected.shape[0]):
        class_label = (y_projected.ravel()[sample_index]).astype(int) - 1
        color = [colors[class_label][0] / 255, colors[class_label][1] / 255, colors[class_label][2] / 255]
        ax.plot(X_projected[sample_index, 0], X_projected[sample_index, 1], '.', color=color, markersize=markersize)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def get_spaced_colors(n):
    # https://www.quora.com/How-do-I-generate-n-visually-distinct-RGB-colours-in-Python
    max_value = 16581375  # 255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]

def plot_components(X_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def plot_components_with_test(X_projected, X_test_projected, which_dimensions_to_plot, images=None, images_test=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    n_test_samples = X_test_projected.shape[1]
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_test_projected = np.vstack((X_test_projected[which_dimensions_to_plot[0], :], X_test_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    X_test_projected = X_test_projected.T
    ax = ax or plt.gca()
    ax.plot(X_projected[:, 0], X_projected[:, 1], '.k', markersize=markersize)
    ax.plot(X_test_projected[:, 0], X_test_projected[:, 1], '.r', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    images_test = resize(images_test, (images_test.shape[0], images_test.shape[1]*image_scale, images_test.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            # test image:
            if i < n_test_samples:
                dist = np.sum((X_test_projected[i] - shown_images) ** 2, 1)
                if np.min(dist) < min_dist_2:
                    # don't show points that are too close
                    continue
                shown_images = np.vstack([shown_images, X_test_projected[i]])
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images_test[i], cmap=cmap), X_test_projected[i], bboxprops =dict(edgecolor='red', lw=3))
                ax.add_artist(imagebox)
            # training image:
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame):
        # https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # https://matplotlib.org/users/annotations_guide.html
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

def plot_components_2(X_projected, Y_projected, which_dimensions_to_plot, images=None, ax=None, image_scale=1, markersize=10, thumb_frac=0.05, cmap='gray'):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.10-manifold-learning.html
    # X_projected: rows are features and columns are samples
    # which_dimensions_to_plot: a list of two integers, index starting from 0
    X_projected = np.vstack((X_projected[which_dimensions_to_plot[0], :], X_projected[which_dimensions_to_plot[1], :])) #--> only take two dimensions to plot
    X_projected = X_projected.T
    ax = ax or plt.gca()
    Y_projected = Y_projected.ravel()
    ax.plot(X_projected[Y_projected == 0, 0], X_projected[Y_projected == 0, 1], '.k', markersize=markersize)
    ax.plot(X_projected[Y_projected == 1, 0], X_projected[Y_projected == 1, 1], '.r', markersize=markersize)
    images = resize(images, (images.shape[0], images.shape[1]*image_scale, images.shape[2]*image_scale), order=5, preserve_range=True, mode="constant")
    if images is not None:
        min_dist_2 = (thumb_frac * max(X_projected.max(0) - X_projected.min(0))) ** 2
        shown_images = np.array([2 * X_projected.max(0)])
        for i in range(X_projected.shape[0]):
            dist = np.sum((X_projected[i] - shown_images) ** 2, 1)
            if np.min(dist) < min_dist_2:
                # don't show points that are too close
                continue
            shown_images = np.vstack([shown_images, X_projected[i]])
            if Y_projected[i] == 0:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i])
            else:
                imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[i], cmap=cmap), X_projected[i], bboxprops =dict(edgecolor='red', lw=3))
            ax.add_artist(imagebox)
        # # plot the first (original) image once more to be on top of other images:
        # # change color of frame (I googled: python OffsetImage highlight frame): https://stackoverflow.com/questions/40342379/show-images-in-a-plot-using-matplotlib-with-a-coloured-frame
        # imagebox = offsetbox.AnnotationBbox(offsetbox.OffsetImage(images[0], cmap=cmap), X_projected[0], bboxprops =dict(edgecolor='red'))
        # ax.add_artist(imagebox)
    plt.xlabel("dimension " + str(which_dimensions_to_plot[0] + 1), fontsize=13)
    plt.ylabel("dimension " + str(which_dimensions_to_plot[1] + 1), fontsize=13)
    plt.show()

if __name__ == '__main__':
    main()