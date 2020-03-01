"""
Module to define transformation into an image
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA, KernelPCA
import os


# recieve the pca results and return 2 tuples of (max_[axis], minדג[axis])
def min_max_pt(points):
    """
    method to identify boarders of cropped image
    :param points: points
    :return: tuples of (x_max, x_min), (y_max, y_min)
    """
    min_x = 0
    max_x = 0
    min_y = 0
    max_y = 0

    for arr in points:
        cur_x = arr[0]
        cur_y = arr[1]

        if cur_x < min_x:
            min_x = cur_x
        elif cur_x > max_x:
            max_x = cur_x

        if cur_y < min_y:
            min_y = cur_y
        elif cur_y > max_y:
            max_y = cur_y

    return (max_x, min_x), (max_y, min_y)


# x_tup, y_tup = min_max_pt(sample)
# print(x_tup, y_tup, dist_tup)

# receieve pca plot and resolution.
# return a list that each element is a list related to a featur and containing [x_pixel_coordinate, y_pixel_coordinate] related to that featur in the image map
def divide_to_pixels(scatter, resolution):
    """
    function to define each feature it's pixels coordinate in the picture
    x: goes through column axis, y: goes through rows axis

    :param scatter: an array-like of features each containing it's [x,y] coordinates
    :param resolution: resolution of desired image
    :return: list that each element is a list related to a feature and contains its pixel coordinates
                with respect to resolution
    """
    res = []  # a list of tuples

    x_range, y_range = min_max_pt(scatter)

    x_dist = x_range[0] - x_range[1]
    y_dist = y_range[0] - y_range[1]

    x_skip_step = x_dist / resolution
    y_skip_step = y_dist / resolution

    x_pixels = [x_range[1] + i * x_skip_step for i in range(1, resolution + 1)]
    y_pixels = [y_range[1] + i * y_skip_step for i in range(1, resolution + 1)]

    for arr in scatter:
        cur_x = arr[0]
        cur_y = arr[1]

        point_indices = [None] * 2
        for i in range(len(x_pixels) - 1, -1, -1):

            if cur_x <= x_pixels[i]:
                point_indices[0] = i
            if cur_y <= y_pixels[i]:
                point_indices[1] = i

        res.append(point_indices)

    return res


# for every feature what is the pixel coordinates such as [1,8] means that that featur is related to that pixel
# feature_pix_coordinate = divide_to_pixels(sample, resolution= resolution)
# print(feature_pix_coordinate)
# print(len(feature_pix_coordinate))

"""### assigning each pixel with a list of corresponding features"""


# func that will assign list of features number to every pixel
# returns a 2-d array-like that represents every pixel in the image. each value is a list containing number of feature(s) in that pixel (number is by index i.e: using iloc).
def to_pixel_map(feat_in_pixels, resolution=10):
    """
    function to assign each pixel in the image to it's corresponding features
    :param feat_in_pixels: an array-like of features with their pixel location
    :param resolution: image resolution, defaults to 10
    :return: 2-d array that each element is a list of features contained in that location
    """
    pixel_map = [[[] for __ in range(resolution)] for _ in range(resolution)]
    bad_pixels_count = 0
    for num, feat in enumerate(feat_in_pixels):
        if feat[0] is None or feat[1] is None:
            bad_pixels_count += 1
            continue
        pixel_map[feat[1]][feat[0]].append(num)
        # pixel_map[feat[0]][feat[1]].append(num)

    return pixel_map


# count how many features are in each pixel
# returns a heat map of counts how many features are in every pixel
def feat_count_per_pixel(resolution, feat_in_pixels):
    """
    function to count the amount of features in each pixel
    :param resolution: resolution of the image
    :param feat_in_pixels: array-like of features pixel locations
    :return: 2-d array-like representing an image with count of features in pixels
    """
    heat_map = np.zeros((resolution, resolution))
    bad_pixels_count = 0
    for feat in feat_in_pixels:
        if feat[0] is None or feat[1] is None:
            bad_pixels_count += 1
            continue
        heat_map[feat[1], feat[0]] += 1

    return heat_map


def plot_heat_map(count_of_feat_in_pixel, fig_size):
    """
    function to plot pixels count in image
        ?should be here?
    :param count_of_feat_in_pixel: 2-d array-like of counts
    :param fig_size: size of the image
    :return: None
    """
    df_cm = pd.DataFrame(count_of_feat_in_pixel)
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(df_cm, annot=True)


def df_to_array_of_images(df, pixel_map, resolution, take_average_of_pixel):
    """
    function to...
    :param df:
    :param pixel_map:
    :param resolution:
    :return:
    """
    # print (df.get_value(0, 0, takeable = True))
    # print(df.iloc[0:1,0][0])
    dataset_as_images = []
    for index, row in df.iterrows():
        sample_as_image = np.zeros((resolution, resolution))
        for i in range(resolution):
            for j in range(resolution):
                listOfFeatures = pixel_map[i][j]
                if len(listOfFeatures) == 0:
                    sample_as_image[i][j] = 0
                else:
                    summary = 0
                    max_feature = 0
                    for feature in listOfFeatures:
                        current_feature = df.iat[index, feature] # try df.at() also
                        if take_average_of_pixel:
                            summary = summary + current_feature
                        elif current_feature > max_feature:
                            max_feature = current_feature
                    # summary = summary + df.iloc[index:index+1,gene][0]
                    if (take_average_of_pixel):
                        sample_as_image[i][j] = summary / len(listOfFeatures)
                    else:
                        sample_as_image[i][j] = max_feature
        dataset_as_images.append(sample_as_image)
    return dataset_as_images


# returns a list of lists that each represent a feature coordinates in the new 2-d space
def KPCA(kernel, features_df, fig_size=8):
    """
    function to receive 2-d dimensionality reduction using k-pca
    :param kernel: the used kernel. Options (from sklearn doc):
            'linear', 'poly', 'rbf', 'sigmoid', 'cosine'
    :param features_df: data with a high dimensionality
    :return: a list of each [x,y] coordinates
    """
    k = KernelPCA()
    kpca = KernelPCA(kernel=kernel, gamma=1, n_components=2)
    result = kpca.fit_transform(features_df)

    return result


def plot_kpca(plot_data, fig_size=8):
    """
    function to plot the result of dim
    :param plot_data: the scatter to be plotted
    :param fig_size: the size of plot figure. defaults to 8
    """
    plt.figure(figsize=(fig_size, fig_size))
    plt.scatter(plot_data[:, 0], plot_data[:, 1], edgecolor='', alpha=0.5)
    plt.show()


def save_scatter(plot_data, info, location=None, fig_size=8):
    """
    function to save scatter plot
    :param plot_data: the scatter to be plotted
    :param info: info about the plot. will be used as name of file
    :param location: the location to be saved. defaults to 'Final_Project\\feature maps'
    :param fig_size: size of plot figure. defaults to 8
    """
    save_path = os.getcwd() if location is None else location
    save_path += os.path.join(save_path, 'feature maps', info+'.png')
    plt.figure(figsize=(fig_size, fig_size))
    plt.scatter(plot_data[:, 0], plot_data[:, 1], edgecolor='', alpha=0.5)
    plt.savefig(save_path, dpi=200)


def create_info(dataset_name, reduction_method, method_args, experiment_num):
    """
    function to create the info of a particular experiment
    :param dataset_name:
    :param reduction_method:
    :param method_args:
    :param experiment_num:
    :return: string representing the experiment info
    """
    return f'{dataset_name}_{reduction_method}_{method_args}_{str(experiment_num)}'
