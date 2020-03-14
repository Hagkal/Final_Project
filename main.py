# -*- coding: utf-8 -*-
"""Final Project - convolutional neural network for low level continuos features .ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15J0MUiHK2UF74eMMJqrtfDPbpth9qlGj

## Imports
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import print_function
import os
import math
import time
import configparser

# Viz
import matplotlib
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np

import ImageTransformer as imgt
import CNN
import Datasets as ds

# settings
import warnings

warnings.filterwarnings("ignore")
matplotlib.use('Qt5Agg')
"""
# running tests
"""


def log(step, expect):
    t = time.localtime()
    current_time = time.strftime("%H:%M:%S", t)
    padding = '\t'
    sep = '------'
    print(f'{padding} {sep} step: {step} {sep} status: {expect}')


# ------------------    configuration reading    ----------------- #
configs = configparser.ConfigParser()
configs.read('configurations.ini')

config_working_dataset = configs.get('dataset', 'name')
config_working_labels = configs.getboolean('dataset', 'label')
config_reduction_method = configs.get('reduction', 'method')
config_kpca_kernel = configs.get(config_reduction_method, 'used metric')
config_resolution = configs.getint('image', 'resolution')
config_remote = configs.getboolean('dataset', 'remote')
config_num_of_classes = configs.getint('dataset', 'classes')

# -------------------    reading the data    ------------------ #
dataset, labels = ds.get_dataset(config_working_dataset, remote=config_remote)
print(f'Dataset `{config_working_dataset}` shape is: {dataset.shape}')
if config_working_labels is not False:
    print(f'Dataset labels `{config_working_labels}` shape is: {labels.shape}')

# ------------------    cleaning data and splitting to [train, validation, test]    ------------------ #
splits = ds.split_data(dataset, df_label=labels, data_name=config_working_dataset)

# ------------------    unpacking train    ------------------ #
train_x = splits['train']
validation_x = splits['validation']
test_x = splits['test']
print(f'Original dataset view: ')
print(train_x['original'].head(3))

print(f'\nTranspose dataset view: ')
print(train_x['transpose'].head(3))

# ------------------    applying and plotting k-pca    ------------------ #
log(step='applying and plotting dimensionality reduction', expect='figure should be plotted and saved')
reduction_points = imgt.dimension_reduction(features_df=train_x['transpose'], reduction=config_reduction_method,
                                            metric=config_kpca_kernel)
imgt.plot_scatter(reduction_points)
imgt.save_scatter(reduction_points,
                  imgt.create_info(config_working_dataset, config_reduction_method, config_kpca_kernel, 1))

# ------------------    constructing the image    ------------------ #
log(step='analyzing how to transform dataset to image', expect='might take a few secs')
features_to_pixels = imgt.divide_to_pixels(scatter=reduction_points, resolution=config_resolution)
pixels_to_features = imgt.to_pixel_map(features_to_pixels, resolution=config_resolution)
pixels_features_heat_map = imgt.feat_count_per_pixel(resolution=config_resolution, feat_in_pixels=features_to_pixels)
# imgt.plot_heat_map(pixels_features_heat_map, fig_size=config_resolution)

# ------------------    transferring data to image for train    ------------------ #
log(step='actually do the transformation of the dataset', expect='relax for a few seconds bro')
train_x['original'].reset_index(inplace=True)  # need to check reset index adding extra column 'index'
validation_x['original'].reset_index(inplace=True)

X_train_as_image = imgt.df_to_array_of_images(train_x['original'], pixels_to_features, config_resolution,
                                              take_average_of_pixel=False)
X_validation_as_image = imgt.df_to_array_of_images(validation_x['original'], pixels_to_features, config_resolution,
                                                   take_average_of_pixel=False)

X_train_as_image = np.asarray(X_train_as_image)
X_validation_as_image = np.asarray(X_validation_as_image)

X_train_as_image = np.expand_dims(X_train_as_image, axis=3)
X_validation_as_image = np.expand_dims(X_validation_as_image, axis=3)

# ------------------    create and train the model    ------------------ #
log(step='constructing the NN with configurations', expect='train to be started')
model = CNN.create_model(config_resolution, config_resolution, config_num_of_classes)
history = CNN.train(model, X_train_as_image, train_x['labels'], X_validation_as_image, validation_x['labels'])
log(step='finished training, saving history', expect='that it would work')
CNN.save_history(history=history, info=imgt.create_info(config_working_dataset,
                                                        config_reduction_method,
                                                        config_kpca_kernel,
                                                        1)
                 )

log(step='finished running', expect='program to be ended')


# cont'

def experiment_types_of_kernels(df):
    kernels = ['linear', 'poly', 'rbf', 'sigmoid', 'cosine']
    kernels = ['poly']

    for kernel in kernels:
        print(f'Experiment results for kernel: {kernel}')
        # applying and plotting k-pca
        kpca_point = imgt.dimension_reduction(kernel=kernel, features_df=df)
        imgt.plot_scatter(kpca_point)
        imgt.save_scatter(kpca_point, imgt.create_info(config_working_dataset, 'kpca', f'(kernel={kernel})', 1))

        # constructing the image
        features_to_pixels = imgt.divide_to_pixels(scatter=kpca_point, resolution=config_resolution)
        pixels_to_features = imgt.to_pixel_map(features_to_pixels)
        pixels_features_heat_map = imgt.feat_count_per_pixel(resolution=config_resolution,
                                                             feat_in_pixels=features_to_pixels)
        imgt.plot_heat_map(pixels_features_heat_map, fig_size=config_resolution)

        print()

# experiment_types_of_kernels(train_x['transpose'])
