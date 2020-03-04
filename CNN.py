"""
Module containing the method and definitions needed by our CNN
"""

import os
import numpy as np
from keras.layers import Dense, Conv2D, Flatten, Dropout, BatchNormalization, \
    MaxPooling2D
from keras.models import Sequential
from keras.utils import to_categorical
import pickle


def create_model(height, width, num_of_classes):
    """
    :param height: height of image
    :param width: width of image
    :param num_of_classes: number of labels to be classified
    :return: a proper model for training
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same',
                     input_shape=(height, width, 1)))
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(BatchNormalization())
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(64, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(Conv2D(128, (3, 3), activation='relu', kernel_initializer='he_uniform', padding='same'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.4))
    model.add(Flatten())
    model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(num_of_classes, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer="adam", metrics=['accuracy'])
    print(model.summary())
    return model


def train(current_model, current_X_train, current_y_train, current_X_validation, current_y_validation):
    """
    function to train a given model
    :param current_model: model to be trained
    :param current_X_train: image-like training samples
    :param current_y_train: classification corresponding to train set
    :param current_X_validation: image-like validation samples
    :param current_y_validation: classification corresponding to validation set
    :return: history of train
    """
    history = current_model.fit(
        np.reshape(current_X_train, (current_X_train.shape[0], current_X_train.shape[1], current_X_train.shape[2], 1)),
        to_categorical(current_y_train), validation_data=(current_X_validation, to_categorical(current_y_validation)),
        shuffle=True, epochs=150, batch_size=32
    )
    return history


def predict(current_model, current_X_test):
    """
    function to predict image-like based on a trained model
    :param current_model: a trained model
    :param current_X_test: image-like to be predicted
    :return: predicted values corresponding to the given data
    """
    return current_model.predict(current_X_test)


def save_history(history, info):
    """
    function to save the history of a training

    :param history: the history object
    :param info: info about the history of this model
    """
    with open(os.path.join(os.getcwd(), 'histories', info), 'wb') as file_pi:
        pickle.dump(history.history, file_pi)
