

import numpy as np
import os
import cv2
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils
from numpy import argmax
from sklearn.metrics import confusion_matrix, accuracy_score

# path to images
path = '/Users/tacobakker/machinelearning/data/small/'

HEIGHT = 32
WIDTH = 55
N_CHANNELS = 3


def get_all_categories(path_to_folders):
    categories = []
    with os.scandir(path_to_folders) as entries:
        for entry in entries:
            categories.append(entry.name)
    return categories


def get_all_paths_with_category(category_list, path_to_images):
    image_paths = []
    for k, category in enumerate(category_list):
         for f in os.listdir(path + category):
            image_paths.append([path_to_images + category + '/' + f, k])
    return image_paths


def resize_all_images(image_paths):
    all_images = []
    for imagePath in image_paths:
        image = cv2.imread(imagePath[0])
        image = cv2.resize(image, (WIDTH, HEIGHT))
        all_images.append(image)
    return all_images


def create_label_list(image_paths):
    labels = []
    for imagePath in image_paths:
        label = imagePath[1]
        labels.append(label)
    return np.array(labels)


def scale_pixels_range(all_images):
    return np.array(all_images, dtype="float") / 255.0


def create_model(number_of_categories):
    model = Sequential()
    model.add(Convolution2D(32, (2, 2), activation='relu', input_shape=(HEIGHT, WIDTH, N_CHANNELS)))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Convolution2D(32, (2, 2), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(number_of_categories, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def test_the_model(model, testX, testY):
    pred = model.predict(testX)
    predictions = argmax(pred, axis=1)
    accuracy = accuracy_score(testY, predictions)
    print("Accuracy : %.2f%%" % (accuracy * 100.0))


def main():
    list_of_all_categories = get_all_categories(path)
    list_of_image_paths = get_all_paths_with_category(list_of_all_categories, path)
    list_of_all_labels = create_label_list(list_of_image_paths)
    normalized_list_images = scale_pixels_range(resize_all_images(list_of_image_paths))
    (trainX, testX, trainY, testY) = train_test_split(normalized_list_images,
                                                      list_of_all_labels, test_size=0.2, random_state=42)
    trainY = np_utils.to_categorical(trainY, len(list_of_all_categories))
    image_model = create_model(len(list_of_all_categories))
    image_model.fit(trainX, trainY, batch_size=32, epochs=5, verbose=1)
    test_the_model(image_model, testX, testY)


if __name__ == '__main__':
    main()
