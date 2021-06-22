import cv2
import numpy as np
import os

# Less verbose tensorflow log (INFO messages are not printed)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'

import sys
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Global variables
EPOCHS = 20
IMG_WIDTH = 30
IMG_HEIGHT = 30
NUM_CATEGORIES = 43
TEST_SIZE = 0.4

def main():

    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python traffic.py data_directory [model.h5]")

    # Get image arrays and labels for all image files
    print("Loading data...")
    images, labels = load_data(sys.argv[1])
    print("Data loaded.")

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Get a compiled neural network
    model = get_model()

    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)
    
    # Evaluate neural network performance
    print("EVALUATE")
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")

def load_data(data_dir):
    """
    Load image data from directory `data_dir`.

    Assume `data_dir` has one directory named after each category, numbered
    0 through NUM_CATEGORIES - 1. Inside each category directory will be some
    number of image files.

    Return tuple `(images, labels)`. `images` should be a list of all
    of the images in the data directory, where each image is formatted as a
    numpy ndarray with dimensions IMG_WIDTH x IMG_HEIGHT x 3. `labels` should
    be a list of integer labels, representing the categories for each of the
    corresponding `images`.
    """
    images = []
    labels = []

    # Iterate through folders inside the given directory
    for folder in os.listdir(data_dir):

        # Iterate through image files inside each folder
        for imageFile in os.listdir(os.path.join(data_dir, folder)):

            # Read image file with cv2 and resize it to the specified global variable width and height
            img = cv2.imread(os.path.join(data_dir, folder, imageFile))
            newimg = cv2.resize(img,(IMG_WIDTH,IMG_HEIGHT))
            # print(newimg.shape)

            # Append each resized image and label to the tuple lists
            images.append(newimg)
            labels.append(folder)

    # Return tuple
    # print('length of image list', len(images))
    # print('length of labels list', len(labels))
    return (images, labels)

def get_model():
    """
    Returns a compiled convolutional neural network model. Assume that the
    `input_shape` of the first layer is `(IMG_WIDTH, IMG_HEIGHT, 3)`.
    The output layer should have `NUM_CATEGORIES` units, one for each category.
    """
    # We start a keras Sequential model
    model = keras.Sequential()

    # First we add an input shape as given
    model.add(keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 3)))

    # Here we add the convolutional and pooling layers
    model.add(layers.Conv2D(32, 3, activation="relu"))
    model.add(layers.Conv2D(32, 4, activation="relu"))
    model.add(layers.MaxPooling2D(3))

    # Dropout layer to avoid overfitting
    model.add(layers.Dropout(0.2))

    # Flattening layer for input to the nn
    model.add(layers.Flatten())

    # We had a hidden layer
    model.add(layers.Dense(128, activation='relu'))
    
    # Finally, we add a classification layer with NUM_CATEGORIES possibilities
    model.add(layers.Dense(NUM_CATEGORIES, activation='softmax'))

    model.summary()

    # Compile the model
    model.compile(optimizer='adam',
                  loss=keras.losses.CategoricalCrossentropy(),
                  metrics=['accuracy'])

    return model

if __name__ == "__main__":
    main()
