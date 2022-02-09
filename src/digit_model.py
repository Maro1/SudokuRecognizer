import numpy as np

import cv2

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt

import os


class DigitModel:

    def __init__(self, weights: str = None):
        # Disable eager execution for better performance
        tf.compat.v1.disable_eager_execution()

        self.model = self.get_model()

        # Use weights if provided
        if weights:
            self.model.load_weights(weights)
            self.model.build(input_shape=(None, 28, 28, 1))
            self.model.summary()

    def get_model(self) -> keras.models.Sequential:
        """
        Returns the Keras Convolutional Neural Network Model
        """
        model = keras.models.Sequential()
        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam',
                      metrics=['sparse_categorical_accuracy'])

        return model

    def train(self, epochs=20) -> None:
        """
        Trains the model for the provided amount of epochs
        """
        # Load data and reshape to fit network
        X, y = self._load_data()
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

        # Split data into train and test splits
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        # Keep track of minimum loss and history
        min_loss = float('inf')
        history = {'loss': [], 'val_loss': [],
                   'sparse_categorical_accuracy': [],
                   'val_sparse_categorical_accuracy': []}

        for i in range(epochs):
            epoch_result = self.model.fit(
                x=X_train, y=y_train, validation_data=(X_test, y_test))

            # Store history
            for metric in history.keys():
                history[metric].append(epoch_result.history[metric][-1])

            # If validation loss is minimal, store weights
            if epoch_result.history['val_loss'][-1] < min_loss:
                min_loss = epoch_result.history['val_loss'][-1]
                self.model.save_weights('model/checkpoint')

        self.model.summary()
        self.model.evaluate(x=X_test, y=y_test)
        self.plot_results(history)

    def plot_results(self, history):
        """
        Plots training results
        """
        plt.plot(history['sparse_categorical_accuracy'])
        plt.plot(history['val_sparse_categorical_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.show()

    def classify_number(self, img) -> np.ndarray:
        """
        Classifies a digit image
        """
        img = np.reshape(img, (img.shape[0], 28, 28, 1))
        result = self.model.predict(img)
        result = [np.argmax(x) for x in result]

        return result

    def _load_data(self):
        """
        Loads dataset into numpy arrays
        """
        labels = []
        images = []

        for i in range(10):
            directory = 'data_set/assets/' + str(i)

            for file_name in os.listdir(directory):
                file_path = directory + '/' + file_name

                # Read image and convert to array of floats between 0 and 1
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = np.array(img)
                img = img.astype('float32')
                img /= 255.

                labels.append(i)
                images.append(img)

        return np.array(images), np.array(labels)
