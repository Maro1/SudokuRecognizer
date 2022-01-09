import numpy as np
import cv2
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow_datasets as tfds

from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

import os


class DigitModel:

    def __init__(self, weights: str = None):
        self.model = self.get_model()
        if weights:
            self.model.load_weights(weights)
            self.model.build(input_shape=(None, 28, 28, 1))
            self.model.summary()

    def get_model(self) -> keras.models.Sequential:
        model = keras.models.Sequential()

        model.add(keras.layers.Conv2D(32, (3, 3), activation='relu'))
        model.add(keras.layers.Conv2D(64, (3, 3), activation='relu'))
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))
        model.add(keras.layers.Dropout(0.25))
        model.add(keras.layers.Flatten())
        model.add(keras.layers.Dense(128, activation='relu'))
        model.add(keras.layers.Dropout(0.5))
        model.add(keras.layers.Dense(10, activation='softmax'))

        model.compile(loss='sparse_categorical_crossentropy',
                      optimizer='adam', metrics=['sparse_categorical_accuracy'])

        return model

    def train(self, epochs=20) -> None:
        X, y = self._load_data()
        X = np.reshape(X, (X.shape[0], X.shape[1], X.shape[2], 1))

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42)

        loss_history = []
        for i in range(epochs):
            epoch_result = self.model.fit(
                x=X_train, y=y_train, validation_data=(X_test, y_test))

            loss_history.append(epoch_result.history['loss'])
            if min(loss_history) == epoch_result.history['loss']:
                print('Minimal loss so far: ', min(loss_history))
                self.model.save_weights('model/weights.hdf')

        self.model.summary()
        self.model.evaluate(x=X_test, y=y_test)

    def classify_number(self, img) -> np.ndarray:
        """
        Generates the given amount of sequences
        """
        img = np.reshape(img, (img.shape[0], 28, 28, 1))
        result = self.model.predict(img)
        result = [np.argmax(x) for x in result]

        return result

    def _load_data(self):
        labels = []
        images = []

        for i in range(10):
            directory = 'data_set/assets/' + str(i)

            for file_name in os.listdir(directory):
                file_path = directory + '/' + file_name

                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                img = np.array(img)
                img = img.astype('float32')
                img /= 255.

                labels.append(i)
                images.append(img)

        return np.array(images), np.array(labels)


if __name__ == '__main__':
    d = DigitModel()
    d._load_data()
